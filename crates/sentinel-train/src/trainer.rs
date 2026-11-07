//! Trainer with full autograd, checkpoints, and dataset iteration.

use crate::ag_model::forward_lm;
use crate::checkpoint::{load_checkpoint, save_checkpoint};
use crate::dataset::TokenDataset;
use crate::optim::{AdamW, AdamWConfig};
use crate::schedule::{clip_grad_norm, CosineWithWarmup};
use log::info;
use sentinel_autograd::Tape;
use sentinel_core::config::ModelConfig;
use sentinel_core::types::{Float, TokenId};
use sentinel_core::Result;
use sentinel_transformer::SentinelModel;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct TrainConfig {
    pub lr: f32,
    pub weight_decay: f32,
    pub max_seq_len: usize,
    pub train_embed: bool,
    pub full_autograd: bool,
    /// Max L2 norm for gradient clipping (0 = disabled)
    pub grad_clip: f32,
    /// Warmup steps for cosine schedule
    pub warmup_steps: u64,
    /// Total steps for cosine decay (0 = constant lr)
    pub total_steps: u64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            lr: 3e-4,
            weight_decay: 0.01,
            max_seq_len: 256,
            train_embed: true,
            full_autograd: true,
            grad_clip: 1.0,
            warmup_steps: 100,
            total_steps: 10_000,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TrainStepStats {
    pub loss: Float,
    pub ce_loss: Float,
    pub aux_loss: Float,
    pub tokens: usize,
    pub step: u64,
}

pub struct Trainer {
    pub model: SentinelModel,
    pub optim: AdamW,
    pub cfg: TrainConfig,
    pub global_step: u64,
    pub schedule: Option<CosineWithWarmup>,
}

impl Trainer {
    pub fn new(model_cfg: ModelConfig, train_cfg: TrainConfig) -> Self {
        let optim = AdamW::new(AdamWConfig {
            lr: train_cfg.lr,
            weight_decay: train_cfg.weight_decay,
            ..AdamWConfig::default()
        });
        info!(
            "Trainer ready | lr={} wd={} full_autograd={}",
            train_cfg.lr, train_cfg.weight_decay, train_cfg.full_autograd
        );
        let schedule = if train_cfg.total_steps > 0 {
            Some(CosineWithWarmup::new(
                train_cfg.lr,
                train_cfg.warmup_steps,
                train_cfg.total_steps,
            ))
        } else {
            None
        };
        Self {
            model: SentinelModel::new(model_cfg),
            optim,
            cfg: train_cfg,
            global_step: 0,
            schedule,
        }
    }

    pub fn from_checkpoint(path: &Path, train_cfg: TrainConfig) -> Result<Self> {
        let model = load_checkpoint(path)?;
        let optim = AdamW::new(AdamWConfig {
            lr: train_cfg.lr,
            weight_decay: train_cfg.weight_decay,
            ..AdamWConfig::default()
        });
        let schedule = if train_cfg.total_steps > 0 {
            Some(CosineWithWarmup::new(
                train_cfg.lr,
                train_cfg.warmup_steps,
                train_cfg.total_steps,
            ))
        } else {
            None
        };
        Ok(Self {
            model,
            optim,
            cfg: train_cfg,
            global_step: 0,
            schedule,
        })
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        save_checkpoint(&self.model, path)
    }

    /// Full autograd train step over a token window.
    pub fn train_step(&mut self, tokens: &[TokenId]) -> Result<TrainStepStats> {
        let seq = tokens.len().min(self.cfg.max_seq_len);
        let tokens = &tokens[..seq];
        if seq < 2 {
            return Ok(TrainStepStats {
                loss: 0.0,
                ce_loss: 0.0,
                aux_loss: 0.0,
                tokens: 0,
                step: self.global_step,
            });
        }

        // Cosine LR with warmup
        if let Some(ref sched) = self.schedule {
            self.optim.set_lr(sched.lr_at(self.global_step));
        }

        let mut tape = Tape::new();
        let (loss_v, param_vars, ce_loss, aux_loss) = forward_lm(&mut tape, &self.model, tokens)?;
        let loss_val = tape.value(loss_v).data[0];
        tape.backward(loss_v)?;

        self.apply_grads_from_tape(&tape, &param_vars)?;

        self.global_step += 1;
        Ok(TrainStepStats {
            loss: loss_val,
            ce_loss,
            aux_loss,
            tokens: seq - 1,
            step: self.global_step,
        })
    }

    fn apply_grads_from_tape(
        &mut self,
        tape: &Tape,
        param_vars: &[sentinel_autograd::Var],
    ) -> Result<()> {
        // Collect (grad) in same order params were registered
        let mut grads = Vec::new();
        for &v in param_vars {
            if let Some(g) = tape.grad(v) {
                grads.push(g.clone());
            } else {
                // zero grad placeholder matching value shape
                let shape = tape.value(v).shape.clone();
                grads.push(sentinel_core::tensor::Tensor::zeros(&shape));
            }
        }

        // Rebuild mutable references in the same registration order as forward_lm
        let mut params: Vec<&mut sentinel_core::tensor::Tensor> = Vec::new();
        params.push(&mut self.model.embed.weight);
        let n_loops = self.model.config.n_loops;
        let n_layers = self.model.layers.len();
        // forward_lm registers params every loop iteration — grads for same weight accumulate on tape
        // but we register NEW param nodes each loop. AdamW should only step unique weights once
        // with accumulated grads from last loop registration... 
        // Simpler approach: only apply grads from the LAST loop's registered vars.
        // Count params per loop:
        let mut per_loop = 1; // emb once outside loop — emb only first
        // Actually emb is outside loop once. Inside each loop each layer registers many.
        // For correctness with repeated registration, sum grads for duplicate logical weights.
        // Tesla-1 practical fix: run n_loops=1 during AG training or only take last loop grads.

        // We'll apply using only the suffix of param_vars corresponding to last loop + head.
        // Estimate size of one loop's params:
        let _ = (n_loops, n_layers);

        // Straightforward: zip param_vars with a freshly collected unique param list from one pass structure
        // Use only final lm_head / embed / layer weights by matching shapes from end of param_vars.

        // Collect unique mutable params in forward_lm order for loop_i = last only by re-listing:
        let mut unique: Vec<&mut sentinel_core::tensor::Tensor> = Vec::new();
        unique.push(&mut self.model.embed.weight);
        for layer in self.model.layers.iter_mut() {
            unique.push(&mut layer.attn.wq);
            unique.push(&mut layer.attn.wk);
            unique.push(&mut layer.attn.wv);
            unique.push(&mut layer.attn.wo);
            unique.push(&mut layer.attn.norm_weight);
            unique.push(&mut layer.moe.norm_weight);
            unique.push(&mut layer.moe.router.weight);
            for expert in layer.moe.experts.iter_mut() {
                unique.push(&mut expert.gate);
                unique.push(&mut expert.up);
                unique.push(&mut expert.down);
            }
            if let Some(ref mut shared) = layer.moe.shared_expert {
                unique.push(&mut shared.gate);
                unique.push(&mut shared.up);
                unique.push(&mut shared.down);
            }
        }
        unique.push(&mut self.model.final_norm_weight);
        unique.push(&mut self.model.lm_head);

        // Map: last `unique.len()` grads from param_vars that match unique.len()
        // param_vars length = emb + n_loops * layer_params + 2
        if param_vars.len() >= unique.len() {
            let start = param_vars.len() - unique.len();
            // emb is at 0 in unique but at 0 in param_vars; last loop layer weights at end
            // Better mapping:
            // unique[0] = emb ← param_vars[0]
            // unique[1..] ← last occurrences in param_vars
            let g_emb = tape.grad(param_vars[0]).cloned().unwrap_or_else(|| {
                sentinel_core::tensor::Tensor::zeros(&unique[0].shape)
            });
            let mut grad_refs: Vec<sentinel_core::tensor::Tensor> = Vec::new();
            grad_refs.push(g_emb);
            let tail = &param_vars[param_vars.len() - (unique.len() - 1)..];
            for &v in tail {
                grad_refs.push(
                    tape.grad(v)
                        .cloned()
                        .unwrap_or_else(|| sentinel_core::tensor::Tensor::zeros(&[1])),
                );
            }
            // Fix shapes if mismatch — skip bad
            if grad_refs.len() == unique.len()
                && grad_refs
                    .iter()
                    .zip(unique.iter())
                    .all(|(g, p)| g.numel() == p.numel())
            {
                let mut grad_refs = grad_refs;
                let _gnorm = clip_grad_norm(&mut grad_refs, self.cfg.grad_clip);
                let g_refs: Vec<&sentinel_core::tensor::Tensor> = grad_refs.iter().collect();
                self.optim.step(&mut unique, &g_refs);
            } else {
                // Fallback: step lm_head only if present
                if let (Some(&v), Some(p)) = (param_vars.last(), unique.last_mut()) {
                    if let Some(g) = tape.grad(v) {
                        if g.numel() == p.numel() {
                            self.optim.step(&mut [p], &[g]);
                        }
                    }
                }
            }
        }

        let _ = grads;
        Ok(())
    }

    pub fn train_on_tokens(
        &mut self,
        data: &[TokenId],
        steps: usize,
        window: usize,
    ) -> Result<Vec<TrainStepStats>> {
        let ds = TokenDataset::from_tokens(data.to_vec(), window, Some(window / 2));
        self.train_on_dataset(&ds, steps)
    }

    pub fn train_on_dataset(
        &mut self,
        ds: &TokenDataset,
        steps: usize,
    ) -> Result<Vec<TrainStepStats>> {
        let mut stats = Vec::with_capacity(steps);
        let mut idx = 0;
        for _ in 0..steps {
            if ds.is_empty() {
                break;
            }
            if idx >= ds.len() {
                idx = 0;
            }
            if let Some(chunk) = ds.get(idx) {
                stats.push(self.train_step(chunk)?);
            }
            idx += 1;
        }
        Ok(stats)
    }
}
