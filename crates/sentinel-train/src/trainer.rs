//! Trainer: forward → CE loss → lm_head gradient → AdamW.

use crate::loss::cross_entropy_loss;
use crate::optim::{AdamW, AdamWConfig};
use log::info;
use sentinel_core::config::ModelConfig;
use sentinel_core::tensor::Tensor;
use sentinel_core::types::{Float, TokenId};
use sentinel_core::Result;
use sentinel_transformer::SentinelModel;

#[derive(Debug, Clone)]
pub struct TrainConfig {
    pub lr: f32,
    pub weight_decay: f32,
    pub max_seq_len: usize,
    /// Also update token embeddings with a simple pass-through grad estimate
    pub train_embed: bool,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            lr: 3e-4,
            weight_decay: 0.01,
            max_seq_len: 256,
            train_embed: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TrainStepStats {
    pub loss: Float,
    pub tokens: usize,
    pub step: u64,
}

pub struct Trainer {
    pub model: SentinelModel,
    pub optim: AdamW,
    pub cfg: TrainConfig,
    pub global_step: u64,
}

impl Trainer {
    pub fn new(model_cfg: ModelConfig, train_cfg: TrainConfig) -> Self {
        let optim = AdamW::new(AdamWConfig {
            lr: train_cfg.lr,
            weight_decay: train_cfg.weight_decay,
            ..AdamWConfig::default()
        });
        info!(
            "Trainer ready | lr={} wd={} train_embed={}",
            train_cfg.lr, train_cfg.weight_decay, train_cfg.train_embed
        );
        Self {
            model: SentinelModel::new(model_cfg),
            optim,
            cfg: train_cfg,
            global_step: 0,
        }
    }

    /// One causal-LM step on a token sequence (len >= 2).
    pub fn train_step(&mut self, tokens: &[TokenId]) -> Result<TrainStepStats> {
        let seq = tokens.len().min(self.cfg.max_seq_len);
        let tokens = &tokens[..seq];
        if seq < 2 {
            return Ok(TrainStepStats {
                loss: 0.0,
                tokens: 0,
                step: self.global_step,
            });
        }

        // Forward
        let logits = self.model.forward_logits(tokens)?; // [seq, vocab]
        let (loss, d_logits) = cross_entropy_loss(&logits, tokens)?;

        // Hidden states before lm_head for embedding/grad path:
        // Recompute hidden (Tesla-1: extra forward — acceptable for prototype)
        let hidden = self.forward_hidden(tokens)?; // [seq, d]

        // dL/dW_lm = hidden^T @ d_logits   → [d, vocab]
        // analytic: for each pos, outer product h[t] * d_logits[t]
        let d_model = hidden.shape[1];
        let vocab = d_logits.shape[1];
        let mut g_lm = vec![0.0f32; d_model * vocab];
        for t in 0..seq {
            let h = &hidden.data[t * d_model..(t + 1) * d_model];
            let dl = &d_logits.data[t * vocab..(t + 1) * vocab];
            for i in 0..d_model {
                let hi = h[i];
                if hi == 0.0 {
                    continue;
                }
                let row = i * vocab;
                for j in 0..vocab {
                    g_lm[row + j] += hi * dl[j];
                }
            }
        }
        let grad_lm = Tensor::from_vec(g_lm, &[d_model, vocab])?;

        // Optional embed grad: dL/dh = d_logits @ W^T, accumulate into embed rows
        if self.cfg.train_embed {
            // g_h[t] = d_logits[t] @ W_lm^T  → [d]
            // W is [d, vocab], so g_h = W * d_logits_row (matrix-vector)
            let w = &self.model.lm_head;
            for t in 0..seq - 1 {
                let tid = tokens[t] as usize;
                let dl = &d_logits.data[t * vocab..(t + 1) * vocab];
                for i in 0..d_model {
                    let mut ge = 0.0f32;
                    let wrow = i * vocab;
                    for j in 0..vocab {
                        ge += w.data[wrow + j] * dl[j];
                    }
                    // SGD-style embed update directly (scaled by lr via optim on a fake tensor is heavy);
                    // apply small direct step for embed rows
                    let idx = tid * d_model + i;
                    if idx < self.model.embed.weight.data.len() {
                        self.model.embed.weight.data[idx] -= self.cfg.lr * ge;
                    }
                }
            }
        }

        // AdamW on lm_head
        self.optim
            .step(&mut [&mut self.model.lm_head], &[&grad_lm]);

        self.global_step += 1;
        Ok(TrainStepStats {
            loss,
            tokens: seq - 1,
            step: self.global_step,
        })
    }

    fn forward_hidden(&self, tokens: &[TokenId]) -> Result<Tensor> {
        let mut x = self.model.embed.forward(tokens)?;
        for _ in 0..self.model.config.n_loops {
            for layer in &self.model.layers {
                x = layer.forward(&x)?;
            }
        }
        // match model path: pre-lm norm
        sentinel_kernel::rms_norm(
            &x,
            &self.model.final_norm_weight,
            self.model.config.norm_eps,
        )
    }

    /// Run multiple steps over a flat token stream (sliding windows).
    pub fn train_on_tokens(
        &mut self,
        data: &[TokenId],
        steps: usize,
        window: usize,
    ) -> Result<Vec<TrainStepStats>> {
        let window = window.max(2).min(self.cfg.max_seq_len);
        if data.len() < window {
            return Ok(vec![]);
        }
        let mut stats = Vec::with_capacity(steps);
        let mut offset = 0;
        for _ in 0..steps {
            if offset + window > data.len() {
                offset = 0;
            }
            let chunk = &data[offset..offset + window];
            stats.push(self.train_step(chunk)?);
            offset += window / 2; // stride
        }
        Ok(stats)
    }
}
