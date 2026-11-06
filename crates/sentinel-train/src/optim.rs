//! AdamW optimizer (Tesla-1).

use sentinel_core::tensor::Tensor;
use sentinel_core::types::Float;

#[derive(Debug, Clone)]
pub struct AdamWConfig {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
}

impl Default for AdamWConfig {
    fn default() -> Self {
        Self {
            lr: 3e-4,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
        }
    }
}

struct Slot {
    m: Vec<Float>,
    v: Vec<Float>,
}

/// AdamW over a list of parameter tensors (by mutable pointer identity in step order).
pub struct AdamW {
    pub cfg: AdamWConfig,
    slots: Vec<Slot>,
    step: u64,
}

impl AdamW {
    pub fn new(cfg: AdamWConfig) -> Self {
        Self {
            cfg,
            slots: Vec::new(),
            step: 0,
        }
    }

    fn ensure_slot(&mut self, idx: usize, n: usize) {
        while self.slots.len() <= idx {
            self.slots.push(Slot {
                m: Vec::new(),
                v: Vec::new(),
            });
        }
        if self.slots[idx].m.len() != n {
            self.slots[idx].m = vec![0.0; n];
            self.slots[idx].v = vec![0.0; n];
        }
    }

    /// Apply gradients: params[i] and grads[i] must align.
    pub fn step(&mut self, params: &mut [&mut Tensor], grads: &[&Tensor]) {
        assert_eq!(params.len(), grads.len());
        self.step += 1;
        let t = self.step as f32;
        let cfg = &self.cfg;
        let bc1 = 1.0 - cfg.beta1.powf(t);
        let bc2 = 1.0 - cfg.beta2.powf(t);

        for (i, (p, g)) in params.iter_mut().zip(grads.iter()).enumerate() {
            assert_eq!(p.numel(), g.numel());
            self.ensure_slot(i, p.numel());
            let slot = &mut self.slots[i];

            for j in 0..p.numel() {
                let mut gi = g.data[j];
                // decoupled weight decay
                if cfg.weight_decay != 0.0 {
                    p.data[j] *= 1.0 - cfg.lr * cfg.weight_decay;
                }
                slot.m[j] = cfg.beta1 * slot.m[j] + (1.0 - cfg.beta1) * gi;
                slot.v[j] = cfg.beta2 * slot.v[j] + (1.0 - cfg.beta2) * gi * gi;
                let mhat = slot.m[j] / bc1;
                let vhat = slot.v[j] / bc2;
                p.data[j] -= cfg.lr * mhat / (vhat.sqrt() + cfg.eps);
            }
        }
    }
}
