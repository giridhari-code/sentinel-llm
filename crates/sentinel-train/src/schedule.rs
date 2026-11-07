//! Learning-rate schedules for Sentinel training.

/// Linear warmup then cosine decay to `min_lr_ratio * base_lr`.
#[derive(Debug, Clone)]
pub struct CosineWithWarmup {
    pub base_lr: f32,
    pub min_lr_ratio: f32,
    pub warmup_steps: u64,
    pub total_steps: u64,
}

impl CosineWithWarmup {
    pub fn new(base_lr: f32, warmup_steps: u64, total_steps: u64) -> Self {
        Self {
            base_lr,
            min_lr_ratio: 0.1,
            warmup_steps,
            total_steps: total_steps.max(1),
        }
    }

    /// LR at global step (1-based or 0-based both fine; uses `step` as completed steps).
    pub fn lr_at(&self, step: u64) -> f32 {
        if self.warmup_steps > 0 && step < self.warmup_steps {
            let t = (step + 1) as f32 / self.warmup_steps as f32;
            return self.base_lr * t;
        }
        let denom = (self.total_steps.saturating_sub(self.warmup_steps)).max(1) as f32;
        let progress = (step.saturating_sub(self.warmup_steps)) as f32 / denom;
        let progress = progress.clamp(0.0, 1.0);
        let min_lr = self.base_lr * self.min_lr_ratio;
        let cosine = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
        min_lr + (self.base_lr - min_lr) * cosine
    }
}

/// Global L2 grad clip: scale all grads if total norm > max_norm.
pub fn clip_grad_norm(grads: &mut [sentinel_core::tensor::Tensor], max_norm: f32) -> f32 {
    if max_norm <= 0.0 {
        return 0.0;
    }
    let mut total_sq = 0.0f32;
    for g in grads.iter() {
        for &x in &g.data {
            total_sq += x * x;
        }
    }
    let total = total_sq.sqrt();
    if total > max_norm {
        let scale = max_norm / (total + 1e-6);
        for g in grads.iter_mut() {
            g.scale_inplace(scale);
        }
    }
    total
}
