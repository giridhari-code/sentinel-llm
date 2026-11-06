//! Top-K expert router.

use sentinel_core::tensor::Tensor;
use sentinel_core::types::{ExpertId, Float};
use sentinel_core::{Result, ModelConfig};
use sentinel_kernel::matmul;
use rand::Rng;

pub struct TopKRouter {
    pub weight: Tensor, // [d_model, n_experts]
    pub n_experts: usize,
    pub top_k: usize,
}

impl TopKRouter {
    pub fn new(config: &ModelConfig) -> Self {
        let std = (1.0 / config.d_model as f32).sqrt();
        Self {
            weight: Tensor::randn(&[config.d_model, config.n_experts], 0.0, std),
            n_experts: config.n_experts,
            top_k: config.top_k,
        }
    }

    /// Returns (expert_indices [tokens, top_k], expert_weights [tokens, top_k])
    pub fn route(&self, x: &Tensor) -> Result<(Vec<Vec<ExpertId>>, Tensor)> {
        // logits: [tokens, n_experts]
        let logits = matmul(x, &self.weight)?;
        let tokens = logits.shape[0];
        let n = self.n_experts;
        let k = self.top_k;

        let mut indices = vec![vec![0usize; k]; tokens];
        let mut weights_data = vec![0.0f32; tokens * k];

        for t in 0..tokens {
            let row = &logits.data[t * n..(t + 1) * n];
            // Simple top-k (no heap for Tesla-1)
            let mut scored: Vec<(Float, usize)> = row.iter().cloned().zip(0..n).collect();
            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

            let mut sum = 0.0f32;
            for i in 0..k {
                let (score, id) = scored[i];
                indices[t][i] = id;
                let w = score.exp(); // will normalize
                weights_data[t * k + i] = w;
                sum += w;
            }
            let inv = 1.0 / sum;
            for i in 0..k {
                weights_data[t * k + i] *= inv;
            }
        }

        let weights = Tensor::from_vec(weights_data, &[tokens, k])?;
        Ok((indices, weights))
    }
}
