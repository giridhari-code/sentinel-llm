//! Full MoE layer: router + experts (+ optional shared expert).

use crate::expert::ExpertMLP;
use crate::router::TopKRouter;
use sentinel_core::tensor::Tensor;
use sentinel_core::{Result, ModelConfig};
use sentinel_kernel::rms_norm;

pub struct MoELayer {
    pub router: TopKRouter,
    pub experts: Vec<ExpertMLP>,
    pub shared_expert: Option<ExpertMLP>,
    pub norm_weight: Tensor,
    pub norm_eps: f32,
}

impl MoELayer {
    pub fn new(config: &ModelConfig) -> Self {
        let experts = (0..config.n_experts)
            .map(|_| ExpertMLP::new(config))
            .collect();
        let shared = if config.use_shared_expert {
            Some(ExpertMLP::new(config))
        } else {
            None
        };
        Self {
            router: TopKRouter::new(config),
            experts,
            shared_expert: shared,
            norm_weight: Tensor::ones(&[config.d_model]),
            norm_eps: config.norm_eps,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [seq, d_model]  (batch flattened for Tesla-1)
        let residual = x.clone();
        let x_norm = rms_norm(x, &self.norm_weight, self.norm_eps)?;

        let (indices, weights) = self.router.route(&x_norm)?;
        let tokens = x_norm.shape[0];
        let d = x_norm.shape[1];
        let k = self.router.top_k;

        let mut out = Tensor::zeros(&[tokens, d]);

        // Dispatch tokens to experts (naive for Tesla-1)
        for t in 0..tokens {
            let token = Tensor::from_vec(
                x_norm.data[t * d..(t + 1) * d].to_vec(),
                &[1, d],
            )?;
            for i in 0..k {
                let expert_id = indices[t][i];
                let w = weights.data[t * k + i];
                let expert_out = self.experts[expert_id].forward(&token)?;
                for j in 0..d {
                    out.data[t * d + j] += w * expert_out.data[j];
                }
            }
        }

        // Shared expert (always on)
        if let Some(ref shared) = self.shared_expert {
            let shared_out = shared.forward(&x_norm)?;
            out = out.add(&shared_out)?;
        }

        // Residual
        residual.add(&out)
    }
}
