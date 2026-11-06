//! Single expert MLP (SwiGLU style).

use sentinel_core::tensor::Tensor;
use sentinel_core::{Result, ModelConfig};
use sentinel_kernel::{matmul, silu};

/// One expert: gate_proj + up_proj → SiLU → down_proj
pub struct ExpertMLP {
    pub gate: Tensor, // [d_model, d_ff]
    pub up: Tensor,   // [d_model, d_ff]
    pub down: Tensor, // [d_ff, d_model]
}

impl ExpertMLP {
    pub fn new(config: &ModelConfig) -> Self {
        let d = config.d_model;
        let f = config.d_ff;
        // Xavier-ish init
        let std = (2.0 / d as f32).sqrt();
        Self {
            gate: Tensor::randn(&[d, f], 0.0, std),
            up: Tensor::randn(&[d, f], 0.0, std),
            down: Tensor::randn(&[f, d], 0.0, std),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [tokens, d_model]
        let gate = matmul(x, &self.gate)?;
        let up = matmul(x, &self.up)?;
        let activated = silu(&gate)?;
        // element-wise mul
        let hidden = activated.add(&up)?; // temporary; real SwiGLU is gate * silu(up) or similar
        // Correct SwiGLU: silu(gate) * up
        let mut swiglu = activated;
        for (a, b) in swiglu.data.iter_mut().zip(up.data.iter()) {
            *a *= *b;
        }
        matmul(&swiglu, &self.down)
    }
}
