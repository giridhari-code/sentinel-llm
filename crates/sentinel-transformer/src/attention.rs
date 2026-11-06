//! Multi-head self-attention using Tesla Kernel.

use sentinel_core::tensor::Tensor;
use sentinel_core::{Result, ModelConfig};
use sentinel_kernel::{matmul, rms_norm, apply_rope, scaled_dot_product_attention};

pub struct MultiHeadAttention {
    pub wq: Tensor,
    pub wk: Tensor,
    pub wv: Tensor,
    pub wo: Tensor,
    pub norm_weight: Tensor,
    pub n_heads: usize,
    pub d_head: usize,
    pub d_model: usize,
    pub rope_theta: f32,
    pub norm_eps: f32,
}

impl MultiHeadAttention {
    pub fn new(config: &ModelConfig) -> Self {
        let d = config.d_model;
        let std = (1.0 / d as f32).sqrt();
        Self {
            wq: Tensor::randn(&[d, d], 0.0, std),
            wk: Tensor::randn(&[d, d], 0.0, std),
            wv: Tensor::randn(&[d, d], 0.0, std),
            wo: Tensor::randn(&[d, d], 0.0, std),
            norm_weight: Tensor::ones(&[d]),
            n_heads: config.n_heads,
            d_head: config.d_head,
            d_model: d,
            rope_theta: config.rope_theta,
            norm_eps: config.norm_eps,
        }
    }

    /// x: [seq, d_model]
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x.clone();
        let x = rms_norm(x, &self.norm_weight, self.norm_eps)?;

        let seq = x.shape[0];
        let d = self.d_model;
        let h = self.n_heads;
        let dh = self.d_head;

        let q = matmul(&x, &self.wq)?;
        let k = matmul(&x, &self.wk)?;
        let v = matmul(&x, &self.wv)?;

        // Reshape to [1, heads, seq, head_dim] for Tesla-1 attention
        let q = q.reshape(&[1, h, seq, dh])?;
        let k = k.reshape(&[1, h, seq, dh])?;
        let v = v.reshape(&[1, h, seq, dh])?;

        let q = apply_rope(&q, self.rope_theta)?;
        let k = apply_rope(&k, self.rope_theta)?;

        let attn = scaled_dot_product_attention(&q, &k, &v, None)?;
        // back to [seq, d_model]
        let attn = attn.reshape(&[seq, d])?;
        let out = matmul(&attn, &self.wo)?;

        residual.add(&out)
    }
}
