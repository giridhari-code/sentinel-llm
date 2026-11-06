//! Rotary Position Embeddings (RoPE) — Tesla-1 implementation.

use sentinel_core::tensor::Tensor;
use sentinel_core::types::Float;
use sentinel_core::Result;

/// Apply RoPE to a tensor of shape [batch, n_heads, seq_len, head_dim]
/// head_dim must be even.
pub fn apply_rope(x: &Tensor, theta: Float) -> Result<Tensor> {
    let shape = &x.shape;
    assert!(shape.len() == 4, "RoPE expects 4D tensor");
    let (_b, _h, seq_len, head_dim) = (shape[0], shape[1], shape[2], shape[3]);
    assert!(head_dim % 2 == 0);

    let mut out = x.data.clone();
    let half = head_dim / 2;

    for b in 0..shape[0] {
        for h in 0..shape[1] {
            for t in 0..seq_len {
                let base = ((b * shape[1] + h) * seq_len + t) * head_dim;
                for i in 0..half {
                    let freq = 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32);
                    let angle = t as f32 * freq;
                    let (sin, cos) = angle.sin_cos();

                    let x0 = x.data[base + 2 * i];
                    let x1 = x.data[base + 2 * i + 1];

                    out[base + 2 * i] = x0 * cos - x1 * sin;
                    out[base + 2 * i + 1] = x0 * sin + x1 * cos;
                }
            }
        }
    }

    Tensor::from_vec(out, shape)
}
