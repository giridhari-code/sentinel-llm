//! Scaled Dot-Product Attention — Tesla Kernel 1.

use crate::gemm::batched_matmul;
use crate::softmax::softmax;
use sentinel_core::tensor::Tensor;
use sentinel_core::types::Float;
use sentinel_core::{Result, SentinelError};

/// Standard SDPA: softmax(Q K^T / sqrt(d)) V
/// Shapes: Q,K,V = [batch, heads, seq, head_dim]
pub fn scaled_dot_product_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: Option<Float>,
) -> Result<Tensor> {
    if q.shape != k.shape || q.shape.len() != 4 {
        return Err(SentinelError::Kernel(
            "Q and K must be 4D and same shape for Tesla-1 SDPA".into(),
        ));
    }

    let head_dim = q.shape[3] as Float;
    let scale = scale.unwrap_or(1.0 / head_dim.sqrt());

    // scores = Q @ K^T   → [B, H, T, T]
    // For Tesla-1 we do a simple transpose of K then matmul.
    let k_t = transpose_last_two(k)?;
    let mut scores = batched_matmul(q, &k_t)?;
    scores.scale_inplace(scale);

    let probs = softmax(&scores)?;
    let out = batched_matmul(&probs, v)?;
    Ok(out)
}

fn transpose_last_two(t: &Tensor) -> Result<Tensor> {
    let rank = t.rank();
    if rank < 2 {
        return Err(SentinelError::Kernel("rank < 2".into()));
    }
    let mut shape = t.shape.clone();
    let a = shape[rank - 2];
    let b = shape[rank - 1];
    shape[rank - 2] = b;
    shape[rank - 1] = a;

    // Naive transpose for last two dims
    let outer: usize = t.shape[..rank - 2].iter().product();
    let mut data = vec![0.0; t.numel()];

    for o in 0..outer {
        for i in 0..a {
            for j in 0..b {
                let src = o * a * b + i * b + j;
                let dst = o * b * a + j * a + i;
                data[dst] = t.data[src];
            }
        }
    }

    Tensor::from_vec(data, &shape)
}
