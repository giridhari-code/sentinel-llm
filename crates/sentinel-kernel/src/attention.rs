//! Scaled Dot-Product Attention — Tesla Kernel 1 (with causal mask).

use crate::gemm::batched_matmul;
use crate::softmax::softmax;
use sentinel_core::tensor::Tensor;
use sentinel_core::types::Float;
use sentinel_core::{Result, SentinelError};

/// Apply causal mask in-place on scores with shape [..., Tq, Tk] (last two dims).
/// Query row `i` has absolute position `pos_offset + i` and may attend to keys `j <= pos`.
pub fn apply_causal_mask(scores: &mut Tensor, pos_offset: usize) -> Result<()> {
    let rank = scores.rank();
    if rank < 2 {
        return Err(SentinelError::Kernel("causal mask needs rank >= 2".into()));
    }
    let tq = scores.shape[rank - 2];
    let tk = scores.shape[rank - 1];
    let outer: usize = scores.shape[..rank - 2].iter().product();
    let neg = Float::NEG_INFINITY;

    for o in 0..outer {
        for i in 0..tq {
            let q_pos = pos_offset + i;
            let row = o * tq * tk + i * tk;
            for j in 0..tk {
                if j > q_pos {
                    scores.data[row + j] = neg;
                }
            }
        }
    }
    Ok(())
}

/// SDPA: softmax(Q K^T / sqrt(d) + causal_mask) V
///
/// Shapes:
/// - Q: [batch, heads, Tq, head_dim]
/// - K,V: [batch, heads, Tk, head_dim]
/// - Tq may differ from Tk (decode with KV cache).
///
/// `causal`: if true, apply causal mask. For decode, `pos_offset = Tk - Tq`
/// so the last query aligns with the last key.
pub fn scaled_dot_product_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: Option<Float>,
) -> Result<Tensor> {
    scaled_dot_product_attention_ex(q, k, v, scale, true)
}

/// Extended SDPA with explicit causal flag.
pub fn scaled_dot_product_attention_ex(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: Option<Float>,
    causal: bool,
) -> Result<Tensor> {
    if q.rank() != 4 || k.rank() != 4 || v.rank() != 4 {
        return Err(SentinelError::Kernel(
            "SDPA expects 4D tensors [B,H,T,D]".into(),
        ));
    }
    if q.shape[0] != k.shape[0]
        || q.shape[1] != k.shape[1]
        || q.shape[3] != k.shape[3]
        || k.shape != v.shape
    {
        return Err(SentinelError::Kernel(
            "SDPA batch/heads/dim mismatch between Q,K,V".into(),
        ));
    }

    let tq = q.shape[2];
    let tk = k.shape[2];
    let head_dim = q.shape[3] as Float;
    let scale = scale.unwrap_or(1.0 / head_dim.sqrt());

    // scores = Q @ K^T → [B, H, Tq, Tk]
    let k_t = transpose_last_two(k)?;
    let mut scores = batched_matmul(q, &k_t)?;
    scores.scale_inplace(scale);

    if causal {
        // Prefill: Tq==Tk → offset 0. Decode append: offset = Tk - Tq.
        let pos_offset = if tk >= tq { tk - tq } else { 0 };
        apply_causal_mask(&mut scores, pos_offset)?;
    }

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
