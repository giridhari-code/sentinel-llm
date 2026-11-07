//! Rotary Position Embeddings (RoPE) — Tesla-1 implementation.

use sentinel_core::tensor::Tensor;
use sentinel_core::types::Float;
use sentinel_core::Result;

/// Apply RoPE to a tensor of shape [batch, n_heads, seq_len, head_dim]
/// Positions are `pos_offset .. pos_offset+seq_len`.
pub fn apply_rope(x: &Tensor, theta: Float) -> Result<Tensor> {
    apply_rope_with_offset(x, theta, 0)
}

/// Same as apply_rope but positions start at `pos_offset` (needed for decode).
pub fn apply_rope_with_offset(x: &Tensor, theta: Float, pos_offset: usize) -> Result<Tensor> {
    let shape = &x.shape;
    assert!(shape.len() == 4, "RoPE expects 4D tensor [B,H,T,D]");
    let (_b, _h, seq_len, head_dim) = (shape[0], shape[1], shape[2], shape[3]);
    assert!(head_dim % 2 == 0, "head_dim must be even for RoPE");

    let mut out = x.data.clone();
    let half = head_dim / 2;

    for b in 0..shape[0] {
        for h in 0..shape[1] {
            for t in 0..seq_len {
                let pos = (pos_offset + t) as f32;
                let base = ((b * shape[1] + h) * seq_len + t) * head_dim;
                for i in 0..half {
                    let freq = 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32);
                    let angle = pos * freq;
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

/// Apply RoPE to packed multi-head tensor [seq, n_heads * d_head] (layout: head-major within each token).
pub fn apply_rope_packed(
    x: &Tensor,
    n_heads: usize,
    d_head: usize,
    theta: Float,
    pos_offset: usize,
) -> Result<Tensor> {
    let seq = x.shape[0];
    let d = x.shape[1];
    assert!(n_heads * d_head == d, "d_model must equal n_heads * d_head");
    // view as [1, H, T, Dh]
    // Explicit pack [seq, H*Dh] → [1,H,T,Dh] then RoPE then pack back.
    let mut data4 = vec![0.0; 1 * n_heads * seq * d_head];
    for t in 0..seq {
        for h in 0..n_heads {
            for i in 0..d_head {
                let src = t * d + h * d_head + i;
                let dst = ((0 * n_heads + h) * seq + t) * d_head + i;
                data4[dst] = x.data[src];
            }
        }
    }
    let x4 = Tensor::from_vec(data4, &[1, n_heads, seq, d_head])?;
    let y4 = apply_rope_with_offset(&x4, theta, pos_offset)?;
    // pack back to [seq, H*Dh]
    let mut out = vec![0.0; seq * d];
    for t in 0..seq {
        for h in 0..n_heads {
            for i in 0..d_head {
                let src = ((0 * n_heads + h) * seq + t) * d_head + i;
                let dst = t * d + h * d_head + i;
                out[dst] = y4.data[src];
            }
        }
    }
    Tensor::from_vec(out, &[seq, d])
}

/// RoPE backward for packed layout = RoPE with negated angles (orthogonal).
pub fn apply_rope_packed_backward(
    go: &Tensor,
    n_heads: usize,
    d_head: usize,
    theta: Float,
    pos_offset: usize,
) -> Result<Tensor> {
    // R(-θ) via conjugating: apply rope with theta but we need -sin.
    // Implement by applying packed rope with modified approach: call forward with
    // position angles negated by using a dedicated inverse.
    apply_rope_packed_inverse(go, n_heads, d_head, theta, pos_offset)
}

fn apply_rope_packed_inverse(
    x: &Tensor,
    n_heads: usize,
    d_head: usize,
    theta: Float,
    pos_offset: usize,
) -> Result<Tensor> {
    let seq = x.shape[0];
    let d = x.shape[1];
    let mut data4 = vec![0.0; n_heads * seq * d_head];
    for t in 0..seq {
        for h in 0..n_heads {
            for i in 0..d_head {
                let src = t * d + h * d_head + i;
                let dst = (h * seq + t) * d_head + i;
                data4[dst] = x.data[src];
            }
        }
    }
    let mut out4 = data4.clone();
    let half = d_head / 2;
    for h in 0..n_heads {
        for t in 0..seq {
            let pos = (pos_offset + t) as f32;
            let base = (h * seq + t) * d_head;
            for i in 0..half {
                let freq = 1.0 / theta.powf(2.0 * i as f32 / d_head as f32);
                let angle = pos * freq;
                let (sin, cos) = angle.sin_cos();
                // R^T = R(-θ): [cos sin; -sin cos] applied to (x0,x1)
                let x0 = data4[base + 2 * i];
                let x1 = data4[base + 2 * i + 1];
                out4[base + 2 * i] = x0 * cos + x1 * sin;
                out4[base + 2 * i + 1] = -x0 * sin + x1 * cos;
            }
        }
    }
    let mut out = vec![0.0; seq * d];
    for t in 0..seq {
        for h in 0..n_heads {
            for i in 0..d_head {
                let src = (h * seq + t) * d_head + i;
                let dst = t * d + h * d_head + i;
                out[dst] = out4[src];
            }
        }
    }
    Tensor::from_vec(out, &[seq, d])
}
