//! Shared numeric ops used by tape forward/backward.

use sentinel_core::tensor::Tensor;
use sentinel_core::types::Float;
use sentinel_core::{Result, SentinelError};

pub fn softmax_cross_entropy(logits: &Tensor, targets: &[u32]) -> Result<(Float, Tensor)> {
    if logits.rank() != 2 {
        return Err(SentinelError::Kernel("CE expects [seq,vocab]".into()));
    }
    let seq = logits.shape[0];
    let vocab = logits.shape[1];
    let n = seq.min(targets.len());
    if n == 0 {
        return Err(SentinelError::Other("empty CE".into()));
    }
    let mut loss = 0.0f32;
    let mut d = vec![0.0f32; seq * vocab];
    let mut count = 0usize;
    for t in 0..n {
        if t >= seq {
            break;
        }
        let target = targets[t] as usize;
        if target >= vocab {
            continue;
        }
        let row = &logits.data[t * vocab..(t + 1) * vocab];
        let max_v = row.iter().cloned().fold(Float::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0f32;
        for &v in row {
            sum_exp += (v - max_v).exp();
        }
        let log_z = max_v + sum_exp.ln();
        loss -= row[target] - log_z;
        count += 1;
        for i in 0..vocab {
            let p = (row[i] - max_v).exp() / sum_exp;
            d[t * vocab + i] = if i == target { p - 1.0 } else { p };
        }
    }
    if count == 0 {
        return Err(SentinelError::Other("no valid CE targets".into()));
    }
    let inv = 1.0 / count as f32;
    loss *= inv;
    for x in d.iter_mut() {
        *x *= inv;
    }
    Ok((loss, Tensor::from_vec(d, &[seq, vocab])?))
}

/// Softmax over last dim. Returns probs same shape as x.
pub fn softmax_last(x: &Tensor) -> Result<Tensor> {
    let last = *x.shape.last().unwrap();
    let outer = x.numel() / last;
    let mut out = vec![0.0; x.numel()];
    for i in 0..outer {
        let start = i * last;
        let row = &x.data[start..start + last];
        let max_v = row.iter().cloned().fold(Float::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for j in 0..last {
            let e = (row[j] - max_v).exp();
            out[start + j] = e;
            sum += e;
        }
        let inv = 1.0 / sum;
        for j in 0..last {
            out[start + j] *= inv;
        }
    }
    Tensor::from_vec(out, &x.shape)
}

/// Softmax backward: dX = P * (dP - sum(dP * P)) along last dim.
pub fn softmax_backward(probs: &Tensor, d_probs: &Tensor) -> Result<Tensor> {
    let last = *probs.shape.last().unwrap();
    let outer = probs.numel() / last;
    let mut dx = vec![0.0; probs.numel()];
    for i in 0..outer {
        let start = i * last;
        let mut dot = 0.0f32;
        for j in 0..last {
            dot += d_probs.data[start + j] * probs.data[start + j];
        }
        for j in 0..last {
            dx[start + j] = probs.data[start + j] * (d_probs.data[start + j] - dot);
        }
    }
    Tensor::from_vec(dx, &probs.shape)
}

/// 2D matmul A[m,k] @ B[k,n]
pub fn matmul2(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let m = a.shape[0];
    let k = a.shape[1];
    let n = b.shape[1];
    let mut data = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0f32;
            for p in 0..k {
                s += a.data[i * k + p] * b.data[p * n + j];
            }
            data[i * n + j] = s;
        }
    }
    Tensor::from_vec(data, &[m, n])
}

/// Transpose 2D matrix
pub fn transpose2(t: &Tensor) -> Result<Tensor> {
    let (m, n) = (t.shape[0], t.shape[1]);
    let mut data = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            data[j * m + i] = t.data[i * n + j];
        }
    }
    Tensor::from_vec(data, &[n, m])
}

/// Scaled dot-product attention for 2D Q,K,V: each [seq, d]
/// scores = Q K^T / scale; probs = softmax(scores); out = probs @ V
/// Returns (out, probs) — probs cached for backward.
pub fn sdpa2(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: Float,
) -> Result<(Tensor, Tensor)> {
    let seq = q.shape[0];
    let d = q.shape[1];
    if k.shape != [seq, d] || v.shape[0] != seq {
        return Err(SentinelError::ShapeMismatch {
            expected: q.shape.clone(),
            got: k.shape.clone(),
        });
    }
    // scores [seq, seq] = Q @ K^T * scale
    let k_t = transpose2(k)?;
    let mut scores = matmul2(q, &k_t)?;
    scores.scale_inplace(scale);
    let probs = softmax_last(&scores)?;
    let out = matmul2(&probs, v)?;
    Ok((out, probs))
}

/// SDPA backward for 2D.
/// Given dout [seq,d], probs [seq,seq], Q,K,V — returns (dq, dk, dv).
pub fn sdpa2_backward(
    dout: &Tensor,
    probs: &Tensor,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: Float,
) -> Result<(Tensor, Tensor, Tensor)> {
    // dV = probs^T @ dout
    let probs_t = transpose2(probs)?;
    let dv = matmul2(&probs_t, dout)?;

    // dProbs = dout @ V^T
    let v_t = transpose2(v)?;
    let d_probs = matmul2(dout, &v_t)?;

    // dScores = softmax_backward(probs, dProbs)
    let d_scores = softmax_backward(probs, &d_probs)?;

    // dQ = dScores @ K * scale
    let mut dq = matmul2(&d_scores, k)?;
    dq.scale_inplace(scale);

    // dK = dScores^T @ Q * scale
    let d_scores_t = transpose2(&d_scores)?;
    let mut dk = matmul2(&d_scores_t, q)?;
    dk.scale_inplace(scale);

    Ok((dq, dk, dv))
}

/// Top-k over last dim per row. Returns (indices [rows][k], hard weights [rows,k] softmax-normalized).
pub fn topk_route(
    logits: &Tensor,
    top_k: usize,
) -> Result<(Vec<Vec<usize>>, Tensor, Tensor)> {
    // also returns full soft softmax for STE backward
    let rows = logits.shape[0];
    let n = logits.shape[1];
    let k = top_k.min(n);
    let soft = softmax_last(logits)?;

    let mut indices = vec![vec![0usize; k]; rows];
    let mut hard = vec![0.0f32; rows * k];

    for t in 0..rows {
        let row = &logits.data[t * n..(t + 1) * n];
        let mut scored: Vec<(Float, usize)> = row.iter().cloned().zip(0..n).collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let mut sum = 0.0f32;
        for i in 0..k {
            let (score, id) = scored[i];
            indices[t][i] = id;
            let w = score.exp();
            hard[t * k + i] = w;
            sum += w;
        }
        let inv = 1.0 / sum.max(1e-12);
        for i in 0..k {
            hard[t * k + i] *= inv;
        }
    }
    let hard_t = Tensor::from_vec(hard, &[rows, k])?;
    Ok((indices, hard_t, soft))
}
