//! Shared numeric ops used by tape backward.

use sentinel_core::tensor::Tensor;
use sentinel_core::types::Float;
use sentinel_core::{Result, SentinelError};

/// Causal CE: logits [seq,vocab], targets length seq (targets[t] predicts from logits[t-1] or we use targets as next-token at each row 0..seq-2).
/// Here targets[i] is the label for row i (caller should pass tokens[1..] aligned with logits[..seq-1] OR full seq with last ignored).
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
