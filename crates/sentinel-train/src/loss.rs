//! Causal language-modeling cross-entropy.

use sentinel_core::tensor::Tensor;
use sentinel_core::types::{Float, TokenId};
use sentinel_core::{Result, SentinelError};

/// Mean CE over positions 0..seq-2 predicting next token.
/// logits: [seq, vocab], targets = tokens[1..]
/// Returns (loss, d_logits [seq, vocab] with grad only on rows 0..seq-2).
pub fn cross_entropy_loss(
    logits: &Tensor,
    tokens: &[TokenId],
) -> Result<(Float, Tensor)> {
    if logits.rank() != 2 {
        return Err(SentinelError::Kernel("CE expects [seq, vocab]".into()));
    }
    let seq = logits.shape[0];
    let vocab = logits.shape[1];
    if tokens.len() != seq {
        return Err(SentinelError::ShapeMismatch {
            expected: vec![seq],
            got: vec![tokens.len()],
        });
    }
    if seq < 2 {
        return Err(SentinelError::Other("need seq >= 2 for CE".into()));
    }

    let n = (seq - 1) as Float;
    let mut loss = 0.0f32;
    let mut d_logits = vec![0.0f32; seq * vocab];

    for t in 0..seq - 1 {
        let row = &logits.data[t * vocab..(t + 1) * vocab];
        let target = tokens[t + 1] as usize;
        if target >= vocab {
            return Err(SentinelError::Other(format!(
                "target id {} >= vocab {}",
                target, vocab
            )));
        }

        // stable log-softmax
        let max_v = row.iter().cloned().fold(Float::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0f32;
        for &v in row {
            sum_exp += (v - max_v).exp();
        }
        let log_z = max_v + sum_exp.ln();
        let log_prob = row[target] - log_z;
        loss -= log_prob;

        // dL/dlogit_i = softmax_i - 1_{i=target}
        let inv_n = 1.0 / n;
        for i in 0..vocab {
            let p = (row[i] - max_v).exp() / sum_exp;
            let g = if i == target { p - 1.0 } else { p };
            d_logits[t * vocab + i] = g * inv_n;
        }
    }

    loss /= n;
    let d = Tensor::from_vec(d_logits, &[seq, vocab])?;
    Ok((loss, d))
}
