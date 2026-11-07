//! Evaluation harness: cross-entropy loss and perplexity.

use crate::dataset::TokenDataset;
use crate::loss::cross_entropy_loss;
use log::info;
use sentinel_core::types::{Float, TokenId};
use sentinel_core::Result;
use sentinel_transformer::SentinelModel;

/// Metrics from one eval run.
#[derive(Debug, Clone)]
pub struct EvalMetrics {
    /// Mean next-token cross-entropy (nats)
    pub loss: Float,
    /// Perplexity = exp(loss)
    pub perplexity: Float,
    /// Number of predicted tokens (sum over windows of seq_len-1)
    pub num_tokens: usize,
    /// Number of sequence windows evaluated
    pub num_windows: usize,
}

impl EvalMetrics {
    pub fn empty() -> Self {
        Self {
            loss: 0.0,
            perplexity: 1.0,
            num_tokens: 0,
            num_windows: 0,
        }
    }
}

/// Mean CE only (no grad allocation beyond loss helper).
fn window_ce(model: &SentinelModel, tokens: &[TokenId]) -> Result<(Float, usize)> {
    if tokens.len() < 2 {
        return Ok((0.0, 0));
    }
    let logits = model.forward_logits(tokens)?;
    let (loss, _d) = cross_entropy_loss(&logits, tokens)?;
    Ok((loss, tokens.len() - 1))
}

/// Evaluate causal LM on a flat token stream with sliding windows.
pub fn evaluate_tokens(
    model: &SentinelModel,
    tokens: &[TokenId],
    seq_len: usize,
    stride: Option<usize>,
) -> Result<EvalMetrics> {
    let ds = TokenDataset::from_tokens(tokens.to_vec(), seq_len, stride);
    evaluate_dataset(model, &ds)
}

/// Evaluate over all windows in a `TokenDataset`.
pub fn evaluate_dataset(model: &SentinelModel, ds: &TokenDataset) -> Result<EvalMetrics> {
    if ds.is_empty() {
        return Ok(EvalMetrics::empty());
    }

    let mut total_loss_weighted = 0.0f32;
    let mut total_tokens = 0usize;
    let mut windows = 0usize;

    for window in ds.iter() {
        let (loss, ntok) = window_ce(model, window)?;
        if ntok == 0 {
            continue;
        }
        total_loss_weighted += loss * ntok as f32;
        total_tokens += ntok;
        windows += 1;
    }

    if total_tokens == 0 {
        return Ok(EvalMetrics::empty());
    }

    let mean_loss = total_loss_weighted / total_tokens as f32;
    let ppl = mean_loss.exp();

    info!(
        "eval | windows={} tokens={} loss={:.4} ppl={:.2}",
        windows, total_tokens, mean_loss, ppl
    );

    Ok(EvalMetrics {
        loss: mean_loss,
        perplexity: ppl,
        num_tokens: total_tokens,
        num_windows: windows,
    })
}

/// Run eval every `every` train steps (helper for training loops).
pub fn should_eval(step: u64, every: u64) -> bool {
    every > 0 && step > 0 && step % every == 0
}
