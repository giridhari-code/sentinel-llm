//! Token sampling: greedy, temperature, top-k, top-p (nucleus).

use sentinel_core::tensor::Tensor;
use sentinel_core::types::{Float, TokenId};
use rand::Rng;

/// Sampling configuration for decode / generate.
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    /// Softmax temperature. 0.0 or very small → greedy.
    pub temperature: f32,
    /// Nucleus sampling threshold in (0, 1]. 1.0 = disabled.
    pub top_p: f32,
    /// Keep only top-k logits before nucleus. 0 = disabled.
    pub top_k: usize,
    /// Optional hard ban list
    pub ban: Vec<TokenId>,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: 0.9,
            top_k: 0,
            ban: Vec::new(),
        }
    }
}

impl SamplingConfig {
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            ban: Vec::new(),
        }
    }

    pub fn creative() -> Self {
        Self {
            temperature: 0.8,
            top_p: 0.95,
            top_k: 50,
            ban: Vec::new(),
        }
    }
}

/// Sample one token from logits shaped [vocab].
pub fn sample_token(logits: &Tensor, cfg: &SamplingConfig) -> TokenId {
    let vocab = logits.numel();
    let mut scores: Vec<(TokenId, Float)> = logits
        .data
        .iter()
        .enumerate()
        .map(|(i, &v)| (i as TokenId, v))
        .collect();

    // Ban tokens
    for &b in &cfg.ban {
        if (b as usize) < vocab {
            scores[b as usize].1 = Float::NEG_INFINITY;
        }
    }

    // Temperature → greedy shortcut
    if cfg.temperature <= 1e-6 {
        return scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(id, _)| *id)
            .unwrap_or(0);
    }

    // Scale by temperature
    let inv_t = 1.0 / cfg.temperature;
    for s in scores.iter_mut() {
        s.1 *= inv_t;
    }

    // Top-k filter
    if cfg.top_k > 0 && cfg.top_k < vocab {
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let kth = scores[cfg.top_k.min(scores.len()) - 1].1;
        for s in scores.iter_mut() {
            if s.1 < kth {
                s.1 = Float::NEG_INFINITY;
            }
        }
    }

    // Softmax (numerically stable)
    let max_v = scores
        .iter()
        .map(|(_, v)| *v)
        .fold(Float::NEG_INFINITY, f32::max);
    let mut probs: Vec<(TokenId, Float)> = scores
        .iter()
        .map(|(id, v)| {
            let e = if v.is_finite() {
                (v - max_v).exp()
            } else {
                0.0
            };
            (*id, e)
        })
        .collect();
    let sum: Float = probs.iter().map(|(_, p)| *p).sum();
    if sum <= 0.0 {
        return 0;
    }
    for p in probs.iter_mut() {
        p.1 /= sum;
    }

    // Top-p (nucleus)
    if cfg.top_p < 1.0 - 1e-6 {
        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let mut cum = 0.0f32;
        let mut cutoff = probs.len();
        for (i, (_, p)) in probs.iter().enumerate() {
            cum += *p;
            if cum >= cfg.top_p {
                cutoff = i + 1;
                break;
            }
        }
        probs.truncate(cutoff.max(1));
        let renorm: Float = probs.iter().map(|(_, p)| *p).sum();
        for p in probs.iter_mut() {
            p.1 /= renorm;
        }
    }

    // Multinomial sample
    let mut rng = rand::thread_rng();
    let r: Float = rng.gen();
    let mut cum = 0.0f32;
    for (id, p) in &probs {
        cum += *p;
        if r <= cum {
            return *id;
        }
    }
    probs.last().map(|(id, _)| *id).unwrap_or(0)
}
