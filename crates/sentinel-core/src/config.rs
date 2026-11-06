//! Model configuration for Sentinel (MoE Looped Transformer).

use serde::{Deserialize, Serialize};

/// Full configuration for the Sentinel model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    // --- Architecture ---
    /// Model name
    pub name: String,

    /// Vocabulary size
    pub vocab_size: usize,

    /// Hidden / embedding dimension
    pub d_model: usize,

    /// Number of attention heads
    pub n_heads: usize,

    /// Head dimension (usually d_model / n_heads)
    pub d_head: usize,

    /// Intermediate size of FFN / expert MLP
    pub d_ff: usize,

    /// Number of layers in the base transformer block
    pub n_layers: usize,

    /// Number of times the transformer block is looped (Looped Transformer)
    pub n_loops: usize,

    // --- Mixture of Experts ---
    /// Total number of experts
    pub n_experts: usize,

    /// Number of experts activated per token (top-k)
    pub top_k: usize,

    /// Capacity factor for expert routing
    pub capacity_factor: f32,

    /// Whether to use shared expert(s)
    pub use_shared_expert: bool,

    // --- Positional & context ---
    /// Maximum sequence length
    pub max_seq_len: usize,

    /// Rope theta (if using RoPE)
    pub rope_theta: f32,

    // --- Normalization & activation ---
    /// Epsilon for RMSNorm / LayerNorm
    pub norm_eps: f32,

    /// Dropout (usually 0.0 for inference)
    pub dropout: f32,

    // --- Kernel ---
    /// Which Tesla kernel version to use
    pub kernel_version: String,
}

impl Default for ModelConfig {
    fn default() -> Self {
        // Small prototype config so it can actually run on CPU
        Self {
            name: "Sentinel-1B".to_string(),
            vocab_size: 32_000,
            d_model: 1024,
            n_heads: 16,
            d_head: 64,
            d_ff: 4096,
            n_layers: 8,
            n_loops: 4,              // Looped Transformer: same block reused
            n_experts: 8,
            top_k: 2,
            capacity_factor: 1.25,
            use_shared_expert: true,
            max_seq_len: 4096,
            rope_theta: 10_000.0,
            norm_eps: 1e-6,
            dropout: 0.0,
            kernel_version: "Tesla-1".to_string(),
        }
    }
}

impl ModelConfig {
    pub fn sentinel_small() -> Self {
        Self::default()
    }

    pub fn sentinel_medium() -> Self {
        Self {
            name: "Sentinel-3B".to_string(),
            d_model: 2048,
            n_heads: 16,
            d_head: 128,
            d_ff: 8192,
            n_layers: 12,
            n_loops: 6,
            n_experts: 16,
            top_k: 2,
            ..Self::default()
        }
    }

    /// Rough parameter count estimate (ignoring embeddings somewhat).
    pub fn estimate_params(&self) -> u64 {
        let emb = (self.vocab_size * self.d_model) as u64;
        let attn_per_layer = (4 * self.d_model * self.d_model) as u64; // QKVO
        let expert_mlp = (2 * self.d_model * self.d_ff) as u64;
        let moe_per_layer = expert_mlp * self.n_experts as u64
            + if self.use_shared_expert { expert_mlp } else { 0 };
        let layer = attn_per_layer + moe_per_layer;
        // Looped: we only store one set of weights, reused n_loops times
        emb + layer * self.n_layers as u64
    }
}
