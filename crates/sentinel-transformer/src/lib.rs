//! **Sentinel** — Orbit MoE + Looped Transformer model.
//! Looped Transformer architecture for Sentinel.
//!
//! Inference path: **Prefill** (prompt) → **Decode** (autoregressive).

pub mod attention;
pub mod block;
pub mod model;
pub mod embedding;
pub mod kv_cache;
pub mod sample;

pub use model::SentinelModel;
pub use block::TransformerBlock;
pub use kv_cache::{KVCache, StepOutput};
pub use sample::{sample_token, SamplingConfig};
