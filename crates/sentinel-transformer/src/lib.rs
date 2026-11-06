//! Looped Transformer architecture for Sentinel.
//!
//! Inference path: **Prefill** (prompt) → **Decode** (autoregressive).

pub mod attention;
pub mod block;
pub mod model;
pub mod embedding;
pub mod kv_cache;

pub use model::SentinelModel;
pub use block::TransformerBlock;
pub use kv_cache::{KVCache, StepOutput};
