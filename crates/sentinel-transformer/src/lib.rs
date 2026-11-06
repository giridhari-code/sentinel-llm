//! Looped Transformer architecture for Sentinel.
//!
//! Key idea: instead of stacking many unique layers, we have a smaller number
//! of unique layers that are applied repeatedly (looped). This reduces parameter
//! count while increasing effective depth.

pub mod attention;
pub mod block;
pub mod model;
pub mod embedding;

pub use model::SentinelModel;
pub use block::TransformerBlock;
