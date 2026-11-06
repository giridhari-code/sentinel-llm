//! Sentinel Core — fundamental types, configuration, and tensor primitives.
//!
//! No third-party tensor libraries. Pure Rust implementation.

pub mod config;
pub mod error;
pub mod tensor;
pub mod types;

pub use config::ModelConfig;
pub use error::{Result, SentinelError};
pub use tensor::Tensor;
pub use types::*;
