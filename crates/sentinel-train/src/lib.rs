//! Training loop for Sentinel.
//!
//! Tesla-1 scope:
//! - Cross-entropy loss (causal LM)
//! - Analytic gradient through lm_head (softmax CE)
//! - AdamW optimizer on registered parameter tensors
//! - Trainable: lm_head (+ optional embed)
//!
//! Full transformer backprop lands in a later Tesla version.

pub mod loss;
pub mod optim;
pub mod trainer;

pub use loss::cross_entropy_loss;
pub use optim::{AdamW, AdamWConfig};
pub use trainer::{TrainConfig, Trainer, TrainStepStats};
