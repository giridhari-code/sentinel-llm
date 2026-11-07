//! Training loop for Sentinel: full autograd, checkpoints, datasets.

pub mod ag_model;
pub mod checkpoint;
pub mod dataset;
pub mod loss;
pub mod optim;
pub mod schedule;
pub mod trainer;

pub use checkpoint::{load_checkpoint, save_checkpoint};
pub use dataset::{Batch, TokenDataset};
pub use loss::cross_entropy_loss;
pub use optim::{AdamW, AdamWConfig};
pub use schedule::{clip_grad_norm, CosineWithWarmup};
pub use trainer::{TrainConfig, Trainer, TrainStepStats};
