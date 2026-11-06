//! Training infrastructure for Sentinel (skeleton).

use sentinel_core::config::ModelConfig;
use sentinel_transformer::SentinelModel;
use log::info;

pub struct Trainer {
    pub model: SentinelModel,
    pub lr: f32,
}

impl Trainer {
    pub fn new(config: ModelConfig, lr: f32) -> Self {
        Self {
            model: SentinelModel::new(config),
            lr,
        }
    }

    /// Placeholder training step — full optimizer (AdamW etc.) will come later.
    pub fn train_step(&mut self, _tokens: &[u32], _targets: &[u32]) {
        info!("train_step called (optimizer not yet implemented in Tesla-1)");
        // TODO: forward → loss → backward → update
    }
}
