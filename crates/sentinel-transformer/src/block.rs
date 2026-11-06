//! Single Transformer block = Attention + MoE.

use crate::attention::MultiHeadAttention;
use sentinel_core::tensor::Tensor;
use sentinel_core::{Result, ModelConfig};
use sentinel_moe::MoELayer;

pub struct TransformerBlock {
    pub attn: MultiHeadAttention,
    pub moe: MoELayer,
}

impl TransformerBlock {
    pub fn new(config: &ModelConfig) -> Self {
        Self {
            attn: MultiHeadAttention::new(config),
            moe: MoELayer::new(config),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.attn.forward(x)?;
        self.moe.forward(&x)
    }
}
