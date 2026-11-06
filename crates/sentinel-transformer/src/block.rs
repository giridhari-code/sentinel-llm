//! Single Transformer block = Attention + MoE (with KV cache support).

use crate::attention::MultiHeadAttention;
use crate::kv_cache::LayerKV;
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

    /// Training / no-cache path.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.attn.forward(x)?;
        self.moe.forward(&x)
    }

    /// Prefill: full sequence, returns output + this layer's K/V.
    pub fn forward_prefill(&self, x: &Tensor) -> Result<(Tensor, LayerKV)> {
        let (x, kv) = self.attn.forward_prefill(x)?;
        let x = self.moe.forward(&x)?;
        Ok((x, kv))
    }

    /// Decode: new tokens only, using + updating this layer's cache.
    pub fn forward_decode(&self, x: &Tensor, cache: &LayerKV) -> Result<(Tensor, LayerKV)> {
        let (x, kv) = self.attn.forward_decode(x, cache)?;
        let x = self.moe.forward(&x)?;
        Ok((x, kv))
    }
}
