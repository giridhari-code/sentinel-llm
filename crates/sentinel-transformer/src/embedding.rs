use sentinel_core::tensor::Tensor;
use sentinel_core::{Result, ModelConfig};
use sentinel_core::types::TokenId;

pub struct Embedding {
    pub weight: Tensor, // [vocab, d_model]
}

impl Embedding {
    pub fn new(config: &ModelConfig) -> Self {
        let std = (1.0 / config.d_model as f32).sqrt();
        Self {
            weight: Tensor::randn(&[config.vocab_size, config.d_model], 0.0, std),
        }
    }

    pub fn forward(&self, tokens: &[TokenId]) -> Result<Tensor> {
        let d = self.weight.shape[1];
        let mut data = Vec::with_capacity(tokens.len() * d);
        for &tid in tokens {
            let start = tid as usize * d;
            data.extend_from_slice(&self.weight.data[start..start + d]);
        }
        Tensor::from_vec(data, &[tokens.len(), d])
    }
}
