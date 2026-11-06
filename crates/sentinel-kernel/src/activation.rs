//! Activation functions.

use sentinel_core::tensor::Tensor;
use sentinel_core::types::Float;
use sentinel_core::Result;

/// SiLU / Swish: x * sigmoid(x)
pub fn silu(x: &Tensor) -> Result<Tensor> {
    let data: Vec<Float> = x
        .data
        .iter()
        .map(|&v| {
            let sig = 1.0 / (1.0 + (-v).exp());
            v * sig
        })
        .collect();
    Tensor::from_vec(data, &x.shape)
}

/// GELU (approximate)
pub fn gelu(x: &Tensor) -> Result<Tensor> {
    let data: Vec<Float> = x
        .data
        .iter()
        .map(|&v| {
            // tanh approximation
            let c = (2.0 / std::f32::consts::PI).sqrt();
            let u = c * (v + 0.044715 * v * v * v);
            0.5 * v * (1.0 + u.tanh())
        })
        .collect();
    Tensor::from_vec(data, &x.shape)
}
