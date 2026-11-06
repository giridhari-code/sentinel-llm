//! Normalization kernels.

use sentinel_core::tensor::Tensor;
use sentinel_core::types::Float;
use sentinel_core::Result;

/// RMSNorm: x * weight / sqrt(mean(x^2) + eps)
pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: Float) -> Result<Tensor> {
    let last_dim = *x.shape.last().unwrap();
    let outer = x.numel() / last_dim;

    debug_assert_eq!(weight.numel(), last_dim);

    let mut out = vec![0.0; x.numel()];

    for i in 0..outer {
        let start = i * last_dim;
        let row = &x.data[start..start + last_dim];

        let mut sq_sum = 0.0f32;
        for &v in row {
            sq_sum += v * v;
        }
        let rms = (sq_sum / last_dim as f32 + eps).sqrt();
        let inv = 1.0 / rms;

        for j in 0..last_dim {
            out[start + j] = row[j] * inv * weight.data[j];
        }
    }

    Tensor::from_vec(out, &x.shape)
}
