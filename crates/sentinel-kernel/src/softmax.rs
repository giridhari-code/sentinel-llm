//! Softmax kernel (Tesla-1).

use sentinel_core::tensor::Tensor;
use sentinel_core::types::Float;
use sentinel_core::Result;

/// Softmax over the last dimension.
pub fn softmax(x: &Tensor) -> Result<Tensor> {
    let last_dim = *x.shape.last().unwrap();
    let outer: usize = x.numel() / last_dim;

    let mut out = vec![0.0; x.numel()];

    for i in 0..outer {
        let start = i * last_dim;
        let row = &x.data[start..start + last_dim];

        // Numerical stability
        let max_val = row.iter().cloned().fold(Float::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for (j, &v) in row.iter().enumerate() {
            let e = (v - max_val).exp();
            out[start + j] = e;
            sum += e;
        }
        let inv = 1.0 / sum;
        for j in 0..last_dim {
            out[start + j] *= inv;
        }
    }

    Tensor::from_vec(out, &x.shape)
}
