//! General Matrix Multiply (GEMM) — core of Tesla Kernel.
//!
//! Pure Rust + Rayon. No BLAS, no cuBLAS, no OpenBLAS.

use rayon::prelude::*;
use sentinel_core::tensor::Tensor;
use sentinel_core::types::Float;
use sentinel_core::{Result, SentinelError};

/// C = A @ B  (row-major)
/// A: [M, K], B: [K, N] → C: [M, N]
pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if a.rank() != 2 || b.rank() != 2 {
        return Err(SentinelError::Kernel(
            "matmul currently supports only 2D tensors".into(),
        ));
    }
    let m = a.shape[0];
    let k = a.shape[1];
    let k2 = b.shape[0];
    let n = b.shape[1];

    if k != k2 {
        return Err(SentinelError::ShapeMismatch {
            expected: vec![m, k],
            got: vec![k2, n],
        });
    }

    let mut c_data = vec![0.0; m * n];

    // Parallel over rows of A / C
    c_data
        .par_chunks_mut(n)
        .enumerate()
        .for_each(|(i, row)| {
            let a_row = &a.data[i * k..(i + 1) * k];
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a_row[p] * b.data[p * n + j];
                }
                row[j] = sum;
            }
        });

    Tensor::from_vec(c_data, &[m, n])
}

/// Batched matmul for attention: [B, H, T, D] @ [B, H, D, S] → [B, H, T, S]
/// Simplified version for Tesla-1 (will be optimized later).
pub fn batched_matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    // For Tesla-1 we flatten the batch+head dimensions and call matmul.
    // Later versions will have specialized kernels.
    if a.rank() < 2 || b.rank() < 2 {
        return Err(SentinelError::Kernel("batched_matmul rank error".into()));
    }

    let a_rows: usize = a.shape[..a.rank() - 1].iter().product();
    let a_k = a.shape[a.rank() - 1];
    let b_k = b.shape[b.rank() - 2];
    let b_cols = b.shape[b.rank() - 1];

    if a_k != b_k {
        return Err(SentinelError::ShapeMismatch {
            expected: a.shape.clone(),
            got: b.shape.clone(),
        });
    }

    let a2 = a.reshape(&[a_rows, a_k])?;
    let b2 = b.reshape(&[b_k, b_cols])?;
    let c2 = matmul(&a2, &b2)?;

    let mut out_shape = a.shape[..a.rank() - 1].to_vec();
    out_shape.push(b_cols);
    c2.reshape(&out_shape)
}
