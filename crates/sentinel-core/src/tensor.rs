//! Minimal pure-Rust Tensor implementation.
//! No ndarray, no candle, no tch — fully custom.

use crate::error::{Result, SentinelError};
use crate::types::Float;
use bytemuck::{cast_slice, cast_slice_mut};
use std::fmt;
use std::ops::{Add, Mul, Sub};

/// Contiguous row-major tensor.
#[derive(Clone)]
pub struct Tensor {
    pub data: Vec<Float>,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn zeros(shape: &[usize]) -> Self {
        let size: usize = shape.iter().product();
        Self {
            data: vec![0.0; size],
            shape: shape.to_vec(),
        }
    }

    pub fn ones(shape: &[usize]) -> Self {
        let size: usize = shape.iter().product();
        Self {
            data: vec![1.0; size],
            shape: shape.to_vec(),
        }
    }

    pub fn from_vec(data: Vec<Float>, shape: &[usize]) -> Result<Self> {
        let expected: usize = shape.iter().product();
        if data.len() != expected {
            return Err(SentinelError::ShapeMismatch {
                expected: shape.to_vec(),
                got: vec![data.len()],
            });
        }
        Ok(Self {
            data,
            shape: shape.to_vec(),
        })
    }

    pub fn randn(shape: &[usize], mean: Float, std: Float) -> Self {
        use rand::Rng;
        use rand_distr::{Distribution, Normal};
        let mut rng = rand::thread_rng();
        let normal = Normal::new(mean, std).unwrap();
        let size: usize = shape.iter().product();
        let data: Vec<Float> = (0..size).map(|_| normal.sample(&mut rng) as Float).collect();
        Self {
            data,
            shape: shape.to_vec(),
        }
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn reshape(&self, new_shape: &[usize]) -> Result<Self> {
        let expected: usize = new_shape.iter().product();
        if expected != self.numel() {
            return Err(SentinelError::ShapeMismatch {
                expected: new_shape.to_vec(),
                got: self.shape.clone(),
            });
        }
        Ok(Self {
            data: self.data.clone(),
            shape: new_shape.to_vec(),
        })
    }

    pub fn view(&self, new_shape: &[usize]) -> Result<TensorView<'_>> {
        let expected: usize = new_shape.iter().product();
        if expected != self.numel() {
            return Err(SentinelError::ShapeMismatch {
                expected: new_shape.to_vec(),
                got: self.shape.clone(),
            });
        }
        Ok(TensorView {
            data: &self.data,
            shape: new_shape.to_vec(),
        })
    }

    #[inline]
    pub fn as_slice(&self) -> &[Float] {
        &self.data
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [Float] {
        &mut self.data
    }

    /// Element-wise add (same shape).
    pub fn add(&self, other: &Tensor) -> Result<Tensor> {
        if self.shape != other.shape {
            return Err(SentinelError::ShapeMismatch {
                expected: self.shape.clone(),
                got: other.shape.clone(),
            });
        }
        let data: Vec<Float> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        Ok(Tensor {
            data,
            shape: self.shape.clone(),
        })
    }

    /// In-place scale.
    pub fn scale_inplace(&mut self, factor: Float) {
        for x in &mut self.data {
            *x *= factor;
        }
    }
}

/// Borrowed view over tensor data.
pub struct TensorView<'a> {
    pub data: &'a [Float],
    pub shape: Vec<usize>,
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor(shape={:?}, numel={})", self.shape, self.numel())
    }
}

impl Tensor {
    /// Concatenate two tensors along `dim`. Both must have same rank and matching sizes on other dims.
    pub fn concat(a: &Tensor, b: &Tensor, dim: usize) -> Result<Tensor> {
        if a.rank() != b.rank() {
            return Err(SentinelError::Kernel("concat: rank mismatch".into()));
        }
        if dim >= a.rank() {
            return Err(SentinelError::InvalidDim(format!("dim {} out of range", dim)));
        }
        for (i, (sa, sb)) in a.shape.iter().zip(b.shape.iter()).enumerate() {
            if i != dim && sa != sb {
                return Err(SentinelError::ShapeMismatch {
                    expected: a.shape.clone(),
                    got: b.shape.clone(),
                });
            }
        }

        let mut out_shape = a.shape.clone();
        out_shape[dim] = a.shape[dim] + b.shape[dim];

        // Generic concat via outer/inner product
        let outer: usize = a.shape[..dim].iter().product();
        let a_mid = a.shape[dim];
        let b_mid = b.shape[dim];
        let inner: usize = a.shape[dim + 1..].iter().product();

        let mut data = vec![0.0; outer * (a_mid + b_mid) * inner];
        for o in 0..outer {
            let a_src = o * a_mid * inner;
            let b_src = o * b_mid * inner;
            let dst = o * (a_mid + b_mid) * inner;
            data[dst..dst + a_mid * inner]
                .copy_from_slice(&a.data[a_src..a_src + a_mid * inner]);
            data[dst + a_mid * inner..dst + (a_mid + b_mid) * inner]
                .copy_from_slice(&b.data[b_src..b_src + b_mid * inner]);
        }
        Tensor::from_vec(data, &out_shape)
    }

    /// Slice along dim 0: rows [start, end)
    pub fn slice_seq(&self, start: usize, end: usize) -> Result<Tensor> {
        if self.rank() == 0 || end > self.shape[0] || start >= end {
            return Err(SentinelError::InvalidDim(format!(
                "slice_seq [{}, {}) on shape {:?}",
                start, end, self.shape
            )));
        }
        let inner: usize = self.shape[1..].iter().product();
        let mut out_shape = self.shape.clone();
        out_shape[0] = end - start;
        let data = self.data[start * inner..end * inner].to_vec();
        Tensor::from_vec(data, &out_shape)
    }
}
