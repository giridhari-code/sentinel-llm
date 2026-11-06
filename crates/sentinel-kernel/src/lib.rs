//! Tesla Kernel 1 — Custom pure-Rust compute kernels for Sentinel.
//!
//! Design goals:
//! - Zero third-party GPU / BLAS / FlashAttention libraries
//! - Fully auditable, owned by the project ("mere khuda ka kernel")
//! - Rayon for multi-core CPU parallelism
//! - Clear path to later SIMD / GPU backends under the same API

pub mod gemm;
pub mod attention;
pub mod norm;
pub mod activation;
pub mod softmax;
pub mod rope;

pub use gemm::matmul;
pub use attention::scaled_dot_product_attention;
pub use norm::rms_norm;
pub use activation::silu;
pub use softmax::softmax;
pub use rope::apply_rope;

/// Kernel version string
pub const KERNEL_VERSION: &str = "Tesla-1";
pub const KERNEL_NAME: &str = "Tesla";
