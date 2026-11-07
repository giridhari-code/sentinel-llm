//! **Tesla** — Orbit compute kernel (CPU Tesla-1 / CUDA Tesla-2).
//!
//! - **Tesla-1**: CPU + Rayon (default)
//! - **Tesla-2**: GPU backend (`set_device(Cuda)`, optional `--features cuda`)
//!
//! No third-party GPU math libraries (no cuBLAS / FlashAttention).

pub mod gemm;
pub mod attention;
pub mod norm;
pub mod activation;
pub mod softmax;
pub mod rope;
pub mod device;
pub mod gpu;

pub use gemm::matmul;
pub use attention::{
    scaled_dot_product_attention, scaled_dot_product_attention_ex, apply_causal_mask,
};
pub use norm::rms_norm;
pub use activation::silu;
pub use softmax::softmax;
pub use rope::{
    apply_rope, apply_rope_with_offset, apply_rope_packed, apply_rope_packed_backward,
};
pub use device::Device;
pub use gpu::{
    active_device, create_context, cuda_feature_enabled, gpu_smoke, matmul_auto, set_device,
    GpuBuffer, GpuContext,
};

pub const KERNEL_VERSION: &str = if cfg!(feature = "cuda") {
    "Tesla-2"
} else {
    "Tesla-1"
};
pub const KERNEL_NAME: &str = "Tesla";
