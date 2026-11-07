//! Tesla-2 GPU backend.
//!
//! Design:
//! - Own CUDA kernels (see `cuda/gemm.cu`) — not cuBLAS / FlashAttention
//! - Optional compile: `--features cuda`
//! - Without the feature, GPU APIs return a clear error

mod buffer;
mod context;
mod gemm;

pub use buffer::GpuBuffer;
pub use context::{GpuBackend, GpuContext};
pub use gemm::gpu_matmul;

use crate::device::Device;
use sentinel_core::{Result, SentinelError};

/// Create a GPU context for `device`, or error if unsupported / feature off.
pub fn create_context(device: Device) -> Result<GpuContext> {
    match device {
        Device::Cpu => Err(SentinelError::Kernel(
            "create_context: CPU is not a GPU device".into(),
        )),
        Device::Cuda(id) => GpuContext::cuda(id),
    }
}

/// True if this build was compiled with Tesla-2 CUDA support.
pub fn cuda_feature_enabled() -> bool {
    cfg!(feature = "cuda")
}
