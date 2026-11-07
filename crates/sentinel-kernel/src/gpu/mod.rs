//! Tesla-2 GPU backend.
//!
//! - Own CUDA kernels in `cuda/` (not cuBLAS / FlashAttention)
//! - `HostMirror` path exercises the full API without a GPU
//! - `--features cuda` enables TeslaCuda backend flag for real VRAM path

mod buffer;
mod context;
mod dispatch;
mod gemm;

pub use buffer::{download, upload, GpuBuffer};
pub use context::{GpuBackend, GpuContext};
pub use dispatch::{active_device, gpu_smoke, matmul_auto, set_device};
pub use gemm::{gpu_matmul, matmul_on_device};

use crate::device::Device;
use sentinel_core::{Result, SentinelError};

pub fn create_context(device: Device) -> Result<GpuContext> {
    match device {
        Device::Cpu => Err(SentinelError::Kernel(
            "create_context: CPU is not a GPU device".into(),
        )),
        Device::Cuda(id) => GpuContext::cuda(id),
    }
}

pub fn cuda_feature_enabled() -> bool {
    cfg!(feature = "cuda")
}
