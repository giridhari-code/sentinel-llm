//! Tesla-2 GPU backend — CUDA driver PTX launch + HostMirror fallback.
//!
//! Custom kernels only (`cuda/*.cu` / embedded PTX). No cuBLAS / FlashAttention.

mod buffer;
mod context;
mod cuda_rt;
mod dispatch;
mod gemm;

pub use buffer::{download, upload, GpuBuffer};
pub use context::{cuda_launch_active, GpuBackend, GpuContext};
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

pub fn cuda_driver_present() -> bool {
    cuda_rt::CudaLaunch::available()
}
