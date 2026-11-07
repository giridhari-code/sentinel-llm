//! GPU context / backend selection.

use crate::device::Device;
use sentinel_core::{Result, SentinelError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuBackend {
    /// Custom Tesla CUDA kernels
    TeslaCuda,
}

#[derive(Debug, Clone)]
pub struct GpuContext {
    pub device: Device,
    pub backend: GpuBackend,
    pub device_name: String,
    pub compute_capability: Option<(u32, u32)>,
}

impl GpuContext {
    /// Initialize CUDA device `id`.
    pub fn cuda(id: u32) -> Result<Self> {
        #[cfg(feature = "cuda")]
        {
            return cuda_init(id);
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = id;
            Err(SentinelError::Kernel(
                "CUDA not enabled: rebuild with `cargo build -p sentinel-kernel --features cuda` \
                 and install CUDA toolkit. Tesla-2 kernels live in crates/sentinel-kernel/cuda/"
                    .into(),
            ))
        }
    }

    pub fn device_id(&self) -> u32 {
        match self.device {
            Device::Cuda(id) => id,
            Device::Cpu => 0,
        }
    }
}

#[cfg(feature = "cuda")]
fn cuda_init(id: u32) -> Result<GpuContext> {
    // Tesla-2: bind to real CUDA runtime here (driver API / cuda_runtime).
    // Scaffold reports device without linking third-party BLAS.
    Ok(GpuContext {
        device: Device::Cuda(id),
        backend: GpuBackend::TeslaCuda,
        device_name: format!("Tesla-2 CUDA device {}", id),
        compute_capability: None,
    })
}
