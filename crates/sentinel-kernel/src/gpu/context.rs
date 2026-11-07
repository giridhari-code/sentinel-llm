//! GPU context / backend selection (Tesla-2).

use crate::device::Device;
use sentinel_core::{Result, SentinelError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuBackend {
    /// Custom Tesla CUDA kernels (target path)
    TeslaCuda,
    /// Host-mirror path used when CUDA toolkit / device absent
    HostMirror,
}

#[derive(Debug, Clone)]
pub struct GpuContext {
    pub device: Device,
    pub backend: GpuBackend,
    pub device_name: String,
    pub compute_capability: Option<(u32, u32)>,
    /// True if kernels should prefer device launch over host
    pub device_ready: bool,
}

impl GpuContext {
    pub fn cuda(id: u32) -> Result<Self> {
        #[cfg(feature = "cuda")]
        {
            return cuda_init(id);
        }
        #[cfg(not(feature = "cuda"))]
        {
            // Allow creating a logical CUDA device context that runs HostMirror.
            // This lets the rest of the stack exercise the GPU API without GPU hardware.
            Ok(GpuContext {
                device: Device::Cuda(id),
                backend: GpuBackend::HostMirror,
                device_name: format!(
                    "Tesla-2 HostMirror (cuda:{} requested; rebuild with --features cuda for VRAM)",
                    id
                ),
                compute_capability: None,
                device_ready: false,
            })
        }
    }

    pub fn device_id(&self) -> u32 {
        match self.device {
            Device::Cuda(id) => id,
            Device::Cpu => 0,
        }
    }

    pub fn is_ready(&self) -> bool {
        self.device_ready
    }
}

#[cfg(feature = "cuda")]
fn cuda_init(id: u32) -> Result<GpuContext> {
    // Placeholder for cudaSetDevice / cuInit.
    // When libcudart is linked, set device_ready = true after successful init.
    Ok(GpuContext {
        device: Device::Cuda(id),
        backend: GpuBackend::TeslaCuda,
        device_name: format!("Tesla-2 CUDA device {}", id),
        compute_capability: None,
        device_ready: true,
    })
}
