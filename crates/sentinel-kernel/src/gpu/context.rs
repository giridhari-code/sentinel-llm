//! GPU context / backend selection (Tesla-2).

use crate::device::Device;
use super::cuda_rt::{try_cuda_launch, CudaLaunch};
use sentinel_core::{Result, SentinelError};
use std::sync::{Mutex, OnceLock};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuBackend {
    /// Real CUDA driver + PTX kernel launch
    TeslaCuda,
    /// Host-mirror path when CUDA driver / device absent
    HostMirror,
}

#[derive(Debug, Clone)]
pub struct GpuContext {
    pub device: Device,
    pub backend: GpuBackend,
    pub device_name: String,
    pub compute_capability: Option<(u32, u32)>,
    pub device_ready: bool,
}

static CUDA_LAUNCH: OnceLock<Mutex<Option<CudaLaunch>>> = OnceLock::new();

fn launch_slot() -> &'static Mutex<Option<CudaLaunch>> {
    CUDA_LAUNCH.get_or_init(|| Mutex::new(None))
}

impl GpuContext {
    pub fn cuda(id: u32) -> Result<Self> {
        // Prefer real CUDA driver launch path
        if let Some(launch) = try_cuda_launch(id) {
            if let Ok(mut slot) = launch_slot().lock() {
                *slot = Some(launch);
            }
            return Ok(GpuContext {
                device: Device::Cuda(id),
                backend: GpuBackend::TeslaCuda,
                device_name: format!("Tesla-2 CUDA device {} (PTX launch)", id),
                compute_capability: None,
                device_ready: true,
            });
        }

        // Fallback: HostMirror so API still works without GPU
        Ok(GpuContext {
            device: Device::Cuda(id),
            backend: GpuBackend::HostMirror,
            device_name: format!(
                "Tesla-2 HostMirror (cuda:{} — driver/PTX launch unavailable)",
                id
            ),
            compute_capability: None,
            device_ready: false,
        })
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

/// Run GEMM on the live CUDA launch path if present.
pub fn cuda_gemm_f32(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Result<Vec<f32>> {
    let slot = launch_slot()
        .lock()
        .map_err(|_| SentinelError::Kernel("CUDA launch lock poisoned".into()))?;
    let launch = slot
        .as_ref()
        .ok_or_else(|| SentinelError::Kernel("no CUDA launch context".into()))?;
    launch.gemm_f32(a, b, m, n, k)
}

pub fn cuda_launch_active() -> bool {
    launch_slot()
        .lock()
        .map(|g| g.is_some())
        .unwrap_or(false)
}
