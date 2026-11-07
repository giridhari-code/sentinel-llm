//! Device-aware op dispatch (CPU Tesla-1 vs GPU Tesla-2).

use super::context::GpuContext;
use super::gemm::matmul_on_device;
use crate::device::Device;
use crate::gemm::matmul as cpu_matmul;
use sentinel_core::tensor::Tensor;
use sentinel_core::{Result, SentinelError};
use std::sync::RwLock;

static ACTIVE_DEVICE: RwLock<Device> = RwLock::new(Device::Cpu);
static GPU_CTX: RwLock<Option<GpuContext>> = RwLock::new(None);

/// Set global compute device for kernel dispatch.
pub fn set_device(device: Device) -> Result<()> {
    match device {
        Device::Cpu => {
            if let Ok(mut g) = GPU_CTX.write() {
                *g = None;
            }
            if let Ok(mut d) = ACTIVE_DEVICE.write() {
                *d = Device::Cpu;
            }
            log::info!("Tesla dispatch → CPU (Tesla-1)");
            Ok(())
        }
        Device::Cuda(id) => {
            let ctx = GpuContext::cuda(id)?;
            log::info!("Tesla dispatch → {} ({:?})", ctx.device_name, ctx.backend);
            if let Ok(mut g) = GPU_CTX.write() {
                *g = Some(ctx);
            }
            if let Ok(mut d) = ACTIVE_DEVICE.write() {
                *d = Device::Cuda(id);
            }
            Ok(())
        }
    }
}

pub fn active_device() -> Device {
    ACTIVE_DEVICE
        .read()
        .map(|d| *d)
        .unwrap_or(Device::Cpu)
}

/// Matmul routed by active device.
pub fn matmul_auto(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let device = active_device();
    match device {
        Device::Cpu => cpu_matmul(a, b),
        Device::Cuda(_) => {
            let guard = GPU_CTX.read().map_err(|_| {
                SentinelError::Kernel("gpu context lock poisoned".into())
            })?;
            let ctx = guard.as_ref().ok_or_else(|| {
                SentinelError::Kernel("CUDA device set but context missing".into())
            })?;
            matmul_on_device(ctx, a, b)
        }
    }
}

/// Smoke: small GEMM on the active device; returns (device label, max abs error vs CPU).
pub fn gpu_smoke() -> Result<(String, f32)> {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
    let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2])?;
    let cpu = cpu_matmul(&a, &b)?;
    let out = matmul_auto(&a, &b)?;
    let mut max_err = 0.0f32;
    for (x, y) in cpu.data.iter().zip(out.data.iter()) {
        max_err = max_err.max((x - y).abs());
    }
    Ok((active_device().to_string(), max_err))
}
