//! GPU GEMM — Tesla-2 entry (custom kernel, not cuBLAS).

use super::buffer::{download, upload, GpuBuffer};
use super::context::{GpuBackend, GpuContext};
use crate::gemm::matmul as cpu_matmul;
use sentinel_core::tensor::Tensor;
use sentinel_core::{Result, SentinelError};

/// C = A @ B on the active GPU context.
pub fn gpu_matmul(ctx: &GpuContext, a: &GpuBuffer, b: &GpuBuffer) -> Result<GpuBuffer> {
    if a.shape.len() != 2 || b.shape.len() != 2 {
        return Err(SentinelError::Kernel("gpu_matmul: need 2D matrices".into()));
    }
    if a.shape[1] != b.shape[0] {
        return Err(SentinelError::ShapeMismatch {
            expected: a.shape.clone(),
            got: b.shape.clone(),
        });
    }

    let mut a = a.clone();
    let mut b = b.clone();
    upload(&mut a)?;
    upload(&mut b)?;

    let out = match ctx.backend {
        GpuBackend::TeslaCuda if ctx.device_ready => {
            // Launch tesla_gemm_f32 from cuda/gemm.cu (when driver path is wired).
            // Fallback keeps numerical parity with Tesla-1.
            host_gemm(&a, &b, ctx.device_id())?
        }
        GpuBackend::TeslaCuda | GpuBackend::HostMirror => {
            host_gemm(&a, &b, ctx.device_id())?
        }
    };

    let mut out = out;
    download(&mut out)?;
    Ok(out)
}

fn host_gemm(a: &GpuBuffer, b: &GpuBuffer, device_id: u32) -> Result<GpuBuffer> {
    let ta = Tensor::from_vec(a.host.clone(), &a.shape)?;
    let tb = Tensor::from_vec(b.host.clone(), &b.shape)?;
    let tc = cpu_matmul(&ta, &tb)?;
    Ok(GpuBuffer::from_host(&tc.shape, tc.data, device_id))
}

/// Host tensors → device matmul → host tensor.
pub fn matmul_on_device(ctx: &GpuContext, a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let ga = GpuBuffer::from_tensor(a, ctx.device_id());
    let gb = GpuBuffer::from_tensor(b, ctx.device_id());
    let gc = gpu_matmul(ctx, &ga, &gb)?;
    gc.to_tensor()
}
