//! GPU GEMM — Tesla-2 entry (custom PTX kernel, not cuBLAS).

use super::buffer::{download, upload, GpuBuffer};
use super::context::{cuda_gemm_f32, cuda_launch_active, GpuBackend, GpuContext};
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

    let m = a.shape[0];
    let k = a.shape[1];
    let n = b.shape[1];

    let mut a = a.clone();
    let mut b = b.clone();
    upload(&mut a)?;
    upload(&mut b)?;

    let out_host = if ctx.backend == GpuBackend::TeslaCuda
        && ctx.device_ready
        && cuda_launch_active()
    {
        match cuda_gemm_f32(&a.host, &b.host, m, n, k) {
            Ok(data) => data,
            Err(e) => {
                log::warn!("CUDA gemm launch failed, host fallback: {}", e);
                host_gemm_data(&a, &b)?
            }
        }
    } else {
        host_gemm_data(&a, &b)?
    };

    let mut out = GpuBuffer::from_host(&[m, n], out_host, ctx.device_id());
    download(&mut out)?;
    Ok(out)
}

fn host_gemm_data(a: &GpuBuffer, b: &GpuBuffer) -> Result<Vec<f32>> {
    let ta = Tensor::from_vec(a.host.clone(), &a.shape)?;
    let tb = Tensor::from_vec(b.host.clone(), &b.shape)?;
    let tc = cpu_matmul(&ta, &tb)?;
    Ok(tc.data)
}

/// Host tensors → device matmul → host tensor.
pub fn matmul_on_device(ctx: &GpuContext, a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let ga = GpuBuffer::from_tensor(a, ctx.device_id());
    let gb = GpuBuffer::from_tensor(b, ctx.device_id());
    let gc = gpu_matmul(ctx, &ga, &gb)?;
    gc.to_tensor()
}
