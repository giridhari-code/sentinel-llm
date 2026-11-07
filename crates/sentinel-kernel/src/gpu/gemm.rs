//! GPU GEMM entry — dispatches to Tesla-2 CUDA kernel (not cuBLAS).

use super::buffer::GpuBuffer;
use super::context::GpuContext;
use crate::gemm::matmul as cpu_matmul;
use sentinel_core::tensor::Tensor;
use sentinel_core::{Result, SentinelError};

/// C = A @ B on the active GPU context.
///
/// Tesla-2.0 scaffold: host-mirror path uses pure-Rust matmul for correctness.
/// Real VRAM kernel: `cuda/gemm.cu` (custom Tesla, not cuBLAS).
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

    let _ = ctx; // device id used when CUDA launch is wired

    let ta = Tensor::from_vec(a.host.clone(), &a.shape)?;
    let tb = Tensor::from_vec(b.host.clone(), &b.shape)?;
    let tc = cpu_matmul(&ta, &tb)?;
    Ok(GpuBuffer::from_host(&tc.shape, tc.data, ctx.device_id()))
}

/// Host tensors → device matmul → host tensor.
pub fn matmul_on_device(ctx: &GpuContext, a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let ga = GpuBuffer::from_host(&a.shape, a.data.clone(), ctx.device_id());
    let gb = GpuBuffer::from_host(&b.shape, b.data.clone(), ctx.device_id());
    let gc = gpu_matmul(ctx, &ga, &gb)?;
    Tensor::from_vec(gc.host, &gc.shape)
}
