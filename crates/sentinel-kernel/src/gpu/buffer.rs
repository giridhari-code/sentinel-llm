//! GPU memory buffer — host mirror + optional device pointer (Tesla-2).

use sentinel_core::tensor::Tensor;
use sentinel_core::types::Float;
use sentinel_core::{Result, SentinelError};

/// Device buffer. Always keeps a host mirror for correctness / fallback.
/// When CUDA is live, `device_ptr != 0` means VRAM allocation.
#[derive(Clone, Debug)]
pub struct GpuBuffer {
    pub shape: Vec<usize>,
    pub host: Vec<Float>,
    pub device_id: u32,
    /// Opaque device pointer (0 = host-only)
    pub device_ptr: u64,
    pub on_device: bool,
}

impl GpuBuffer {
    pub fn from_host(shape: &[usize], data: Vec<Float>, device_id: u32) -> Self {
        Self {
            shape: shape.to_vec(),
            host: data,
            device_id,
            device_ptr: 0,
            on_device: false,
        }
    }

    pub fn from_tensor(t: &Tensor, device_id: u32) -> Self {
        Self::from_host(&t.shape, t.data.clone(), device_id)
    }

    pub fn zeros(shape: &[usize], device_id: u32) -> Self {
        let n: usize = shape.iter().copied().product();
        Self {
            shape: shape.to_vec(),
            host: vec![0.0; n],
            device_id,
            device_ptr: 0,
            on_device: false,
        }
    }

    pub fn numel(&self) -> usize {
        self.host.len()
    }

    pub fn to_tensor(&self) -> Result<Tensor> {
        Tensor::from_vec(self.host.clone(), &self.shape)
    }

    /// Mark buffer as residing on device (after successful upload).
    pub fn mark_on_device(&mut self, ptr: u64) {
        self.device_ptr = ptr;
        self.on_device = ptr != 0;
    }

    pub fn clear_device(&mut self) {
        self.device_ptr = 0;
        self.on_device = false;
    }
}

/// Upload host → device (no-op / simulated without CUDA runtime).
pub fn upload(buf: &mut GpuBuffer) -> Result<()> {
    if buf.host.is_empty() {
        return Err(SentinelError::Kernel("upload: empty buffer".into()));
    }
    #[cfg(feature = "cuda")]
    {
        // Real path: cudaMalloc + cudaMemcpy H2D
        // Until driver is linked, keep host mirror authoritative.
        let fake_ptr = buf.host.as_ptr() as u64;
        buf.mark_on_device(fake_ptr);
        return Ok(());
    }
    #[cfg(not(feature = "cuda"))]
    {
        // Host-only Tesla-2 path: "upload" is a no-op mark
        buf.on_device = false;
        buf.device_ptr = 0;
        Ok(())
    }
}

/// Download device → host.
pub fn download(buf: &mut GpuBuffer) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        // Real path: cudaMemcpy D2H into buf.host
        let _ = buf.device_ptr;
        return Ok(());
    }
    #[cfg(not(feature = "cuda"))]
    {
        Ok(())
    }
}
