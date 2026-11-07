//! GPU memory buffer (host-mirrored for Tesla-2 prototype).

use sentinel_core::types::Float;

/// Device buffer. With CUDA feature this will live in VRAM;
/// prototype keeps a host mirror so CPU fallback paths stay testable.
#[derive(Clone, Debug)]
pub struct GpuBuffer {
    pub shape: Vec<usize>,
    /// Host-side mirror (always present in Tesla-2.0 scaffold)
    pub host: Vec<Float>,
    pub device_id: u32,
    /// Opaque device pointer placeholder (0 = host-only)
    pub device_ptr: u64,
}

impl GpuBuffer {
    pub fn from_host(shape: &[usize], data: Vec<Float>, device_id: u32) -> Self {
        Self {
            shape: shape.to_vec(),
            host: data,
            device_id,
            device_ptr: 0,
        }
    }

    pub fn zeros(shape: &[usize], device_id: u32) -> Self {
        let n: usize = shape.iter().product();
        Self {
            shape: shape.to_vec(),
            host: vec![0.0; n],
            device_id,
            device_ptr: 0,
        }
    }

    pub fn numel(&self) -> usize {
        self.host.len()
    }
}
