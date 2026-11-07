//! Compute device selection for Tesla kernels.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Where tensors / kernels run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Device {
    /// Multi-core CPU (Tesla-1 Rayon path)
    Cpu,
    /// NVIDIA GPU device index (Tesla-2 CUDA path)
    Cuda(u32),
}

impl Default for Device {
    fn default() -> Self {
        Self::Cpu
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Device::Cpu => write!(f, "cpu"),
            Device::Cuda(id) => write!(f, "cuda:{}", id),
        }
    }
}

impl Device {
    pub fn parse(s: &str) -> Result<Self, String> {
        let s = s.trim().to_ascii_lowercase();
        if s == "cpu" {
            return Ok(Device::Cpu);
        }
        if s == "cuda" || s == "gpu" {
            return Ok(Device::Cuda(0));
        }
        if let Some(rest) = s.strip_prefix("cuda:") {
            let id: u32 = rest
                .parse()
                .map_err(|_| format!("invalid cuda device id: {}", rest))?;
            return Ok(Device::Cuda(id));
        }
        Err(format!("unknown device '{}', use cpu|cuda|cuda:N", s))
    }

    pub fn is_cuda(&self) -> bool {
        matches!(self, Device::Cuda(_))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parse_devices() {
        assert_eq!(Device::parse("cpu").unwrap(), Device::Cpu);
        assert_eq!(Device::parse("cuda").unwrap(), Device::Cuda(0));
        assert_eq!(Device::parse("cuda:1").unwrap(), Device::Cuda(1));
        assert!(Device::parse("metal").is_err());
    }
}
