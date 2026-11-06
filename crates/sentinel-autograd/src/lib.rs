//! Reverse-mode autograd for Sentinel (pure Rust, no third-party AD).
//!
//! Includes true SDPA backward and softmax.

mod tape;
mod ops;

pub use tape::{Tape, Var, GradMap};
pub use ops::{
    softmax_cross_entropy, softmax_last, softmax_backward, sdpa2, sdpa2_backward, topk_route,
};
