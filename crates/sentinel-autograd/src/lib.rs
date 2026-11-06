//! Reverse-mode autograd for Sentinel (pure Rust, no third-party AD).
//!
//! Usage:
//!   let mut tape = Tape::new();
//!   let w = tape.param(weight_tensor);
//!   let y = tape.matmul(&x, &w)?;
//!   let loss = tape.mse_or_ce(...)?;
//!   tape.backward(loss)?;
//!   let gw = tape.grad(w);

mod tape;
mod ops;

pub use tape::{Tape, Var, GradMap};
pub use ops::softmax_cross_entropy;
