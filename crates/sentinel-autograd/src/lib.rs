//! Reverse-mode autograd for Sentinel (pure Rust, no third-party AD).
//!
//! Multi-head SDPA backward + MoE load-balancing helpers.

mod tape;
mod ops;

pub use tape::{Tape, Var, GradMap};
pub use ops::{
    softmax_cross_entropy, softmax_last, softmax_backward, sdpa2, sdpa2_backward, topk_route,
    multi_head_sdpa, multi_head_sdpa_backward, split_heads, concat_heads, load_balance_loss,
    apply_causal_mask_2d, sdpa2_ex,
};
