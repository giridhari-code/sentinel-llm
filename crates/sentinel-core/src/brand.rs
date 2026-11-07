//! Orbit naming — product universe for this stack.
//!
//! | Name       | Role                                      |
//! |------------|-------------------------------------------|
//! | **Orbit**  | Full platform / company stack             |
//! | **Sentinel** | The LLM (MoE + looped Transformer)      |
//! | **Tesla**  | Custom compute kernel (CPU/CUDA)          |
//! | **DoBu**   | Data + tokenizer pipeline                 |
//! | **Bhutu**  | Inference runtime (prefill/decode/serve)  |

/// Platform name
pub const ORBIT: &str = "Orbit";
/// LLM model family
pub const SENTINEL: &str = "Sentinel";
/// Compute kernel family
pub const TESLA: &str = "Tesla";
/// Data / tokenizer layer
pub const DOBU: &str = "DoBu";
/// Inference runtime
pub const BHUTU: &str = "Bhutu";

/// One-line product tag
pub fn product_tag() -> String {
    format!(
        "{} · {} model · {} kernel · {} data · {} runtime",
        ORBIT, SENTINEL, TESLA, DOBU, BHUTU
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn product_tag_contains_all_names() {
        let t = product_tag();
        assert!(t.contains(ORBIT));
        assert!(t.contains(SENTINEL));
        assert!(t.contains(TESLA));
        assert!(t.contains(DOBU));
        assert!(t.contains(BHUTU));
    }
}
