//! Sentinel Tokenizer
//!
//! Supports:
//! 1. HuggingFace `tokenizers` library (BPE / WordPiece / Unigram) when a vocab file is present
//! 2. Simple character-level + whitespace fallback (no external files needed for demos)

mod simple;
mod hf;

pub use simple::SimpleTokenizer;
pub use hf::HfTokenizer;

use sentinel_core::types::TokenId;
use sentinel_core::{Result, SentinelError};
use std::path::Path;

/// Common interface every tokenizer must implement.
pub trait Tokenizer: Send + Sync {
    fn encode(&self, text: &str) -> Result<Vec<TokenId>>;
    fn decode(&self, tokens: &[TokenId]) -> Result<String>;
    fn vocab_size(&self) -> usize;
    fn bos_id(&self) -> Option<TokenId>;
    fn eos_id(&self) -> Option<TokenId>;
    fn pad_id(&self) -> Option<TokenId>;
    fn unk_id(&self) -> Option<TokenId>;

    fn encode_with_special(&self, text: &str, add_bos: bool, add_eos: bool) -> Result<Vec<TokenId>> {
        let mut ids = Vec::new();
        if add_bos {
            if let Some(bos) = self.bos_id() {
                ids.push(bos);
            }
        }
        ids.extend(self.encode(text)?);
        if add_eos {
            if let Some(eos) = self.eos_id() {
                ids.push(eos);
            }
        }
        Ok(ids)
    }
}

/// Load a tokenizer.
///
/// Priority:
/// 1. If `path` points to a HuggingFace `tokenizer.json` → use HfTokenizer
/// 2. Otherwise fall back to SimpleTokenizer (character / whitespace)
pub fn load_tokenizer(path: Option<&Path>) -> Result<Box<dyn Tokenizer>> {
    if let Some(p) = path {
        if p.exists() {
            log::info!("Loading HuggingFace tokenizer from {}", p.display());
            let tok = HfTokenizer::from_file(p)?;
            return Ok(Box::new(tok));
        }
        return Err(SentinelError::Other(format!(
            "Tokenizer file not found: {}",
            p.display()
        )));
    }

    log::info!("No tokenizer file provided — using SimpleTokenizer (demo)");
    Ok(Box::new(SimpleTokenizer::new()))
}
