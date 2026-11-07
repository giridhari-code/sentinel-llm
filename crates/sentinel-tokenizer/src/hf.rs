#![cfg(feature = "hf")]
//! HuggingFace `tokenizers` backend.

use crate::Tokenizer;
use sentinel_core::types::TokenId;
use sentinel_core::{Result, SentinelError};
use std::path::Path;
use tokenizers::Tokenizer as HfTok;

pub struct HfTokenizer {
    inner: HfTok,
    vocab_size: usize,
    bos_id: Option<TokenId>,
    eos_id: Option<TokenId>,
    pad_id: Option<TokenId>,
    unk_id: Option<TokenId>,
}

impl HfTokenizer {
    pub fn from_file(path: &Path) -> Result<Self> {
        let inner = HfTok::from_file(path).map_err(|e| {
            SentinelError::Other(format!("Failed to load tokenizer: {}", e))
        })?;

        let vocab_size = inner.get_vocab_size(true);

        // Best-effort special token detection
        let bos_id = inner.token_to_id("<s>")
            .or_else(|| inner.token_to_id("<bos>"))
            .or_else(|| inner.token_to_id("[BOS]"));
        let eos_id = inner.token_to_id("</s>")
            .or_else(|| inner.token_to_id("<eos>"))
            .or_else(|| inner.token_to_id("[EOS]"));
        let pad_id = inner.token_to_id("<pad>")
            .or_else(|| inner.token_to_id("[PAD]"));
        let unk_id = inner.token_to_id("<unk>")
            .or_else(|| inner.token_to_id("[UNK]"));

        Ok(Self {
            inner,
            vocab_size,
            bos_id,
            eos_id,
            pad_id,
            unk_id,
        })
    }
}

impl Tokenizer for HfTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<TokenId>> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| SentinelError::Other(format!("Encode error: {}", e)))?;
        Ok(encoding.get_ids().iter().map(|&id| id as TokenId).collect())
    }

    fn decode(&self, tokens: &[TokenId]) -> Result<String> {
        let ids: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
        self.inner
            .decode(&ids, true)
            .map_err(|e| SentinelError::Other(format!("Decode error: {}", e)))
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn bos_id(&self) -> Option<TokenId> {
        self.bos_id
    }
    fn eos_id(&self) -> Option<TokenId> {
        self.eos_id
    }
    fn pad_id(&self) -> Option<TokenId> {
        self.pad_id
    }
    fn unk_id(&self) -> Option<TokenId> {
        self.unk_id
    }
}
