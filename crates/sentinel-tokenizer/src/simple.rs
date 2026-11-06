//! Simple fallback tokenizer (no external files).
//!
//! - Whitespace splitting + character-level for unknown pieces
//! - Fixed special tokens
//! - Enough for architecture demos and smoke tests

use crate::Tokenizer;
use sentinel_core::types::TokenId;
use sentinel_core::{Result, SentinelError};
use std::collections::HashMap;

const SPECIAL: &[&str] = &["<pad>", "<unk>", "<bos>", "<eos>"];

pub struct SimpleTokenizer {
    token_to_id: HashMap<String, TokenId>,
    id_to_token: Vec<String>,
    vocab_size: usize,
}

impl SimpleTokenizer {
    pub fn new() -> Self {
        let mut token_to_id = HashMap::new();
        let mut id_to_token = Vec::new();

        // Special tokens first
        for (i, t) in SPECIAL.iter().enumerate() {
            token_to_id.insert(t.to_string(), i as TokenId);
            id_to_token.push(t.to_string());
        }

        // Printable ASCII (space + visible)
        for c in 32u8..=126 {
            let s = (c as char).to_string();
            if !token_to_id.contains_key(&s) {
                let id = id_to_token.len() as TokenId;
                token_to_id.insert(s.clone(), id);
                id_to_token.push(s);
            }
        }

        // Common extra chars useful for code / multilingual demos
        for c in ['\n', '\t', '—', '–', '‘', '’', '“', '”'] {
            let s = c.to_string();
            if !token_to_id.contains_key(&s) {
                let id = id_to_token.len() as TokenId;
                token_to_id.insert(s.clone(), id);
                id_to_token.push(s);
            }
        }

        let vocab_size = id_to_token.len();
        Self {
            token_to_id,
            id_to_token,
            vocab_size,
        }
    }

    fn lookup(&self, piece: &str) -> TokenId {
        self.token_to_id
            .get(piece)
            .copied()
            .unwrap_or(self.unk_id().unwrap())
    }
}

impl Default for SimpleTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Tokenizer for SimpleTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<TokenId>> {
        let mut ids = Vec::new();
        for word in text.split_whitespace() {
            // Try whole word first
            if let Some(&id) = self.token_to_id.get(word) {
                ids.push(id);
                continue;
            }
            // Fall back to character level
            for ch in word.chars() {
                let s = ch.to_string();
                ids.push(self.lookup(&s));
            }
        }
        Ok(ids)
    }

    fn decode(&self, tokens: &[TokenId]) -> Result<String> {
        let mut out = String::new();
        for &id in tokens {
            let tok = self
                .id_to_token
                .get(id as usize)
                .map(|s| s.as_str())
                .unwrap_or("<unk>");
            if tok.starts_with('<') && tok.ends_with('>') {
                // skip special tokens in decode for readability
                continue;
            }
            if !out.is_empty() && !tok.chars().all(|c| c.is_ascii_punctuation()) {
                out.push(' ');
            }
            out.push_str(tok);
        }
        Ok(out)
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn bos_id(&self) -> Option<TokenId> {
        Some(2)
    }
    fn eos_id(&self) -> Option<TokenId> {
        Some(3)
    }
    fn pad_id(&self) -> Option<TokenId> {
        Some(0)
    }
    fn unk_id(&self) -> Option<TokenId> {
        Some(1)
    }
}
