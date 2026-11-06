//! Dataset loaders for causal LM training.

use sentinel_core::types::TokenId;
use sentinel_core::{Result, SentinelError};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Flat token stream with sliding-window batches.
#[derive(Clone)]
pub struct TokenDataset {
    pub tokens: Vec<TokenId>,
    pub seq_len: usize,
    pub stride: usize,
}

impl TokenDataset {
    pub fn from_tokens(tokens: Vec<TokenId>, seq_len: usize, stride: Option<usize>) -> Self {
        let seq_len = seq_len.max(2);
        let stride = stride.unwrap_or(seq_len / 2).max(1);
        Self {
            tokens,
            seq_len,
            stride,
        }
    }

    /// Load whitespace / comma separated u32 ids from a text file.
    pub fn from_id_file(path: &Path, seq_len: usize) -> Result<Self> {
        let f = File::open(path)?;
        let reader = BufReader::new(f);
        let mut tokens = Vec::new();
        for line in reader.lines() {
            let line = line?;
            for part in line.split(|c: char| c.is_whitespace() || c == ',') {
                if part.is_empty() {
                    continue;
                }
                let id: u32 = part.parse().map_err(|_| {
                    SentinelError::Other(format!("bad token id '{}'", part))
                })?;
                tokens.push(id);
            }
        }
        if tokens.len() < 2 {
            return Err(SentinelError::Other("dataset too short".into()));
        }
        Ok(Self::from_tokens(tokens, seq_len, None))
    }

    /// Load raw text and map bytes/chars to token ids via simple modulo vocab (demo).
    pub fn from_text_file(path: &Path, seq_len: usize, vocab_size: usize) -> Result<Self> {
        let text = std::fs::read_to_string(path)?;
        let tokens: Vec<TokenId> = text
            .chars()
            .map(|c| (c as u32) % (vocab_size as u32).max(1))
            .collect();
        if tokens.len() < 2 {
            return Err(SentinelError::Other("text dataset too short".into()));
        }
        Ok(Self::from_tokens(tokens, seq_len, None))
    }

    pub fn len(&self) -> usize {
        if self.tokens.len() < self.seq_len {
            0
        } else {
            (self.tokens.len() - self.seq_len) / self.stride + 1
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get window i as token slice.
    pub fn get(&self, index: usize) -> Option<&[TokenId]> {
        let start = index * self.stride;
        if start + self.seq_len > self.tokens.len() {
            return None;
        }
        Some(&self.tokens[start..start + self.seq_len])
    }

    pub fn iter(&self) -> TokenWindowIter<'_> {
        TokenWindowIter {
            ds: self,
            index: 0,
        }
    }
}

pub struct TokenWindowIter<'a> {
    ds: &'a TokenDataset,
    index: usize,
}

impl<'a> Iterator for TokenWindowIter<'a> {
    type Item = &'a [TokenId];

    fn next(&mut self) -> Option<Self::Item> {
        let item = self.ds.get(self.index)?;
        self.index += 1;
        Some(item)
    }
}

/// Simple in-memory batch of sequences (each length seq_len).
pub struct Batch {
    pub sequences: Vec<Vec<TokenId>>,
}

impl Batch {
    pub fn from_dataset(ds: &TokenDataset, start: usize, batch_size: usize) -> Self {
        let mut sequences = Vec::new();
        for i in start..start + batch_size {
            match ds.get(i) {
                Some(s) => sequences.push(s.to_vec()),
                None => break,
            }
        }
        Self { sequences }
    }
}
