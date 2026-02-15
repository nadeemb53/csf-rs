//! Full BERT Tokenizer Implementation
//!
//! Implements WordPiece tokenization with the downloaded tokenizer.json
//! for deterministic, production-ready tokenization.

use crate::tokens;
use crate::{CanonicalError, Result, TokenOutput, EMBEDDING_DIM, MAX_SEQ_LEN};
use serde::Deserialize;
use std::collections::HashMap;

// ============================================================================
// Tokenizer JSON Structure - Simplified
// ============================================================================

#[derive(Debug, Deserialize)]
struct TokenizerJson {
    #[serde(default)]
    added_tokens: Vec<AddedToken>,
    #[serde(default)]
    model: Option<ModelJson>,
}

#[derive(Debug, Deserialize)]
struct AddedToken {
    id: u32,
    content: String,
}

#[derive(Debug, Deserialize)]
struct ModelJson {
    #[serde(default)]
    unk_token: Option<String>,
    #[serde(default)]
    vocab: Option<HashMap<String, u32>>,
}

// ============================================================================
// Full BERT Tokenizer
// ============================================================================

pub struct BertTokenizer {
    vocab: HashMap<String, u32>,
    vocab_size: usize,
    reverse_vocab: Vec<String>,
    unk_token_id: u32,
    cls_token_id: u32,
    sep_token_id: u32,
    pad_token_id: u32,
    mask_token_id: u32,
    unk_token: String,
    continuing_subword_prefix: String,
    max_input_chars_per_word: usize,
}

impl BertTokenizer {
    /// Create a new tokenizer from tokenizer.json bytes
    pub fn new(tokenizer_json: &[u8]) -> Result<Self> {
        // Parse JSON
        let tokenizer: TokenizerJson = serde_json::from_slice(tokenizer_json)
            .map_err(|e| CanonicalError::Tokenizer(format!("Failed to parse tokenizer.json: {}", e)))?;

        // Get vocab from model
        let vocab = tokenizer.model
            .as_ref()
            .and_then(|m| m.vocab.as_ref())
            .ok_or_else(|| CanonicalError::Tokenizer("No vocabulary found in tokenizer.json".to_string()))?
            .clone();

        let vocab_size = vocab.len();

        // Get unk_token from model config
        let unk_token = tokenizer.model
            .as_ref()
            .and_then(|m| m.unk_token.clone())
            .unwrap_or_else(|| "[UNK]".to_string());

        let unk_token_id = vocab.get(&unk_token).copied().unwrap_or(100);

        // Get special tokens from added_tokens
        let mut cls_token_id: u32 = tokens::CLS;
        let mut sep_token_id: u32 = tokens::SEP;
        let mut pad_token_id: u32 = tokens::PAD;
        let mut mask_token_id: u32 = 103;

        for token in &tokenizer.added_tokens {
            match token.content.as_str() {
                "[CLS]" => cls_token_id = token.id,
                "[SEP]" => sep_token_id = token.id,
                "[PAD]" => pad_token_id = token.id,
                "[MASK]" => mask_token_id = token.id,
                _ => {}
            }
        }

        // Also check vocab for special tokens (in case added_tokens doesn't have them)
        if let Some(&id) = vocab.get("[CLS]") { cls_token_id = id; }
        if let Some(&id) = vocab.get("[SEP]") { sep_token_id = id; }
        if let Some(&id) = vocab.get("[PAD]") { pad_token_id = id; }
        if let Some(&id) = vocab.get("[MASK]") { mask_token_id = id; }

        // Build reverse vocab
        let mut reverse_vocab = Vec::with_capacity(vocab_size);
        for _ in 0..vocab_size {
            reverse_vocab.push(String::new());
        }
        for (token, &id) in &vocab {
            if (id as usize) < reverse_vocab.len() {
                reverse_vocab[id as usize] = token.clone();
            }
        }

        Ok(Self {
            vocab,
            vocab_size,
            reverse_vocab,
            unk_token_id,
            cls_token_id,
            sep_token_id,
            pad_token_id,
            mask_token_id,
            unk_token,
            continuing_subword_prefix: "##".to_string(),
            max_input_chars_per_word: 100,
        })
    }

    /// Tokenize input text
    pub fn tokenize(&self, text: &str) -> Result<TokenOutput> {
        // Pre-tokenize (split on whitespace and punctuation)
        let pre_tokens = self.pre_tokenize(text);

        // WordPiece tokenize each word
        let mut token_ids: Vec<u32> = vec![self.cls_token_id];

        for word in &pre_tokens {
            let mut word_ids = self.wordpiece_tokenize(word);
            token_ids.append(&mut word_ids);
        }

        // Truncate if necessary
        if token_ids.len() > MAX_SEQ_LEN - 1 {
            token_ids.truncate(MAX_SEQ_LEN - 1);
        }

        // Add sep token
        token_ids.push(self.sep_token_id);

        // Create attention mask (1 for real tokens, 0 for padding)
        let attention_mask: Vec<u32> = token_ids.iter().map(|_| 1).collect();
        let type_ids: Vec<u32> = token_ids.iter().map(|_| 0).collect();

        Ok(TokenOutput {
            ids: token_ids,
            attention_mask,
            type_ids,
        })
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Pre-tokenize: split on whitespace and punctuation
    fn pre_tokenize(&self, text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut current_token = String::new();

        for c in text.chars() {
            if c.is_whitespace() {
                if !current_token.is_empty() {
                    tokens.push(current_token.clone());
                    current_token.clear();
                }
            } else if c.is_ascii_punctuation() {
                if !current_token.is_empty() {
                    tokens.push(current_token.clone());
                    current_token.clear();
                }
                tokens.push(c.to_string());
            } else {
                current_token.push(c);
            }
        }

        if !current_token.is_empty() {
            tokens.push(current_token);
        }

        tokens
    }

    /// WordPiece tokenization
    fn wordpiece_tokenize(&self, word: &str) -> Vec<u32> {
        let word = word.to_lowercase();

        if word.is_empty() {
            return vec![];
        }

        if word.chars().count() > self.max_input_chars_per_word {
            return vec![self.unk_token_id];
        }

        let mut is_start = true;
        let mut sub_tokens = Vec::new();

        let mut current = String::new();
        for c in word.chars() {
            current.push(c);

            if !is_start {
                // Check if current is in vocab (as full word)
                if self.vocab.contains_key(&current) {
                    // Continue
                } else if self.vocab.contains_key(&format!("{}{}", self.continuing_subword_prefix, current)) {
                    // Need to add previous token and start new one
                    current = format!("{}{}", self.continuing_subword_prefix, current);
                    is_start = true;
                    continue;
                } else {
                    // Unknown
                    if let Some(prev) = sub_tokens.pop() {
                        // Check if we can add previous + current as continuation
                        let combined = format!("{}{}{}", self.continuing_subword_prefix, prev, current);
                        if self.vocab.contains_key(&combined) {
                            sub_tokens.push(combined);
                        } else {
                            sub_tokens.push(prev);
                            // Current becomes unknown
                            break;
                        }
                    }
                    current.clear();
                    break;
                }
            }

            // Check if current is in vocab
            if self.vocab.contains_key(&current) {
                sub_tokens.push(current.clone());
                is_start = false;
            }
        }

        // Handle remaining
        if !current.is_empty() && !is_start {
            if self.vocab.contains_key(&current) {
                sub_tokens.push(current);
            } else if let Some(prev) = sub_tokens.pop() {
                let combined = format!("{}{}{}", self.continuing_subword_prefix, prev, current);
                if self.vocab.contains_key(&combined) {
                    sub_tokens.push(combined);
                } else {
                    sub_tokens.push(prev);
                }
            }
        }

        // Convert to IDs
        sub_tokens.iter()
            .map(|t| self.vocab.get(t).copied().unwrap_or(self.unk_token_id))
            .collect()
    }
}
