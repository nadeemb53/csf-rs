//! CFS Canonical Embeddings - Deterministic WASM-based Embedding Generation
//!
//! This module provides bit-exact deterministic embedding generation
//! by using SoftFloat for normalization and deterministic quantization.
//!
//! Per CFS-010 and CFS-003:
//! - Uses SoftFloat for L2 normalization (no hardware FPU)
//! - Deterministic round_ties_even quantization with dead-zone
//! - Embeddings stored as i16 for cross-platform consistency
//!
//! Model: all-MiniLM-L6-v2 (384 dimensions)

use thiserror::Error;

// ============================================================================
// Error Types
// ============================================================================

#[derive(Error, Debug)]
pub enum CanonicalError {
    #[error("Tokenizer error: {0}")]
    Tokenizer(String),

    #[error("Model inference error: {0}")]
    Inference(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Serialization error: {0}")]
    Serialization(String),
}

pub type Result<T> = std::result::Result<T, CanonicalError>;

// ============================================================================
// Model Constants (all-MiniLM-L6-v2)
// ============================================================================

/// Model identifier
pub const MODEL_ID: &str = "sentence-transformers/all-MiniLM-L6-v2";
pub const MODEL_VERSION: &str = "1.0.0";

/// Embedding dimension
pub const EMBEDDING_DIM: usize = 384;

/// Maximum sequence length
pub const MAX_SEQ_LEN: usize = 256;

/// Special token IDs
pub mod tokens {
    pub const PAD: u32 = 0;
    pub const UNK: u32 = 100;
    pub const CLS: u32 = 101;
    pub const SEP: u32 = 102;
}

// ============================================================================
// Include tokenizer and model data (downloaded at build time)
// ============================================================================

include!("tokenizer_data.rs");

// ============================================================================
// SoftFloat Implementation
// ============================================================================

mod softfloat;
pub use softfloat::{l2_normalize_softfloat as normalize_softfloat, SoftFloat32, SoftFloat64};

// ============================================================================
// Tokenizer Implementation
// ============================================================================

mod tokenizer_impl;
pub use tokenizer_impl::BertTokenizer;

// ============================================================================
// Model Implementation
// ============================================================================

mod model;
use model::MiniLMModel;

// ============================================================================
// Tokenizer Output
// ============================================================================

/// Tokenizer output
#[derive(Debug, Clone, PartialEq)]
pub struct TokenOutput {
    pub ids: Vec<u32>,
    pub attention_mask: Vec<u32>,
    pub type_ids: Vec<u32>,
}

/// Tokenizer - wrapper around BertTokenizer
pub struct Tokenizer {
    inner: BertTokenizer,
}

impl Tokenizer {
    /// Create tokenizer from embedded tokenizer.json
    pub fn new() -> Result<Self> {
        let inner = BertTokenizer::new(TOKENIZER_JSON)?;
        Ok(Self { inner })
    }

    /// Tokenize input text
    pub fn tokenize(&self, text: &str) -> Result<TokenOutput> {
        self.inner.tokenize(text)
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }
}

impl Default for Tokenizer {
    fn default() -> Self {
        Self::new().expect("Failed to create tokenizer")
    }
}

// ============================================================================
// Model Manifest - Per CFS-010 §12
// ============================================================================

/// Model manifest for provenance tracking
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelManifest {
    pub model_id: String,
    pub version: String,
    pub weights_hash: [u8; 32],
    pub tokenizer_hash: [u8; 32],
    pub config_hash: [u8; 32],
    pub manifest_hash: [u8; 32],
}

impl ModelManifest {
    /// Create manifest from component hashes
    pub fn new(
        model_id: String,
        version: String,
        weights_hash: [u8; 32],
        tokenizer_hash: [u8; 32],
        config_hash: [u8; 32],
    ) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&weights_hash);
        hasher.update(&tokenizer_hash);
        hasher.update(&config_hash);
        let manifest_hash = *hasher.finalize().as_bytes();

        Self {
            model_id,
            version,
            weights_hash,
            tokenizer_hash,
            config_hash,
            manifest_hash,
        }
    }

    /// Get manifest hash as hex string
    pub fn manifest_hash_hex(&self) -> String {
        self.manifest_hash
            .iter()
            .map(|b| format!("{:02x}", b))
            .collect()
    }
}

// ============================================================================
// Deterministic Embedding Generation
// ============================================================================

/// Generate a deterministic embedding from input text
pub fn generate_embedding(
    text: &str,
    tokenizer: &Tokenizer,
    model: &Model,
) -> Result<CanonicalEmbedding> {
    let token_output = tokenizer.tokenize(text)?;
    let hidden_states = model.forward(&token_output.ids, &token_output.attention_mask)?;
    let pooled = mean_pooling(&hidden_states, &token_output.attention_mask);
    let normalized = l2_normalize_softfloat(&pooled);
    let quantized = quantize_f32_to_i16(&normalized);
    let embedding_hash = compute_embedding_hash(&quantized);

    let embedding = CanonicalEmbedding {
        vector: quantized,
        embedding_hash,
    };

    Ok(embedding)
}

/// Mean pooling over token embeddings
fn mean_pooling(hidden_states: &[Vec<f32>], attention_mask: &[u32]) -> [f32; EMBEDDING_DIM] {
    let mut sum = [0.0f32; EMBEDDING_DIM];
    let mut count = 0.0f32;

    for (i, token_emb) in hidden_states.iter().enumerate() {
        if i < attention_mask.len() && attention_mask[i] == 1 {
            for (j, val) in token_emb.iter().enumerate() {
                sum[j] += val;
            }
            count += 1.0;
        }
    }

    if count > 0.0 {
        for val in &mut sum {
            *val /= count;
        }
    }

    sum
}

// ============================================================================
// SoftFloat L2 Normalization - Per CFS-010 §8
// ============================================================================

/// L2 normalize using software float for deterministic results
/// Uses true SoftFloat implementation for bit-exact results across platforms
fn l2_normalize_softfloat(input: &[f32; EMBEDDING_DIM]) -> [f32; EMBEDDING_DIM] {
    softfloat::l2_normalize_softfloat(input)
}

// ============================================================================
// Deterministic Quantization - Per CFS-010 §9
// ============================================================================

fn quantize_f32_to_i16(input: &[f32; EMBEDDING_DIM]) -> [i16; EMBEDDING_DIM] {
    let mut result = [0i16; EMBEDDING_DIM];

    for (i, &val) in input.iter().enumerate() {
        let scaled = val * 32767.0;
        let dead_zone = 0.5;

        if scaled > dead_zone {
            result[i] = round_half_to_even(scaled) as i16;
        } else if scaled < -dead_zone {
            result[i] = round_half_to_even(scaled.abs()) as i16;
            result[i] = -result[i];
        } else {
            result[i] = scaled.trunc() as i16;
        }
    }

    result
}

fn round_half_to_even(x: f32) -> f32 {
    let floor = x.floor();
    let diff = x - floor;

    if diff < 0.5 {
        floor
    } else if diff > 0.5 {
        floor + 1.0
    } else {
        if floor as i32 % 2 == 0 {
            floor
        } else {
            floor + 1.0
        }
    }
}

fn compute_embedding_hash(vector: &[i16; EMBEDDING_DIM]) -> [u8; 32] {
    let bytes: Vec<u8> = vector.iter()
        .flat_map(|&v| v.to_le_bytes())
        .collect();

    *blake3::hash(&bytes).as_bytes()
}

// ============================================================================
// Canonical Embedding
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalEmbedding {
    pub vector: [i16; EMBEDDING_DIM],
    pub embedding_hash: [u8; 32],
}

impl CanonicalEmbedding {
    pub fn from_quantized(vector: [i16; EMBEDDING_DIM]) -> Self {
        let embedding_hash = compute_embedding_hash(&vector);
        Self { vector, embedding_hash }
    }

    pub fn to_f32(&self) -> Vec<f32> {
        self.vector.iter()
            .map(|&v| v as f32 / 32767.0)
            .collect()
    }

    pub fn dot_product(&self, other: &CanonicalEmbedding) -> i64 {
        let mut sum: i64 = 0;
        for i in 0..EMBEDDING_DIM {
            sum += self.vector[i] as i64 * other.vector[i] as i64;
        }
        sum
    }

    pub fn l2_norm(&self) -> i64 {
        let mut sum: i64 = 0;
        for &v in &self.vector {
            sum += (v as i64) * (v as i64);
        }
        sum
    }
}

// ============================================================================
// Model - MiniLM-L6-v2 Implementation
// ============================================================================

/// The MiniLM model for generating embeddings
pub struct Model {
    inner: MiniLMModel,
    config: model::ModelConfig,
}

impl Model {
    /// Create a new model from embedded weights
    pub fn new() -> Result<Self> {
        let config = model::ModelConfig::from_config_json(CONFIG_JSON)?;
        let inner = MiniLMModel::new(MODEL_DATA, config.clone())?;
        Ok(Self { inner, config })
    }

    /// Forward pass - returns mean-pooled hidden states as embedding
    pub fn forward(&self, input_ids: &[u32], attention_mask: &[u32]) -> Result<Vec<Vec<f32>>> {
        // Get pooled output (mean pooling)
        let seq_len = input_ids.len();

        // Embeddings
        let token_type_ids: Vec<u32> = input_ids.iter().map(|_| 0).collect();
        let mut hidden_states = self.inner.embeddings.forward(input_ids, &token_type_ids);

        // Encoder layers
        hidden_states = self.inner.encoder.forward(&hidden_states, None);

        // Mean pooling
        let mut sum = vec![0.0f32; EMBEDDING_DIM];
        let mut count = 0.0f32;

        for (i, hs) in hidden_states.iter().enumerate() {
            if i < attention_mask.len() && attention_mask[i] == 1 {
                for (j, val) in hs.iter().enumerate() {
                    sum[j] += val;
                }
                count += 1.0;
            }
        }

        if count > 0.0 {
            for val in &mut sum {
                *val /= count;
            }
        }

        // Return as single vector per sequence (for now we only support batch=1)
        // The embedding for each position would be the same due to mean pooling
        Ok(vec![sum])
    }
}

impl Default for Model {
    fn default() -> Self {
        Self::new().expect("Failed to create model")
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_round_half_to_even() {
        assert_eq!(round_half_to_even(0.4), 0.0);
        assert_eq!(round_half_to_even(0.5), 0.0);
        assert_eq!(round_half_to_even(0.6), 1.0);
        assert_eq!(round_half_to_even(1.5), 2.0);
        assert_eq!(round_half_to_even(2.5), 2.0);
        assert_eq!(round_half_to_even(3.5), 4.0);
    }

    #[test]
    fn test_quantize_deterministic() {
        let input = [0.1234f32; EMBEDDING_DIM];
        let result1 = quantize_f32_to_i16(&input);
        let result2 = quantize_f32_to_i16(&input);
        assert_eq!(result1, result2);
    }

    #[test]
    fn test_embedding_hash() {
        let vector = [100i16; EMBEDDING_DIM];
        let hash1 = compute_embedding_hash(&vector);
        let hash2 = compute_embedding_hash(&vector);
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_dot_product_deterministic() {
        let emb1 = CanonicalEmbedding::from_quantized([100i16; EMBEDDING_DIM]);
        let emb2 = CanonicalEmbedding::from_quantized([200i16; EMBEDDING_DIM]);
        let dot1 = emb1.dot_product(&emb2);
        let dot2 = emb1.dot_product(&emb2);
        assert_eq!(dot1, dot2);
    }

    #[test]
    fn test_l2_norm() {
        let emb = CanonicalEmbedding::from_quantized([100i16; EMBEDDING_DIM]);
        let norm = emb.l2_norm();
        assert_eq!(norm, 3840000);
    }

    #[test]
    fn test_tokenizer_creation() {
        let tokenizer = Tokenizer::new();
        assert!(tokenizer.is_ok());
        let tok = tokenizer.unwrap();
        assert!(tok.vocab_size() > 0);
    }

    #[test]
    fn test_tokenizer_deterministic() {
        let tokenizer = Tokenizer::new().unwrap();

        let result1 = tokenizer.tokenize("Hello world").unwrap();
        let result2 = tokenizer.tokenize("Hello world").unwrap();

        assert_eq!(result1.ids, result2.ids);
    }

    #[test]
    fn test_tokenizer_special_tokens() {
        let tokenizer = Tokenizer::new().unwrap();
        let result = tokenizer.tokenize("hi").unwrap();

        // Should have [CLS] at start
        assert_eq!(result.ids[0], tokens::CLS);
    }

    #[test]
    fn test_model_creation() {
        let model = Model::new();
        assert!(model.is_ok());
    }
}
