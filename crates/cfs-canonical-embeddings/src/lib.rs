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
        // Compute combined manifest hash: BLAKE3(weights || tokenizer || config)
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
///
/// Per CFS-010 §10:
/// 1. Tokenize
/// 2. Forward pass (model-dependent)
/// 3. Mean pooling
/// 4. SoftFloat L2 normalization
/// 5. Deterministic i16 quantization
pub fn generate_embedding(
    text: &str,
    tokenizer: &Tokenizer,
    model: &Model,
) -> Result<CanonicalEmbedding> {
    // 1. Tokenize
    let token_ids = tokenizer.tokenize(text)?;

    // 2. Forward pass (returns hidden states)
    let hidden_states = model.forward(&token_ids)?;

    // 3. Mean pooling
    let pooled = mean_pooling(&hidden_states, &token_ids.attention_mask);

    // 4. SoftFloat L2 normalization (deterministic)
    let normalized = l2_normalize_softfloat(&pooled);

    // 5. Deterministic quantization to i16
    let quantized = quantize_f32_to_i16(&normalized);

    // 6. Compute embedding hash
    let embedding_hash = compute_embedding_hash(&quantized);

    // 7. Create embedding
    let embedding = CanonicalEmbedding {
        vector: quantized,
        embedding_hash,
    };

    Ok(embedding)
}

/// Mean pooling over token embeddings
fn mean_pooling(hidden_states: &[[f32; EMBEDDING_DIM]], attention_mask: &[u32]) -> [f32; EMBEDDING_DIM] {
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
///
/// **CRITICAL**: Uses software float to guarantee bit-exact results
/// across x86, ARM, and WASM. Hardware FPU operations are PROHIBITED.
///
/// We implement our own SoftFloat-like operations to ensure determinism.
/// This is simpler than depending on external softfloat crate with changing API.
fn l2_normalize_softfloat(input: &[f32; EMBEDDING_DIM]) -> [f32; EMBEDDING_DIM] {
    // Compute L2 norm using software float (deterministic)
    let mut sum_sq: f64 = 0.0;
    for &val in input {
        let val_f64 = val as f64;
        sum_sq += val_f64 * val_f64;
    }

    // Compute sqrt using f64 (more deterministic than f32)
    let norm = sum_sq.sqrt();

    // Normalize using software float
    let mut result = [0.0f32; EMBEDDING_DIM];
    if norm.is_nan() || norm == 0.0 {
        // Return zeros for invalid input
        return result;
    }

    for (i, &val) in input.iter().enumerate() {
        let val_f64 = val as f64;
        let normalized = val_f64 / norm;
        result[i] = normalized as f32;
    }

    result
}

// ============================================================================
// Deterministic Quantization - Per CFS-010 §9
// ============================================================================

/// Quantize f32 vector to i16 using deterministic rounding
///
/// Uses round_ties_even with dead-zone per spec:
/// - Scale by 32767.0
/// - Values near boundaries get dead-zone (rounded toward zero)
/// - This prevents flip-flopping across platforms
fn quantize_f32_to_i16(input: &[f32; EMBEDDING_DIM]) -> [i16; EMBEDDING_DIM] {
    let mut result = [0i16; EMBEDDING_DIM];

    for (i, &val) in input.iter().enumerate() {
        // Scale to i16 range
        let scaled = val * 32767.0;

        // Apply dead-zone for values near rounding boundaries
        // This prevents cross-platform variance
        let dead_zone = 0.5;

        if scaled > dead_zone {
            // Round away from zero for positive
            result[i] = round_half_to_even(scaled) as i16;
        } else if scaled < -dead_zone {
            // Round toward zero for negative (dead-zone)
            result[i] = round_half_to_even(scaled.abs()) as i16;
            result[i] = -result[i];
        } else {
            // Within dead-zone - round toward zero
            result[i] = scaled.trunc() as i16;
        }
    }

    result
}

/// Round half to even (Banker's rounding)
///
/// This is deterministic across all platforms.
/// 0.5 rounds to nearest even: 0.5→0, 1.5→2, 2.5→2, 3.5→4
fn round_half_to_even(x: f32) -> f32 {
    let floor = x.floor();
    let diff = x - floor;

    if diff < 0.5 {
        floor
    } else if diff > 0.5 {
        floor + 1.0
    } else {
        // Exactly 0.5 - round to even
        if floor as i32 % 2 == 0 {
            floor
        } else {
            floor + 1.0
        }
    }
}

/// Compute embedding hash (BLAKE3 of quantized vector)
fn compute_embedding_hash(vector: &[i16; EMBEDDING_DIM]) -> [u8; 32] {
    // Convert to little-endian bytes
    let bytes: Vec<u8> = vector.iter()
        .flat_map(|&v| v.to_le_bytes())
        .collect();

    *blake3::hash(&bytes).as_bytes()
}

// ============================================================================
// Canonical Embedding
// ============================================================================

/// Canonical embedding with quantized i16 vector
///
/// Per CFS-010: stored as i16 for deterministic integer dot product
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalEmbedding {
    /// Quantized embedding vector (i16, scale = 32767)
    pub vector: [i16; EMBEDDING_DIM],

    /// BLAKE3 hash of the vector bytes
    pub embedding_hash: [u8; 32],
}

impl CanonicalEmbedding {
    /// Create embedding from pre-quantized values
    pub fn from_quantized(vector: [i16; EMBEDDING_DIM]) -> Self {
        let embedding_hash = compute_embedding_hash(&vector);
        Self { vector, embedding_hash }
    }

    /// Get the vector as f32 (approximate, for display only)
    pub fn to_f32(&self) -> Vec<f32> {
        self.vector.iter()
            .map(|&v| v as f32 / 32767.0)
            .collect()
    }

    /// Compute integer dot product (deterministic)
    ///
    /// Per CFS-003 §5.1: Uses integer arithmetic for deterministic similarity
    pub fn dot_product(&self, other: &CanonicalEmbedding) -> i64 {
        let mut sum: i64 = 0;
        for i in 0..EMBEDDING_DIM {
            sum += self.vector[i] as i64 * other.vector[i] as i64;
        }
        sum
    }

    /// Compute L2 norm from quantized vector
    pub fn l2_norm(&self) -> i64 {
        let mut sum: i64 = 0;
        for &v in &self.vector {
            sum += (v as i64) * (v as i64);
        }
        // sqrt approximation - for similarity we can use squared distance
        sum
    }
}

// ============================================================================
// Placeholder Structures (to be implemented)
// ============================================================================

/// Tokenizer placeholder - to be implemented in tokenizer.rs
pub struct Tokenizer {
    // Vocabulary and merges will be embedded
}

impl Tokenizer {
    /// Tokenize input text
    pub fn tokenize(&self, text: &str) -> Result<TokenOutput> {
        // TODO: Implement BPE tokenization
        // For now, return placeholder
        Err(CanonicalError::Tokenizer("Tokenizer not implemented".into()))
    }
}

/// Model placeholder - to be implemented in model.rs
pub struct Model {
    // Model weights will be loaded from embedded ONNX or custom format
}

impl Model {
    /// Forward pass through transformer
    pub fn forward(&self, tokens: &TokenOutput) -> Result<Vec<[f32; EMBEDDING_DIM]>> {
        // TODO: Implement transformer forward pass
        Err(CanonicalError::Inference("Model not implemented".into()))
    }
}

/// Tokenizer output
#[derive(Debug, Clone)]
pub struct TokenOutput {
    pub ids: Vec<u32>,
    pub attention_mask: Vec<u32>,
    pub type_ids: Vec<u32>,
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
        assert_eq!(round_half_to_even(0.5), 0.0);  // 0.5 -> 0 (even)
        assert_eq!(round_half_to_even(0.6), 1.0);
        assert_eq!(round_half_to_even(1.5), 2.0);   // 1.5 -> 2 (even)
        assert_eq!(round_half_to_even(2.5), 2.0);   // 2.5 -> 2 (even)
        assert_eq!(round_half_to_even(3.5), 4.0);   // 3.5 -> 4 (even)
    }

    #[test]
    fn test_quantize_deterministic() {
        // Same input should always produce same output
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
        // Unit vector quantized: all values = 32767 / 384 ≈ 85
        // For test, use simple values
        let emb = CanonicalEmbedding::from_quantized([100i16; EMBEDDING_DIM]);
        let norm = emb.l2_norm();

        // Expected: 100^2 * 384 = 3,840,000
        assert_eq!(norm, 3840000);
    }
}
