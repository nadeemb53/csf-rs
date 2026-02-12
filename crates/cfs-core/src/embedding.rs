//! Embedding node representing a vector derived from a chunk

use half::f16;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// An embedding vector derived from a chunk
///
/// Stores the vector in f16 format for storage efficiency.
/// Tracks the model used for provenance and compatibility checking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Embedding {
    /// Unique identifier for this embedding
    pub id: Uuid,

    /// Parent chunk ID
    pub chunk_id: Uuid,

    /// The embedding vector (f16 for storage efficiency)
    pub vector: Vec<f16>,

    /// Hash of the model weights used to generate this embedding
    pub model_hash: [u8; 32],

    /// Dimensionality of the vector
    pub dim: u16,

    /// L2 norm of the vector (for cosine similarity)
    pub norm: f32,
}

impl Embedding {
    /// Create a new embedding from an f32 vector
    ///
    /// Automatically converts to f16 and computes the L2 norm.
    /// The embedding ID is deterministic based on chunk_id + model_hash.
    pub fn new(chunk_id: Uuid, vector_f32: &[f32], model_hash: [u8; 32]) -> Self {
        let norm = compute_l2_norm(vector_f32);
        let dim = vector_f32.len() as u16;

        // Convert to f16 for storage
        let vector: Vec<f16> = vector_f32.iter().map(|&v| f16::from_f32(v)).collect();

        // Deterministic ID based on chunk_id + model_hash
        let mut id_input = chunk_id.as_bytes().to_vec();
        id_input.extend_from_slice(&model_hash);
        let id = Uuid::new_v5(&crate::namespaces::EMBEDDING, &id_input);

        Self {
            id,
            chunk_id,
            vector,
            model_hash,
            dim,
            norm,
        }
    }

    /// Convert the vector back to f32 for computation
    pub fn to_f32(&self) -> Vec<f32> {
        self.vector.iter().map(|v| v.to_f32()).collect()
    }

    /// Get the normalized vector for cosine similarity
    pub fn normalized(&self) -> Vec<f32> {
        if self.norm == 0.0 {
            return vec![0.0; self.dim as usize];
        }
        self.to_f32().iter().map(|v| v / self.norm).collect()
    }

    /// Compute cosine similarity with another vector
    pub fn cosine_similarity(&self, other: &[f32]) -> f32 {
        let self_vec = self.to_f32();
        if self_vec.len() != other.len() {
            return 0.0;
        }

        let dot: f32 = self_vec.iter().zip(other.iter()).map(|(a, b)| a * b).sum();
        let other_norm = compute_l2_norm(other);

        if self.norm == 0.0 || other_norm == 0.0 {
            return 0.0;
        }

        dot / (self.norm * other_norm)
    }
}

/// Compute the L2 (Euclidean) norm of a vector
fn compute_l2_norm(vector: &[f32]) -> f32 {
    vector.iter().map(|v| v * v).sum::<f32>().sqrt()
}

impl PartialEq for Embedding {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
            && self.chunk_id == other.chunk_id
            && self.model_hash == other.model_hash
            && self.dim == other.dim
    }
}

impl Eq for Embedding {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_creation() {
        let chunk_id = Uuid::new_v4();
        let vector = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let model_hash = [0u8; 32];

        let emb = Embedding::new(chunk_id, &vector, model_hash);

        assert_eq!(emb.chunk_id, chunk_id);
        assert_eq!(emb.dim, 5);
        assert!(emb.norm > 0.0);
    }

    #[test]
    fn test_cosine_similarity() {
        let emb = Embedding::new(Uuid::new_v4(), &[1.0, 0.0, 0.0], [0u8; 32]);

        // Same direction = 1.0
        assert!((emb.cosine_similarity(&[1.0, 0.0, 0.0]) - 1.0).abs() < 0.01);

        // Orthogonal = 0.0
        assert!((emb.cosine_similarity(&[0.0, 1.0, 0.0])).abs() < 0.01);

        // Opposite = -1.0
        assert!((emb.cosine_similarity(&[-1.0, 0.0, 0.0]) + 1.0).abs() < 0.01);
    }

    #[test]
    fn test_f16_roundtrip() {
        let original = vec![0.12345, -0.6789, 1.234];
        let emb = Embedding::new(Uuid::new_v4(), &original, [0u8; 32]);
        let recovered = emb.to_f32();

        // f16 has ~3 decimal places of precision
        for (o, r) in original.iter().zip(recovered.iter()) {
            assert!((o - r).abs() < 0.01);
        }
    }
}
