//! Chunk node representing a text segment from a document
//!
//! Per CP-011: Chunks use byte offsets and lengths for precise positioning.
//! Per CP-001: Chunk ID is STABLE - does not include text content.

use crate::text::normalize;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A chunk of text extracted from a document
///
/// Documents are split into overlapping chunks for embedding.
/// Each chunk tracks its position within the source document.
///
/// Per CP-011: Uses byte-based offsets (not character-based) for accurate
/// slicing back to original document content.
///
/// Per CP-001: Chunk ID is STABLE - ID = hash(doc_id + sequence) only.
/// This ensures re-chunking with different parameters produces the same IDs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Chunk {
    /// Unique identifier for this chunk (BLAKE3-16 of doc_id + sequence) - STABLE
    pub id: Uuid,

    /// Parent document ID
    pub doc_id: Uuid,

    /// The actual text content (canonicalized)
    pub text: String,

    /// Byte offset within the source document (u64 for large files)
    pub byte_offset: u64,

    /// Length of this chunk in bytes (u64 for large files)
    pub byte_length: u64,

    /// Sequence number within the document (0-indexed)
    pub sequence: u32,

    /// Hash of the canonicalized text content for verification
    pub text_hash: [u8; 32],
}

impl Chunk {
    /// Create a new chunk with automatic ID generation.
    ///
    /// Per CP-001: Chunk ID is STABLE - does NOT include text.
    /// This ensures re-chunking with different parameters produces same IDs.
    /// Content is verified via text_hash field.
    pub fn new(doc_id: Uuid, text: String, byte_offset: u64, sequence: u32) -> Self {
        // Per CP-003: Canonicalize text before hashing for determinism
        let canonical_text = normalize(&text);
        let text_hash = *blake3::hash(canonical_text.as_bytes()).as_bytes();

        // Per CP-001: ID = hash(doc_id + sequence) - STABLE, does NOT include text
        let id_bytes = crate::id::generate_composite_id(&[
            doc_id.as_bytes(),
            &sequence.to_le_bytes(),
        ]);
        let id = Uuid::from_bytes(id_bytes);

        let byte_length = text.len() as u64;

        Self {
            id,
            doc_id,
            text: canonical_text, // Store canonicalized text
            byte_offset,
            byte_length,
            sequence,
            text_hash,
        }
    }

    /// Create a chunk from already-canonicalized text (for internal use)
    #[doc(hidden)]
    pub fn from_canonical(doc_id: Uuid, text: String, byte_offset: u64, sequence: u32) -> Self {
        let text_hash = *blake3::hash(text.as_bytes()).as_bytes();

        // Per CP-001: ID = hash(doc_id + sequence) - STABLE
        let id_bytes = crate::id::generate_composite_id(&[
            doc_id.as_bytes(),
            &sequence.to_le_bytes(),
        ]);
        let id = Uuid::from_bytes(id_bytes);

        let byte_length = text.len() as u64;

        Self {
            id,
            doc_id,
            text,
            byte_offset,
            byte_length,
            sequence,
            text_hash,
        }
    }

    /// Get the text hash as a hex string
    pub fn text_hash_hex(&self) -> String {
        self.text_hash
            .iter()
            .map(|b| format!("{:02x}", b))
            .collect()
    }

    /// Approximate token count (rough estimate: 4 chars per token)
    pub fn approx_tokens(&self) -> usize {
        self.text.len() / 4
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_creation() {
        let doc_id = Uuid::new_v4();
        let chunk = Chunk::new(
            doc_id,
            "This is a test chunk.".to_string(),
            0,
            0,
        );

        assert_eq!(chunk.doc_id, doc_id);
        assert_eq!(chunk.byte_offset, 0);
        assert_eq!(chunk.sequence, 0);
    }

    #[test]
    fn test_chunk_id_stable() {
        let doc_id = Uuid::nil();
        let text = "Test text";

        // Same doc_id + sequence = same chunk ID regardless of text content
        let chunk1 = Chunk::new(doc_id, text.to_string(), 0, 1);
        let chunk2 = Chunk::new(doc_id, "Different text".to_string(), 0, 1);

        assert_eq!(chunk1.id, chunk2.id);

        // Different sequence = different ID
        let chunk3 = Chunk::new(doc_id, text.to_string(), 0, 2);
        assert_ne!(chunk1.id, chunk3.id);
    }

    #[test]
    fn test_chunk_id_determinism() {
        let doc_id = Uuid::nil();
        let text = "Test text";
        let seq = 1;

        let chunk1 = Chunk::new(doc_id, text.to_string(), 0, seq);
        let chunk2 = Chunk::new(doc_id, text.to_string(), 0, seq);

        assert_eq!(chunk1.id, chunk2.id);
    }

    #[test]
    fn test_text_canonicalized() {
        let doc_id = Uuid::nil();

        // Text with different whitespace should canonicalize to same
        let chunk1 = Chunk::new(doc_id, "Hello   \nWorld".to_string(), 0, 0);
        let chunk2 = Chunk::new(doc_id, "Hello\nWorld".to_string(), 0, 0);

        // Text is canonicalized
        assert_eq!(chunk1.text, "Hello\nWorld\n");
        assert_eq!(chunk1.text, chunk2.text);

        // But ID is still stable (not based on text)
        assert_eq!(chunk1.id, chunk2.id);
    }

    #[test]
    fn test_approx_tokens() {
        let chunk = Chunk::new(
            Uuid::new_v4(),
            "A".repeat(400), // ~100 tokens
            0,
            0,
        );

        assert_eq!(chunk.approx_tokens(), 100);
    }

    #[test]
    fn test_byte_offset() {
        let doc_id = Uuid::new_v4();
        let chunk = Chunk::new(doc_id, "Test".to_string(), 14, 1);

        assert_eq!(chunk.byte_offset, 14);
    }
}
