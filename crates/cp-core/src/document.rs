//! Document node representing an ingested file
//!
//! Per CP-001: Documents use content-based IDs for determinism.
//! path_id is added for filesystem change detection.

use crate::text::normalize;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use uuid::Uuid;

/// A document node in the cognitive graph
///
/// Represents a single ingested file with its content hash
/// for change detection and deduplication.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Document {
    /// Unique identifier for this document (BLAKE3-16 of content_hash)
    pub id: Uuid,

    /// Path-based identifier for change detection (BLAKE3-16 of canonicalized path)
    pub path_id: Uuid,

    /// Original file path (relative to watched root)
    pub path: PathBuf,

    /// BLAKE3 hash of file contents (canonicalized)
    pub hash: [u8; 32],

    /// Merkle root of chunks (hierarchical hash)
    pub hierarchical_hash: [u8; 32],

    /// Last modification time (Unix timestamp)
    pub mtime: i64,

    /// File size in bytes
    pub size: u64,

    /// MIME type (e.g., "application/pdf", "text/markdown")
    pub mime_type: String,
}

impl Document {
    pub fn new(path: PathBuf, content: &[u8], mtime: i64) -> Self {
        let mime_type = mime_from_path(&path);

        // Per CP-003: Canonicalize content before hashing for determinism
        let text = String::from_utf8_lossy(content);
        let canonical_content = normalize(&text);
        let canonical_bytes = canonical_content.as_bytes();
        let content_hash = blake3::hash(canonical_bytes);

        // Per CP-001: ID is generated from content hash (content-based identity)
        let mut id_bytes = [0u8; 16];
        id_bytes.copy_from_slice(&content_hash.as_bytes()[0..16]);
        let id = Uuid::from_bytes(id_bytes);

        // Per CP-001: path_id is generated from canonicalized path (for change detection)
        let path_str = path.to_string_lossy();
        let canonical_path = normalize(&path_str);
        let path_id_bytes = blake3::hash(canonical_path.as_bytes());
        let mut path_id = [0u8; 16];
        path_id.copy_from_slice(&path_id_bytes.as_bytes()[0..16]);
        let path_id = Uuid::from_bytes(path_id);

        Self {
            id,
            path_id,
            path,
            hash: *content_hash.as_bytes(),
            hierarchical_hash: [0; 32], // Placeholder, computed after chunking
            mtime,
            size: content.len() as u64, // Original size for display
            mime_type,
        }
    }

    /// Update the hierarchical hash (Merkle root of chunks)
    pub fn set_hierarchical_hash(&mut self, hash: [u8; 32]) {
        self.hierarchical_hash = hash;
    }

    /// Compute Merkle hash from chunks for provable correctness
    pub fn compute_hierarchical_hash(chunk_hashes: &[[u8; 32]]) -> [u8; 32] {
        let mut section_hasher = blake3::Hasher::new();
        for hash in chunk_hashes {
            section_hasher.update(hash);
        }
        *section_hasher.finalize().as_bytes()
    }

    /// Check if the document content has changed
    pub fn content_changed(&self, new_content: &[u8]) -> bool {
        let text = String::from_utf8_lossy(new_content);
        let canonical = normalize(&text);
        let new_hash = blake3::hash(canonical.as_bytes());
        self.hash != *new_hash.as_bytes()
    }

    /// Get the document hash as a hex string
    pub fn hash_hex(&self) -> String {
        hex_encode(&self.hash)
    }
}

/// Infer MIME type from file extension
fn mime_from_path(path: &PathBuf) -> String {
    match path.extension().and_then(|e| e.to_str()) {
        Some("md") | Some("markdown") => "text/markdown".to_string(),
        Some("txt") => "text/plain".to_string(),
        Some("pdf") => "application/pdf".to_string(),
        Some("json") => "application/json".to_string(),
        Some("html") | Some("htm") => "text/html".to_string(),
        Some("rs") => "text/x-rust".to_string(),
        Some("py") => "text/x-python".to_string(),
        Some("js") => "text/javascript".to_string(),
        Some("ts") => "text/typescript".to_string(),
        _ => "application/octet-stream".to_string(),
    }
}

/// Encode bytes as lowercase hex
fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_document_creation() {
        let content = b"Hello, CP!";
        let doc = Document::new(
            PathBuf::from("test.md"),
            content,
            1234567890,
        );

        assert_eq!(doc.path, PathBuf::from("test.md"));
        // Original content is 10 bytes
        assert_eq!(doc.size, 10);
        assert_eq!(doc.mime_type, "text/markdown");
        assert_eq!(doc.hierarchical_hash, [0; 32]);

        // Verify ID generation (first 16 bytes of blake3 hash of canonicalized content)
        let canonical = normalize("Hello, CP!");
        let hash = blake3::hash(canonical.as_bytes());
        let expected_id = Uuid::from_bytes(hash.as_bytes()[0..16].try_into().unwrap());
        assert_eq!(doc.id, expected_id);
    }

    #[test]
    fn test_content_changed() {
        let content = b"Original content";
        let doc = Document::new(PathBuf::from("test.txt"), content, 0);

        assert!(!doc.content_changed(content));
        assert!(doc.content_changed(b"Modified content"));
    }

    #[test]
    fn test_path_id_deterministic() {
        let doc1 = Document::new(PathBuf::from("test.md"), b"content", 0);
        let doc2 = Document::new(PathBuf::from("test.md"), b"content", 0);

        // Same path = same path_id
        assert_eq!(doc1.path_id, doc2.path_id);

        // Different path = different path_id
        let doc3 = Document::new(PathBuf::from("other.md"), b"content", 0);
        assert_ne!(doc1.path_id, doc3.path_id);
    }

    #[test]
    fn test_content_id_deterministic() {
        let doc1 = Document::new(PathBuf::from("a.md"), b"hello", 0);
        let doc2 = Document::new(PathBuf::from("b.md"), b"hello", 0);

        // Same content = same ID regardless of path
        assert_eq!(doc1.id, doc2.id);
    }
}
