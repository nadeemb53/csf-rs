//! CFS Core - Data models and traits for the Cognitive Filesystem
//!
//! This crate defines the fundamental types used across all CFS components:
//! - Document, Chunk, Embedding nodes
//! - Graph edges and relationships
//! - State roots and Merkle commitments
//! - Cognitive diffs for synchronization

mod document;
pub mod chunk;
pub mod context_assembler;
mod embedding;
mod edge;
mod state;
mod diff;
mod error;

pub use document::Document;
pub use chunk::Chunk;
pub use embedding::Embedding;
pub use edge::{Edge, EdgeKind};
pub use state::StateRoot;
pub use diff::{CognitiveDiff, DiffMetadata};
pub use error::{CfsError, Result};
pub use context_assembler::{ContextAssembler, ScoredChunk};

/// Namespaces for deterministic UUIDs (v5)
pub mod namespaces {
    use uuid::Uuid;
    /// Namespace for Document nodes: 550e8400-e29b-41d4-a716-446655440000
    pub const DOCUMENT: Uuid = Uuid::from_u128(0x550e8400_e29b_41d4_a716_446655440000);
    /// Namespace for Chunk nodes: 550e8400-e29b-41d4-a716-446655440001
    pub const CHUNK: Uuid = Uuid::from_u128(0x550e8400_e29b_41d4_a716_446655440001);
    /// Namespace for Embedding nodes: 550e8400-e29b-41d4-a716-446655440002
    pub const EMBEDDING: Uuid = Uuid::from_u128(0x550e8400_e29b_41d4_a716_446655440002);
}

/// Re-export commonly used types
pub mod prelude {
    pub use crate::{
        Document, Chunk, Embedding, Edge, EdgeKind,
        StateRoot, CognitiveDiff, DiffMetadata,
        CfsError, Result,
    };
}
