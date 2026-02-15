//! CFS Core - Data models and traits for the Cognitive Filesystem
//!
//! This crate defines the fundamental types used across all CFS components:
//! - Document, Chunk, Embedding nodes
//! - Graph edges and relationships
//! - State roots and Merkle commitments
//! - Cognitive diffs for synchronization
//! - Hybrid Logical Clocks for causal ordering
//! - Canonical ID generation (BLAKE3-16)

mod document;
pub mod chunk;
pub mod context_assembler;
mod embedding;
mod edge;
mod state;
mod diff;
mod error;
pub mod text;
pub mod id;
pub mod hlc;

pub use document::Document;
pub use chunk::Chunk;
pub use embedding::Embedding;
pub use edge::{Edge, EdgeKind};
pub use state::StateRoot;
pub use diff::{CognitiveDiff, DiffMetadata, ExecutionTrace, Operation, EmbeddingInput};
pub use error::{CfsError, Result};
pub use context_assembler::{
    ContextAssembler, ScoredChunk, AssembledContext, ContextMetadata, TokenBudget, ContextChunk,
};
pub use hlc::Hlc;

/// Re-export commonly used types
pub mod prelude {
    pub use crate::{
        Document, Chunk, Embedding, Edge, EdgeKind,
        StateRoot, CognitiveDiff, DiffMetadata,
        CfsError, Result, Hlc,
    };
}
