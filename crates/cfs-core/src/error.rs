//! Error types for CFS operations

use thiserror::Error;
use serde::Serialize;

/// Unified error type for CFS operations
#[derive(Error, Debug, Serialize)]
#[serde(tag = "type", content = "message")]
pub enum CfsError {
    #[error("IO error: {0}")]
    #[serde(serialize_with = "serialize_io_error")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Database error: {0}")]
    Database(String),

    #[error("Crypto error: {0}")]
    Crypto(String),

    #[error("Invalid state: {0}")]
    InvalidState(String),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Embedding error: {0}")]
    Embedding(String),

    #[error("Parse error: {0}")]
    Parse(String),

    #[error("Sync error: {0}")]
    Sync(String),

    #[error("Verification failed: {0}")]
    Verification(String),

    #[error("Inference error: {0}")]
    Inference(String),
}

fn serialize_io_error<S>(error: &std::io::Error, serializer: S) -> std::result::Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    serializer.serialize_str(&error.to_string())
}

/// Result type alias using CfsError
pub type Result<T> = std::result::Result<T, CfsError>;
