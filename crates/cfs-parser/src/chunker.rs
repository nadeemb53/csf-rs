//! Text chunking for embedding preparation

use cfs_core::{Chunk, Result};
use uuid::Uuid;

/// Configuration for chunking
#[derive(Debug, Clone)]
pub struct ChunkConfig {
    /// Target chunk size in characters
    pub chunk_size: usize,
    /// Overlap between chunks in characters
    pub overlap: usize,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            chunk_size: 1000, // ~250 tokens
            overlap: 200,    // ~50 tokens
        }
    }
}

/// Chunker for splitting text into overlapping segments
pub struct Chunker {
    config: ChunkConfig,
}

impl Default for Chunker {
    fn default() -> Self {
        Self::new(ChunkConfig::default())
    }
}

impl Chunker {
    /// Create a new chunker with the given config
    pub fn new(config: ChunkConfig) -> Self {
        Self { config }
    }

    /// Split text into overlapping chunks
    pub fn chunk(&self, doc_id: Uuid, text: &str) -> Result<Vec<Chunk>> {
        let mut chunks = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        let total_len = chars.len();

        if total_len == 0 {
            return Ok(chunks);
        }

        let mut offset = 0usize;
        let mut seq = 0u32;

        while offset < total_len {
            // Calculate chunk end
            let end = (offset + self.config.chunk_size).min(total_len);

            // Try to find a good break point (end of sentence/paragraph)
            let chunk_end = self.find_break_point(&chars, offset, end, total_len);

            // Extract chunk text
            let chunk_text: String = chars[offset..chunk_end].iter().collect();
            let chunk_text = chunk_text.trim().to_string();

            if !chunk_text.is_empty() {
                chunks.push(Chunk::new(
                    doc_id,
                    chunk_text,
                    offset as u64,
                    seq,
                ));
                seq += 1;
            }

            // Move offset with overlap
            if chunk_end >= total_len {
                break;
            }

            // Only overlap if the chunk is larger than the overlap amount.
            // Otherwise, we'd be jumping backwards or staying still.
            offset = if chunk_end > offset + self.config.overlap {
                chunk_end - self.config.overlap
            } else {
                chunk_end
            };
        }

        Ok(chunks)
    }

    /// Find a good break point near the target end
    fn find_break_point(
        &self,
        chars: &[char],
        start: usize,
        target_end: usize,
        total_len: usize,
    ) -> usize {
        if target_end >= total_len {
            return total_len;
        }

        // Look for paragraph break first
        for i in (start..target_end).rev() {
            if chars[i] == '\n' && i + 1 < total_len && chars[i + 1] == '\n' {
                return i + 2;
            }
        }

        // Look for sentence end
        for i in (start..target_end).rev() {
            if (chars[i] == '.' || chars[i] == '!' || chars[i] == '?')
                && i + 1 < total_len
                && chars[i + 1].is_whitespace()
            {
                return i + 1;
            }
        }

        // Look for any whitespace
        for i in (start..target_end).rev() {
            if chars[i].is_whitespace() {
                return i + 1;
            }
        }

        // Fall back to hard cut
        target_end
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_text() {
        let chunker = Chunker::default();
        let chunks = chunker.chunk(Uuid::new_v4(), "").unwrap();
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_short_text() {
        let chunker = Chunker::default();
        let chunks = chunker.chunk(Uuid::new_v4(), "Short text.").unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "Short text.");
    }

    #[test]
    fn test_long_text_chunking() {
        let chunker = Chunker::new(ChunkConfig {
            chunk_size: 100,
            overlap: 20,
        });

        let text = "A".repeat(250);
        let chunks = chunker.chunk(Uuid::new_v4(), &text).unwrap();

        assert!(chunks.len() > 1);

        // Check sequence numbers
        for (i, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk.sequence, i as u32);
        }
    }

    #[test]
    fn test_sentence_boundary() {
        let chunker = Chunker::new(ChunkConfig {
            chunk_size: 50,
            overlap: 10,
        });

        let text = "First sentence. Second sentence. Third sentence.";
        let chunks = chunker.chunk(Uuid::new_v4(), text).unwrap();

        // Should break at sentence boundaries
        assert!(chunks[0].text.ends_with('.') || chunks[0].text.ends_with("sentence"));
    }
}
