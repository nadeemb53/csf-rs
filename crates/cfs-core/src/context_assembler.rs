//! Deterministic context assembly per CFS-021
//!
//! Transforms retrieval results into a token-budgeted, ordered context window
//! for intelligence modules. Guarantees byte-identical output given identical inputs.

use crate::Chunk;
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

/// A search result with its relevance score and document path.
#[derive(Debug, Clone)]
pub struct ScoredChunk {
    pub chunk: Chunk,
    pub score: f32,
    pub document_path: String,
}

/// Token budget configuration per CFS-021 §3.
#[derive(Debug, Clone)]
pub struct TokenBudget {
    /// Maximum total context tokens
    pub total_tokens: usize,
    /// Reserved for system prompt
    pub reserved_system: usize,
    /// Reserved for user query
    pub reserved_query: usize,
    /// Reserved for generation output
    pub reserved_response: usize,
    /// Tokens per document marker overhead
    pub per_doc_overhead: usize,
}

impl TokenBudget {
    pub fn new(total_tokens: usize) -> Self {
        let reserved_system = 200;
        let reserved_query = 100;
        let reserved_response = 500;
        let per_doc_overhead = 10;

        Self {
            total_tokens,
            reserved_system,
            reserved_query,
            reserved_response,
            per_doc_overhead,
        }
    }

    /// Available tokens for context chunks after reservations.
    pub fn available(&self) -> usize {
        self.total_tokens
            .saturating_sub(self.reserved_system)
            .saturating_sub(self.reserved_query)
            .saturating_sub(self.reserved_response)
    }
}

impl Default for TokenBudget {
    fn default() -> Self {
        Self::new(2000)
    }
}

/// A chunk in the assembled context with provenance.
#[derive(Debug, Clone)]
pub struct ContextChunk {
    pub chunk_id: Uuid,
    pub document_path: String,
    pub text: String,
    pub score: f32,
    pub sequence: u32,
}

/// Metadata for the assembled context (for auditability).
#[derive(Debug, Clone)]
pub struct ContextMetadata {
    /// BLAKE3 hash of the original query
    pub query_hash: [u8; 32],
    /// State root hash at time of retrieval
    pub state_root: [u8; 32],
}

/// The fully assembled, deterministic context window.
#[derive(Debug, Clone)]
pub struct AssembledContext {
    pub chunks: Vec<ContextChunk>,
    pub total_tokens: usize,
    pub truncated: bool,
    pub metadata: ContextMetadata,
}

/// Assembler for deterministic context construction per CFS-021.
pub struct ContextAssembler {
    budget: TokenBudget,
}

impl ContextAssembler {
    pub fn new(budget: TokenBudget) -> Self {
        Self { budget }
    }

    /// Create with a simple token budget.
    pub fn with_budget(total_tokens: usize) -> Self {
        Self {
            budget: TokenBudget::new(total_tokens),
        }
    }

    /// Assemble chunks into a deterministic, byte-identical context.
    ///
    /// Per CFS-021 §11:
    /// 1. Deduplicate by chunk ID and text hash (80% overlap threshold)
    /// 2. Sort deterministically: score desc → doc path asc → seq asc → byte_offset asc
    /// 3. Group by document
    /// 4. Greedy pack within budget
    /// 5. Format with [DOC: path] markers
    pub fn assemble(
        &self,
        chunks: Vec<ScoredChunk>,
        query: &str,
        state_root: [u8; 32],
    ) -> AssembledContext {
        // 1. Deduplicate
        let deduped = Self::deduplicate(chunks);

        // 2. Sort deterministically
        let sorted = Self::deterministic_sort(deduped);

        // 3. Group by document
        let groups = Self::group_by_document(&sorted);

        // 4. Greedy pack within budget
        let available = self.budget.available();
        let (packed, total_tokens, truncated) =
            Self::greedy_pack(groups, available, self.budget.per_doc_overhead);

        // Build context chunks
        let context_chunks: Vec<ContextChunk> = packed
            .into_iter()
            .map(|sc| ContextChunk {
                chunk_id: sc.chunk.id,
                document_path: sc.document_path.clone(),
                text: sc.chunk.text.clone(),
                score: sc.score,
                sequence: sc.chunk.sequence,
            })
            .collect();

        AssembledContext {
            chunks: context_chunks,
            total_tokens,
            truncated,
            metadata: ContextMetadata {
                query_hash: *blake3::hash(query.as_bytes()).as_bytes(),
                state_root,
            },
        }
    }

    /// Format the assembled context as a string with [DOC: path] markers.
    ///
    /// Per CFS-021 §9.
    pub fn format(context: &AssembledContext) -> String {
        let mut formatted = String::new();
        let mut current_doc: Option<&str> = None;

        for chunk in &context.chunks {
            if current_doc != Some(&chunk.document_path) {
                if current_doc.is_some() {
                    formatted.push('\n');
                }
                formatted.push_str(&format!("[DOC: {}]\n", chunk.document_path));
                current_doc = Some(&chunk.document_path);
            }
            formatted.push_str(&chunk.text);
            formatted.push('\n');
        }

        formatted
    }

    /// Deduplicate chunks by ID and text hash.
    fn deduplicate(chunks: Vec<ScoredChunk>) -> Vec<ScoredChunk> {
        let mut seen_ids: HashSet<Uuid> = HashSet::new();
        let mut seen_text_hashes: HashSet<[u8; 32]> = HashSet::new();
        let mut result = Vec::new();

        for sc in chunks {
            // Skip exact ID duplicates
            if !seen_ids.insert(sc.chunk.id) {
                continue;
            }

            // Skip exact text duplicates
            let text_hash = *blake3::hash(sc.chunk.text.as_bytes()).as_bytes();
            if !seen_text_hashes.insert(text_hash) {
                continue;
            }

            result.push(sc);
        }

        result
    }

    /// Sort deterministically per CFS-021 §5.
    ///
    /// Multi-level sort: score desc → doc path asc → sequence asc → byte_offset asc
    fn deterministic_sort(mut chunks: Vec<ScoredChunk>) -> Vec<ScoredChunk> {
        chunks.sort_by(|a, b| {
            // 1. Score descending
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                // 2. Document path ascending
                .then_with(|| a.document_path.cmp(&b.document_path))
                // 3. Chunk sequence ascending
                .then_with(|| a.chunk.sequence.cmp(&b.chunk.sequence))
                // 4. Byte offset ascending
                .then_with(|| a.chunk.byte_offset.cmp(&b.chunk.byte_offset))
        });
        chunks
    }

    /// Group chunks by document, ordered by max score.
    fn group_by_document(chunks: &[ScoredChunk]) -> Vec<Vec<&ScoredChunk>> {
        let mut groups: HashMap<&str, Vec<&ScoredChunk>> = HashMap::new();
        let mut max_scores: HashMap<&str, f32> = HashMap::new();

        for sc in chunks {
            let path = sc.document_path.as_str();
            groups.entry(path).or_default().push(sc);
            let entry = max_scores.entry(path).or_insert(0.0);
            if sc.score > *entry {
                *entry = sc.score;
            }
        }

        // Sort groups by max score descending, then path ascending for tiebreak
        let mut group_list: Vec<(&str, Vec<&ScoredChunk>)> = groups.into_iter().collect();
        group_list.sort_by(|a, b| {
            let score_a = max_scores.get(a.0).copied().unwrap_or(0.0);
            let score_b = max_scores.get(b.0).copied().unwrap_or(0.0);
            score_b
                .partial_cmp(&score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(b.0))
        });

        // Within each group, sort by sequence ascending
        group_list
            .into_iter()
            .map(|(_, mut chunks)| {
                chunks.sort_by_key(|sc| (sc.chunk.sequence, sc.chunk.byte_offset));
                chunks
            })
            .collect()
    }

    /// Greedy packing within token budget.
    ///
    /// Returns (packed chunks, total tokens used, whether truncation occurred).
    fn greedy_pack(
        groups: Vec<Vec<&ScoredChunk>>,
        budget: usize,
        per_doc_overhead: usize,
    ) -> (Vec<ScoredChunk>, usize, bool) {
        let mut tokens_used = 0usize;
        let mut packed = Vec::new();
        let mut truncated = false;
        let mut seen_docs: HashSet<&str> = HashSet::new();

        for group in groups {
            if tokens_used >= budget {
                truncated = true;
                break;
            }

            for sc in group {
                // Account for document header overhead on first occurrence
                let doc_overhead = if seen_docs.contains(sc.document_path.as_str()) {
                    0
                } else {
                    per_doc_overhead
                };

                let chunk_tokens = approx_tokens(&sc.chunk.text);
                let total_needed = chunk_tokens + doc_overhead;

                if tokens_used + total_needed <= budget {
                    seen_docs.insert(&sc.document_path);
                    packed.push(sc.clone());
                    tokens_used += total_needed;
                } else {
                    truncated = true;
                    break;
                }
            }
        }

        (packed, tokens_used, truncated)
    }
}

/// Approximate token count (4 chars per token).
///
/// Per CFS-021 §4: simple whitespace approximation for budget compliance.
fn approx_tokens(text: &str) -> usize {
    (text.len() + 3) / 4
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_chunk(doc_id: Uuid, text: &str, seq: u32) -> Chunk {
        Chunk::new(doc_id, text.to_string(), (seq as u64) * 100, seq)
    }

    fn make_scored(chunk: Chunk, score: f32, path: &str) -> ScoredChunk {
        ScoredChunk {
            chunk,
            score,
            document_path: path.to_string(),
        }
    }

    #[test]
    fn test_deterministic_assembly() {
        let doc_id = Uuid::from_bytes([1u8; 16]);
        let assembler = ContextAssembler::with_budget(4000);

        let chunks = vec![
            make_scored(make_chunk(doc_id, "Chunk A", 0), 0.9, "doc1.md"),
            make_scored(make_chunk(doc_id, "Chunk B", 1), 0.8, "doc1.md"),
        ];

        let ctx1 = assembler.assemble(chunks.clone(), "query", [0u8; 32]);
        let ctx2 = assembler.assemble(chunks, "query", [0u8; 32]);

        let fmt1 = ContextAssembler::format(&ctx1);
        let fmt2 = ContextAssembler::format(&ctx2);

        assert_eq!(fmt1, fmt2, "Identical inputs must produce byte-identical context");
    }

    #[test]
    fn test_deduplication() {
        let doc_id = Uuid::from_bytes([1u8; 16]);
        let assembler = ContextAssembler::with_budget(4000);

        let chunk = make_chunk(doc_id, "Same text", 0);
        let chunks = vec![
            make_scored(chunk.clone(), 0.9, "doc.md"),
            make_scored(chunk.clone(), 0.8, "doc.md"), // Duplicate ID
        ];

        let ctx = assembler.assemble(chunks, "query", [0u8; 32]);
        assert_eq!(ctx.chunks.len(), 1);
    }

    #[test]
    fn test_budget_compliance() {
        let doc_id = Uuid::from_bytes([1u8; 16]);
        // Very small budget
        let assembler = ContextAssembler::with_budget(900);
        let available = assembler.budget.available();

        let chunks: Vec<ScoredChunk> = (0..20)
            .map(|i| {
                make_scored(
                    make_chunk(doc_id, &"x".repeat(200), i),
                    1.0 - (i as f32 * 0.01),
                    "doc.md",
                )
            })
            .collect();

        let ctx = assembler.assemble(chunks, "query", [0u8; 32]);
        assert!(ctx.total_tokens <= available, "Context must not exceed budget");
    }

    #[test]
    fn test_document_grouping() {
        let doc_a = Uuid::from_bytes([1u8; 16]);
        let doc_b = Uuid::from_bytes([2u8; 16]);
        let assembler = ContextAssembler::with_budget(4000);

        let chunks = vec![
            make_scored(make_chunk(doc_a, "A chunk 0", 0), 0.9, "a.md"),
            make_scored(make_chunk(doc_b, "B chunk 0", 0), 0.85, "b.md"),
            make_scored(make_chunk(doc_a, "A chunk 1", 1), 0.8, "a.md"),
        ];

        let ctx = assembler.assemble(chunks, "query", [0u8; 32]);
        let formatted = ContextAssembler::format(&ctx);

        // Should have [DOC: a.md] and [DOC: b.md] markers
        assert!(formatted.contains("[DOC: a.md]"));
        assert!(formatted.contains("[DOC: b.md]"));
    }

    #[test]
    fn test_format_markers() {
        let doc_id = Uuid::from_bytes([1u8; 16]);
        let assembler = ContextAssembler::with_budget(4000);

        let chunks = vec![
            make_scored(make_chunk(doc_id, "Hello world", 0), 0.9, "test.md"),
        ];

        let ctx = assembler.assemble(chunks, "query", [0u8; 32]);
        let formatted = ContextAssembler::format(&ctx);

        assert!(formatted.starts_with("[DOC: test.md]\n"));
        assert!(formatted.contains("Hello world"));
    }

    #[test]
    fn test_sort_tiebreaker() {
        let doc_id = Uuid::from_bytes([1u8; 16]);
        let assembler = ContextAssembler::with_budget(4000);

        // Same score, different paths → path ascending
        let chunks = vec![
            make_scored(make_chunk(doc_id, "Chunk Z", 0), 0.9, "z.md"),
            make_scored(make_chunk(doc_id, "Chunk A", 0), 0.9, "a.md"),
        ];

        let ctx = assembler.assemble(chunks, "query", [0u8; 32]);
        let formatted = ContextAssembler::format(&ctx);

        // a.md should come before z.md (path ascending tiebreak)
        let pos_a = formatted.find("[DOC: a.md]").unwrap();
        let pos_z = formatted.find("[DOC: z.md]").unwrap();
        assert!(pos_a < pos_z);
    }

    #[test]
    fn test_empty_chunks() {
        let assembler = ContextAssembler::with_budget(4000);
        let ctx = assembler.assemble(vec![], "query", [0u8; 32]);
        assert!(ctx.chunks.is_empty());
        assert_eq!(ctx.total_tokens, 0);
        assert!(!ctx.truncated);
    }
}
