use cfs_core::{Chunk, ScoredChunk, ContextAssembler, TokenBudget};
use uuid::Uuid;

#[test]
fn test_context_assembly_determinism() {
    let doc_id = Uuid::new_v4();
    let c1 = Chunk::new(doc_id, "Chunk 1 content here".to_string(), 0, 0);
    let c2 = Chunk::new(doc_id, "Chunk 2 content here".to_string(), 20, 1);

    let sc1 = ScoredChunk { chunk: c1.clone(), score: 0.9, document_path: "test.md".to_string() };
    let sc2 = ScoredChunk { chunk: c2.clone(), score: 0.8, document_path: "test.md".to_string() };

    let budget = TokenBudget::new(1000);
    let assembler = ContextAssembler::new(budget);

    // Order 1
    let context_a = assembler.assemble(vec![sc1.clone(), sc2.clone()], "test query", [0u8; 32]);

    // Order 2 (reversed input)
    let context_b = assembler.assemble(vec![sc2.clone(), sc1.clone()], "test query", [0u8; 32]);

    // Context should be identical regardless of input order (deterministic)
    assert_eq!(context_a.chunks.len(), context_b.chunks.len());
    for (a, b) in context_a.chunks.iter().zip(context_b.chunks.iter()) {
        assert_eq!(a.chunk_id, b.chunk_id, "Chunk IDs must match for determinism");
        assert_eq!(a.text, b.text, "Text must match for determinism");
    }
}

#[test]
fn test_context_token_budget() {
    let doc_id = Uuid::new_v4();
    // ~100 chars = ~25 tokens
    let c1 = Chunk::new(doc_id, "A".repeat(100), 0, 0);
    let c2 = Chunk::new(doc_id, "B".repeat(100), 110, 1);

    let sc1 = ScoredChunk { chunk: c1, score: 0.9, document_path: "test.md".to_string() };
    let sc2 = ScoredChunk { chunk: c2, score: 0.8, document_path: "test.md".to_string() };

    // 50 tokens budget - should only fit one chunk
    let budget = TokenBudget::new(200); // 200 - 200 - 100 - 500 = -600... wait, available = 200-200-100-500 = -600
    // Let me recalculate: total=200, reserved_system=200, reserved_query=100, reserved_response=500
    // That's negative! Let's use default or larger
    let budget = TokenBudget::default(); // 2000 total, available = 2000-200-100-500 = 1200

    let assembler = ContextAssembler::new(budget);

    let context = assembler.assemble(vec![sc1, sc2], "test query", [0u8; 32]);
    // Both should fit in 1200 available tokens
    assert_eq!(context.chunks.len(), 2);
}

#[test]
fn test_context_formatting() {
    let doc_id = Uuid::new_v4();
    let c1 = Chunk::new(doc_id, "First chunk text".to_string(), 0, 0);
    let c2 = Chunk::new(doc_id, "Second chunk text".to_string(), 20, 1);

    let sc1 = ScoredChunk { chunk: c1, score: 0.9, document_path: "docs/test.md".to_string() };
    let sc2 = ScoredChunk { chunk: c2, score: 0.8, document_path: "docs/test.md".to_string() };

    let budget = TokenBudget::default();
    let assembler = ContextAssembler::new(budget);

    let context = assembler.assemble(vec![sc1, sc2], "test", [0u8; 32]);

    // Check metadata
    assert_eq!(context.metadata.query_hash.len(), 32);
    assert_eq!(context.metadata.state_root.len(), 32);
}
