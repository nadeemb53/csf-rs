# CFS-021: Context Assembly

> **Spec Version**: 1.0.0
> **Author**: Nadeem Bhati
> **Category**: Application
> **Requires**: CFS-001, CFS-012, CFS-020

## Synopsis

This specification defines the deterministic context assembly algorithm for CFS. Context assembly transforms retrieval results into a token-budgeted, ordered context window for intelligence modules.

## Motivation

Context assembly is critical because:

1. **Token Limits**: LLMs have finite context windows
2. **Determinism**: Same query must produce same context
3. **Relevance Ordering**: Most relevant content should appear first
4. **Document Coherence**: Related chunks should be grouped
5. **Auditability**: Context selection must be reproducible

## Technical Specification

### 1. Assembly Pipeline

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌────────────┐
│  Retrieval  │───>│   Ranking    │───>│  Grouping   │───>│  Packing   │
│   Results   │    │  (Scores)    │    │  (By Doc)   │    │  (Budget)  │
└─────────────┘    └──────────────┘    └─────────────┘    └────────────┘
                                                                 │
                                                                 v
                                                          ┌────────────┐
                                                          │ Formatting │
                                                          │  (Markers) │
                                                          └────────────┘
                                                                 │
                                                                 v
                                                          ┌────────────┐
                                                          │  Assembled │
                                                          │   Context  │
                                                          └────────────┘
```

### 2. Input: Retrieval Results

```
struct RetrievalResult {
    chunk: Chunk,
    document: Document,
    score: f32,               // Combined RRF score
    semantic_score: f32,      // Vector similarity
    lexical_score: f32,       // FTS match score
}
```

### 3. Token Budget Configuration

```
struct TokenBudget {
    total_tokens: usize,          // Maximum context tokens (e.g., 2000)
    reserved_system: usize,       // Reserved for system prompt (e.g., 200)
    reserved_query: usize,        // Reserved for user query (e.g., 100)
    reserved_response: usize,     // Reserved for generation (e.g., 500)
    per_doc_overhead: usize,      // Tokens per document marker (e.g., 10)
    available: usize,             // Computed: total - reserved
}

impl TokenBudget {
    fn new(total_tokens: usize) -> Self:
        reserved_system = 200
        reserved_query = 100
        reserved_response = 500
        per_doc_overhead = 10

        available = total_tokens - reserved_system - reserved_query - reserved_response

        return TokenBudget {
            total_tokens,
            reserved_system,
            reserved_query,
            reserved_response,
            per_doc_overhead,
            available,
        }
}
```

### 4. Tokenization

Context assembly requires consistent token counting:

```
interface Tokenizer {
    fn count_tokens(text: String) -> usize
    fn truncate_to_tokens(text: String, max_tokens: usize) -> String
}

// Default: Simple whitespace approximation
struct SimpleTokenizer;

impl Tokenizer for SimpleTokenizer {
    fn count_tokens(text: String) -> usize:
        // Approximation: 4 characters per token on average
        return (text.len() + 3) / 4

    fn truncate_to_tokens(text: String, max_tokens: usize) -> String:
        max_chars = max_tokens * 4
        if text.len() <= max_chars:
            return text
        return text[..max_chars].trim() + "..."
}

// Production: Model-specific tokenizer
struct ModelTokenizer {
    vocab: HashMap<String, usize>,
    bpe_rules: Vec<MergeRule>,
}

impl Tokenizer for ModelTokenizer {
    fn count_tokens(text: String) -> usize:
        tokens = self.bpe_encode(text)
        return tokens.len()
}
```

### 5. Deterministic Ordering

Retrieval results are sorted deterministically:

```
function sort_results(results: Vec<RetrievalResult>) -> Vec<RetrievalResult>:
    // Multi-level sort for determinism
    return results.sort_by(|a, b| {
        // 1. Primary: Score (descending)
        cmp1 = b.score.partial_cmp(a.score)
        if cmp1 != Ordering::Equal:
            return cmp1

        // 2. Secondary: Document path (ascending, lexicographic)
        cmp2 = a.document.path.cmp(b.document.path)
        if cmp2 != Ordering::Equal:
            return cmp2

        // 3. Tertiary: Chunk sequence (ascending)
        cmp3 = a.chunk.sequence.cmp(b.chunk.sequence)
        if cmp3 != Ordering::Equal:
            return cmp3

        // 4. Quaternary: Chunk byte offset (ascending)
        return a.chunk.byte_offset.cmp(b.chunk.byte_offset)
    })
```

### 6. Document Grouping

Chunks from the same document are grouped for coherence:

```
struct DocumentGroup {
    document: Document,
    chunks: Vec<RetrievalResult>,
    total_tokens: usize,
    max_score: f32,
}

function group_by_document(results: Vec<RetrievalResult>) -> Vec<DocumentGroup>:
    groups: HashMap<UUID, DocumentGroup> = HashMap::new()

    for result in results:
        doc_id = result.document.id

        if !groups.contains(doc_id):
            groups[doc_id] = DocumentGroup {
                document: result.document.clone(),
                chunks: [],
                total_tokens: 0,
                max_score: 0.0,
            }

        group = groups.get_mut(doc_id)
        group.chunks.push(result.clone())
        group.total_tokens += tokenizer.count_tokens(result.chunk.text)
        group.max_score = max(group.max_score, result.score)

    // Sort groups by max score
    return groups.values()
        .sort_by(|a, b| b.max_score.partial_cmp(a.max_score))
```

### 7. Greedy Packing Algorithm

```
function pack_context(
    groups: Vec<DocumentGroup>,
    budget: TokenBudget,
    tokenizer: Tokenizer
) -> AssembledContext:
    assembled_chunks = []
    tokens_used = 0
    truncated = false

    for group in groups:
        // Check if we can fit any more
        if tokens_used >= budget.available:
            truncated = true
            break

        // Calculate tokens needed for this document
        doc_overhead = budget.per_doc_overhead

        // Sort chunks within group by sequence for reading order
        sorted_chunks = group.chunks.sort_by(|c| c.chunk.sequence)

        for result in sorted_chunks:
            chunk_tokens = tokenizer.count_tokens(result.chunk.text)

            // Check if chunk fits
            if tokens_used + doc_overhead + chunk_tokens <= budget.available:
                assembled_chunks.push(ContextChunk {
                    chunk_id: result.chunk.id,
                    document_path: result.document.path,
                    text: result.chunk.text,
                    score: result.score,
                    sequence: result.chunk.sequence,
                })

                tokens_used += chunk_tokens
                if assembled_chunks.len() == 1 || last_doc != result.document.id:
                    tokens_used += doc_overhead
                    last_doc = result.document.id

            else if tokens_used + doc_overhead < budget.available:
                // Partial fit: truncate chunk
                remaining = budget.available - tokens_used - doc_overhead
                truncated_text = tokenizer.truncate_to_tokens(result.chunk.text, remaining)

                assembled_chunks.push(ContextChunk {
                    chunk_id: result.chunk.id,
                    document_path: result.document.path,
                    text: truncated_text,
                    score: result.score,
                    sequence: result.chunk.sequence,
                })

                truncated = true
                break

    return AssembledContext {
        chunks: assembled_chunks,
        total_tokens: tokens_used,
        truncated: truncated,
        metadata: ContextMetadata { ... },
    }
```

### 8. Alternative Packing Strategies

#### 8.1 Interleaved Packing

Alternates between documents for diversity:

```
function interleaved_pack(groups: Vec<DocumentGroup>, budget: TokenBudget) -> AssembledContext:
    assembled = []
    group_iterators = groups.map(|g| g.chunks.iter())

    while tokens_used < budget.available:
        made_progress = false

        for iter in group_iterators:
            if let Some(result) = iter.next():
                if fits_in_budget(result, tokens_used, budget):
                    assembled.push(result)
                    tokens_used += token_count(result)
                    made_progress = true

        if !made_progress:
            break

    return AssembledContext { chunks: assembled, ... }
```

#### 8.2 Score-Weighted Packing

Prioritizes highest-scoring chunks regardless of document:

```
function score_weighted_pack(results: Vec<RetrievalResult>, budget: TokenBudget) -> AssembledContext:
    // Sort strictly by score
    sorted = results.sort_by(|a, b| b.score.cmp(a.score))

    assembled = []
    for result in sorted:
        if fits_in_budget(result, tokens_used, budget):
            assembled.push(result)
            tokens_used += token_count(result)
        else:
            break

    return AssembledContext { chunks: assembled, ... }
```

### 9. Context Formatting

```
function format_context(context: AssembledContext) -> String:
    formatted = ""
    current_doc = None

    for chunk in context.chunks:
        // Add document header if new document
        if current_doc != chunk.document_path:
            if current_doc != None:
                formatted += "\n"
            formatted += f"[DOC: {chunk.document_path}]\n"
            current_doc = chunk.document_path

        // Add chunk text
        formatted += chunk.text
        formatted += "\n"

    return formatted
```

#### Format Example

```
[DOC: docs/blockchain/consensus.md]
Proof of Stake (PoS) is a consensus mechanism where validators
are selected based on their stake in the network.

Validators must lock tokens as collateral, which can be slashed
for malicious behavior.

[DOC: docs/blockchain/merkle.md]
Merkle trees provide efficient verification of data integrity.
Each leaf node is a hash of a data block.

[DOC: docs/blockchain/consensus.md]
Byzantine Fault Tolerance (BFT) ensures the network can reach
consensus even with malicious participants.
```

### 10. Context Deduplication

Overlapping chunks are deduplicated:

```
function deduplicate_chunks(chunks: Vec<ContextChunk>) -> Vec<ContextChunk>:
    seen_text_hashes: HashSet<[u8; 32]> = HashSet::new()
    deduped = []

    for chunk in chunks:
        text_hash = BLAKE3(chunk.text.as_bytes())

        // Check for exact duplicates
        if seen_text_hashes.contains(text_hash):
            continue

        // Check for high overlap with existing chunks
        is_duplicate = false
        for existing in deduped:
            overlap = compute_overlap(chunk.text, existing.text)
            if overlap > 0.8:  // 80% overlap threshold
                is_duplicate = true
                break

        if !is_duplicate:
            seen_text_hashes.insert(text_hash)
            deduped.push(chunk)

    return deduped
```

### 11. Complete Assembly Function

```
function assemble_context(
    query: String,
    substrate: GraphStore,
    budget: TokenBudget,
    tokenizer: Tokenizer
) -> AssembledContext:
    // 1. Retrieve candidates
    results = hybrid_search(query, substrate, SearchParams {
        semantic_k: 50,
        lexical_k: 50,
        top_k: 100,  // Get more candidates for filtering
    })

    // 2. Deduplicate
    deduped = deduplicate_results(results)

    // 3. Sort deterministically
    sorted = sort_results(deduped)

    // 4. Group by document
    groups = group_by_document(sorted)

    // 5. Pack within budget
    context = pack_context(groups, budget, tokenizer)

    // 6. Add metadata
    context.metadata = ContextMetadata {
        query_hash: BLAKE3(query.as_bytes()),
        timestamp: now(),
        state_root: substrate.get_latest_state_root().hash,
    }

    return context
```

### 12. Determinism Verification

```
function verify_determinism(query: String, substrate: GraphStore) -> bool:
    // Assemble context twice
    context1 = assemble_context(query, substrate, budget, tokenizer)
    context2 = assemble_context(query, substrate, budget, tokenizer)

    // Compare chunk IDs
    ids1 = context1.chunks.map(|c| c.chunk_id)
    ids2 = context2.chunks.map(|c| c.chunk_id)

    if ids1 != ids2:
        return false

    // Compare formatted output
    text1 = format_context(context1)
    text2 = format_context(context2)

    return text1 == text2
```

## Desired Properties

### 1. Determinism

**Property**: Identical queries + identical substrate state MUST produce byte-identical context.

**Verification**:
```
∀ query, state: format(assemble(query, state)) = format(assemble(query, state))
```

### 2. Budget Compliance

**Property**: Assembled context MUST NOT exceed token budget.

**Verification**:
```
∀ context: context.total_tokens <= budget.available
```

### 3. Relevance Ordering

**Property**: Higher-scoring chunks SHOULD appear before lower-scoring chunks.

**Heuristic**: Top 3 chunks by score should be in first 5 assembled chunks.

### 4. Coherence

**Property**: Chunks from the same document SHOULD be grouped together.

**Metric**: Intra-document chunk adjacency > 0.7.

### 5. Coverage

**Property**: Context SHOULD maximize information coverage within budget.

**Metric**: Total unique tokens / budget > 0.9.

## Configuration Parameters

```
struct AssemblyConfig {
    packing_strategy: PackingStrategy,  // Greedy, Interleaved, ScoreWeighted
    dedup_threshold: f32,               // Overlap threshold for dedup (0.8)
    max_chunks_per_doc: Option<usize>,  // Limit chunks per document
    prefer_recent: bool,                // Boost recently modified documents
    include_metadata: bool,             // Include chunk metadata in output
}

enum PackingStrategy {
    Greedy,        // Fill by score, group by doc
    Interleaved,   // Alternate between documents
    ScoreWeighted, // Strictly by score
}
```

## Performance Optimization

### Caching

```
struct ContextCache {
    cache: LRU<CacheKey, AssembledContext>,
    max_size: usize,
}

struct CacheKey {
    query_hash: [u8; 32],
    state_root: [u8; 32],
    budget_hash: [u8; 32],
}

function cached_assemble(query, substrate, budget) -> AssembledContext:
    key = CacheKey {
        query_hash: BLAKE3(query),
        state_root: substrate.state_root(),
        budget_hash: BLAKE3(budget.to_bytes()),
    }

    if let Some(cached) = cache.get(key):
        return cached

    context = assemble_context(query, substrate, budget)
    cache.put(key, context)
    return context
```

### Parallel Processing

```
function parallel_assemble(query, substrate, budget) -> AssembledContext:
    // Parallel retrieval
    (semantic, lexical) = parallel(
        || semantic_search(query, substrate),
        || lexical_search(query, substrate)
    )

    // Merge results
    fused = rrf_fusion(semantic, lexical)

    // Sequential packing (must be deterministic)
    return pack_context(fused, budget)
```

## Test Vectors

### Packing Test

```
Input:
    results = [
        { chunk: "A", tokens: 50, score: 0.9 },
        { chunk: "B", tokens: 100, score: 0.85 },
        { chunk: "C", tokens: 30, score: 0.8 },
        { chunk: "D", tokens: 80, score: 0.75 },
    ]
    budget = 150 tokens

Output:
    context.chunks = ["A", "C"]  // 50 + 30 = 80 tokens (B wouldn't fit after A)
    context.total_tokens = 80
    context.truncated = false
```

### Determinism Test

```
Query: "blockchain consensus"
State Root: 0xabc123...

Run 1:
    chunks = [chunk_A, chunk_B, chunk_C]
    formatted = "[DOC: a.md]\nText A\n[DOC: b.md]\nText B\n..."
    hash = 0x789def...

Run 2:
    chunks = [chunk_A, chunk_B, chunk_C]  // Same order
    formatted = "[DOC: a.md]\nText A\n[DOC: b.md]\nText B\n..."  // Identical
    hash = 0x789def...  // Same hash

Result: PASS (deterministic)
```

### Grouping Test

```
Input:
    results = [
        { doc: "a.md", chunk: 1, score: 0.9 },
        { doc: "b.md", chunk: 1, score: 0.88 },
        { doc: "a.md", chunk: 2, score: 0.85 },
        { doc: "b.md", chunk: 2, score: 0.82 },
    ]

Grouped Output:
    Group 1 (a.md): [chunk_1, chunk_2]  // max_score = 0.9
    Group 2 (b.md): [chunk_1, chunk_2]  // max_score = 0.88

Packed Output (greedy):
    "[DOC: a.md]\nChunk 1\nChunk 2\n[DOC: b.md]\nChunk 1\nChunk 2\n"
```

## References

- [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172)
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- [Efficient Context Window Extension](https://arxiv.org/abs/2310.04433)
