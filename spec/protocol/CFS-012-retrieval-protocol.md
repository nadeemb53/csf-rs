# CFS-012: Retrieval Protocol

> **Spec Version**: 1.0.0
> **Author**: Nadeem Bhati
> **Category**: Protocol
> **Requires**: CFS-001, CFS-002, CFS-010, CFS-003

## Synopsis

This specification defines the hybrid retrieval protocol for CFS, combining semantic (vector) search with lexical (keyword) search using Reciprocal Rank Fusion (RRF).

## Motivation

Hybrid retrieval addresses limitations of single-mode search:

1. **Semantic Search Limitations**: Struggles with exact identifiers, names, technical terms
2. **Lexical Search Limitations**: Misses synonyms, paraphrases, conceptual matches
3. **Hybrid Advantage**: Combines strengths of both approaches

## Technical Specification

### 1. Retrieval Architecture

```
                    ┌─────────────┐
                    │   Query     │
                    └──────┬──────┘
                           │
           ┌───────────────┴───────────────┐
           │                               │
           v                               v
    ┌─────────────┐                 ┌─────────────┐
    │  Semantic   │                 │   Lexical   │
    │   Search    │                 │   Search    │
    │   (HNSW)    │                 │   (FTS5)    │
    └──────┬──────┘                 └──────┬──────┘
           │                               │
           v                               v
    ┌─────────────┐                 ┌─────────────┐
    │  Ranked     │                 │   Ranked    │
    │  Results    │                 │   Results   │
    └──────┬──────┘                 └──────┬──────┘
           │                               │
           └───────────────┬───────────────┘
                           │
                           v
                    ┌─────────────┐
                    │  RRF Fusion │
                    └──────┬──────┘
                           │
                           v
                    ┌─────────────┐
                    │   Final     │
                    │   Results   │
                    └─────────────┘
```

### 2. Semantic Search

Semantic search finds chunks by vector similarity.

#### Query Embedding (Canonical)

**CRITICAL**: Queries contributing to "Deterministic Context" MUST be embedded using the **Canonical Inference** path (WASM/SoftFloat) to ensure every device retrieves the *exact same* set of chunks for the same query.

```
function embed_query_canonical(query: String, model: EmbeddingModel) -> CanonicalEmbedding:
    // 1. Tokenize query
    tokens = tokenize(query)

    // 2. Forward pass (WASM / SoftFloat)
    hidden = model.forward_canonical(tokens)

    // 3. Pool and normalize (SoftFloat)
    pooled = mean_pooling(hidden)
    normalized = l2_normalize_canonical(pooled)
    
    // 4. Quantize to i16 (for search compatibility)
    return quantize_f32_to_i16(normalized)
```

#### Vector Search (Integer Math)

```
function semantic_search(
    query: Vec<i16>,
    hnsw: HNSWIndex,
    k: usize
) -> Vec<SearchResult>:
    // 1. Query HNSW index using Integer Dot Product
    // Returns (embedding_id, distance) pairs
    neighbors = hnsw.search_i16(query, k)

    // 2. Convert distance to similarity score (0.0 - 1.0)
    // HNSW distance is roughly inverse of similarity
    results = neighbors.map(|(id, dist)| {
        SearchResult {
            embedding_id: id,
            score: normalize_score(dist), 
        }
    })

    // 3. Sort by score descending
    return results.sort_by(|r| -r.score)
```

#### HNSW Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| ef_search | 50 | Search quality vs. speed tradeoff |
| k | 20 | Number of candidates to retrieve |

### 3. Lexical Search

Lexical search finds chunks by keyword matching.

#### Query Parsing

```
function parse_fts_query(query: String) -> FTSQuery:
    // 1. Tokenize
    tokens = query.split_whitespace()

    // 2. Handle quoted phrases
    // "exact phrase" -> exact_phrase as single token
    phrases = extract_quoted(query)

    // 3. Build FTS query
    terms = tokens.filter(|t| !is_stop_word(t))
    return FTSQuery { terms, phrases }
```

#### Full-Text Search

```
function lexical_search(
    query: FTSQuery,
    db: SQLite,
    limit: usize
) -> Vec<SearchResult>:
    // 1. Build FTS5 query string
    fts_query = query.terms.join(" OR ")
    for phrase in query.phrases:
        fts_query += f' "{phrase}"'

    // 2. Execute FTS query with BM25 ranking
    sql = """
        SELECT chunk_id, bm25(fts_chunks) as score
        FROM fts_chunks
        WHERE fts_chunks MATCH ?
        ORDER BY score
        LIMIT ?
    """
    rows = db.query(sql, [fts_query, limit])

    // 3. Normalize scores (BM25 is negative, lower = better)
    min_score = rows.map(|r| r.score).min()
    max_score = rows.map(|r| r.score).max()
    range = max_score - min_score

    results = rows.map(|r| SearchResult {
        chunk_id: r.chunk_id,
        score: (r.score - min_score) / range,  // Normalize to 0-1
    })

    return results
```

### 4. Reciprocal Rank Fusion (RRF)

RRF combines multiple ranked lists into a single ranking.

#### Algorithm

```
function rrf_fusion(
    semantic_results: Vec<SearchResult>,
    lexical_results: Vec<SearchResult>,
    k: f32 = 60.0
) -> Vec<SearchResult>:
    // 1. Build score maps indexed by chunk_id
    scores: HashMap<UUID, f32> = HashMap::new()

    // 2. Add semantic scores
    for (rank, result) in semantic_results.enumerate():
        // Get chunk_id from embedding
        chunk_id = get_chunk_id(result.embedding_id)
        rrf_score = 1.0 / (k + rank + 1)
        scores[chunk_id] = scores.get(chunk_id, 0) + rrf_score

    // 3. Add lexical scores
    for (rank, result) in lexical_results.enumerate():
        chunk_id = result.chunk_id
        rrf_score = 1.0 / (k + rank + 1)
        scores[chunk_id] = scores.get(chunk_id, 0) + rrf_score

    // 4. Convert to sorted results
    fused = scores.entries()
        .map(|(id, score)| SearchResult { chunk_id: id, score: score })
        .sort_by(|r| -r.score)

    return fused
```

#### Why k=60?

The constant k controls how much weight is given to top-ranked results:

| k Value | Effect |
|---------|--------|
| Low (10) | Top results dominate |
| Medium (60) | Balanced influence |
| High (100) | Flatter distribution |

k=60 is empirically effective for hybrid search.

#### RRF Score Example

```
Query: "blockchain consensus mechanism"

Semantic Results:        Lexical Results:
1. Chunk A (rank 0)      1. Chunk C (rank 0)
2. Chunk B (rank 1)      2. Chunk A (rank 1)
3. Chunk C (rank 2)      3. Chunk D (rank 2)

RRF Scores (k=60):
Chunk A: 1/(60+1) + 1/(60+2) = 0.0164 + 0.0161 = 0.0325
Chunk C: 1/(60+3) + 1/(60+1) = 0.0159 + 0.0164 = 0.0323
Chunk B: 1/(60+2) = 0.0161
Chunk D: 1/(60+3) = 0.0159

Final Ranking: A, C, B, D
```

### 5. Complete Hybrid Search

```
function hybrid_search(
    query: String,
    graph: GraphStore,
    model: EmbeddingModel,
    params: SearchParams
) -> Vec<RetrievalResult>:
    // 1. Embed query for semantic search
    query_vector = embed_query(query, model)

    // 2. Perform semantic search
    semantic_results = semantic_search(
        query_vector,
        graph.hnsw,
        params.semantic_k
    )

    // 3. Parse and perform lexical search
    fts_query = parse_fts_query(query)
    lexical_results = lexical_search(
        fts_query,
        graph.db,
        params.lexical_k
    )

    // 4. Fuse results with RRF
    fused = rrf_fusion(
        semantic_results,
        lexical_results,
        params.rrf_k
    )

    // 5. Retrieve full chunk data
    results = []
    for result in fused.take(params.top_k):
        chunk = graph.get_chunk(result.chunk_id)
        doc = graph.get_document(chunk.document_id)

        results.push(RetrievalResult {
            chunk: chunk,
            document: doc,
            score: result.score,
        })

    return results
```

#### Search Parameters

```
struct SearchParams {
    semantic_k: usize = 20,   // Candidates from semantic search
    lexical_k: usize = 20,    // Candidates from lexical search
    rrf_k: f32 = 60.0,        // RRF constant
    top_k: usize = 10,        // Final results to return
}
```

### 6. Result Ranking

#### Deterministic Ordering

When scores are equal, CFS uses deterministic tiebreakers:

```
function compare_results(a: RetrievalResult, b: RetrievalResult) -> Ordering:
    // 1. Primary: Score (descending)
    if a.score != b.score:
        return b.score.cmp(a.score)

    // 2. Secondary: Document path (ascending)
    if a.document.path != b.document.path:
        return a.document.path.cmp(b.document.path)

    // 3. Tertiary: Chunk sequence (ascending)
    return a.chunk.sequence.cmp(b.chunk.sequence)
```

This ensures identical queries always produce identical result orderings.

### 7. Query Modes

CFS supports multiple query modes:

#### Semantic-Only Mode

```
function semantic_only_search(query: String, graph: GraphStore) -> Vec<Result>:
    query_vector = embed_query(query)
    results = semantic_search(query_vector, graph.hnsw, k=10)
    return materialize_results(results, graph)
```

Use case: Conceptual queries where exact terms don't matter.

#### Lexical-Only Mode

```
function lexical_only_search(query: String, graph: GraphStore) -> Vec<Result>:
    fts_query = parse_fts_query(query)
    results = lexical_search(fts_query, graph.db, limit=10)
    return materialize_results(results, graph)
```

Use case: Searching for exact identifiers, error messages, code.

#### Hybrid Mode (Default)

```
function hybrid_search(query: String, graph: GraphStore) -> Vec<Result>:
    // As defined in Section 5
```

Use case: General-purpose search.

### 8. Filtering

CFS supports pre-retrieval and post-retrieval filtering.

#### Pre-Retrieval Filtering

```
function filtered_search(
    query: String,
    filters: Vec<Filter>,
    graph: GraphStore
) -> Vec<Result>:
    // 1. Apply filters to get candidate document IDs
    candidate_docs = apply_filters(filters, graph)

    // 2. Get embeddings only from candidate documents
    candidate_embs = get_embeddings_for_docs(candidate_docs, graph)

    // 3. Build temporary filtered HNSW
    filtered_hnsw = build_hnsw(candidate_embs)

    // 4. Search filtered index
    return hybrid_search_with_index(query, filtered_hnsw, graph)
```

#### Filter Types

```
enum Filter {
    DocumentPath(glob: String),     // e.g., "docs/**/*.md"
    MimeType(mime: String),         // e.g., "text/markdown"
    ModifiedAfter(timestamp: i64),  // Unix ms
    ModifiedBefore(timestamp: i64),
    ContentContains(text: String),  // Pre-filter by content
}
```

#### Post-Retrieval Filtering

```
function search_with_post_filter(
    query: String,
    filters: Vec<Filter>,
    graph: GraphStore
) -> Vec<Result>:
    // 1. Perform normal search with higher k
    results = hybrid_search(query, graph, params.with_k(100))

    // 2. Apply filters to results
    filtered = results.filter(|r| matches_filters(r, filters))

    // 3. Return top k
    return filtered.take(params.top_k)
```

### 9. Similarity Metrics

CFS uses **Integer Dot Product** for similarity search, which is equivalent to Cosine Similarity for normalized vectors but deterministic.

#### Integer Dot Product

```
function integer_dot_product(a: Vec<i16>, b: Vec<i16>) -> i32:
    // Accumulate in i32 to prevent overflow
    return sum(a[i] as i32 * b[i] as i32 for i in 0..a.len())
```

**Normalization**:
Since input vectors `a` and `b` are scaled by `S = 32767.0`, the dot product is scaled by `S^2`. To get the cosine similarity (0.0 to 1.0):

```
cosine_sim = integer_dot_product(a, b) / (32767.0 * 32767.0)
```

#### Why Integer Math?

| Metric | Properties | Use Case |
|--------|------------|----------|
| **Integer Dot** | **Deterministic** | **Canonical Search** |
| Cosine (f32) | Fast (SIMD) | Approximate Search |
| L2 Distance | Magnitude sensitive | Clustering |

Text embeddings benefit from scale invariance, which Dot Product preserves.

### 10. Performance Optimization

#### Query Caching

```
struct QueryCache {
    cache: LRU<String, Vec<UUID>>,  // query -> result IDs
    max_size: usize = 1000,
    ttl: Duration = 5 minutes,
}

function cached_search(query: String, cache: QueryCache, graph: GraphStore):
    // Check cache
    if let Some(ids) = cache.get(query):
        return materialize_ids(ids, graph)

    // Perform search
    results = hybrid_search(query, graph)

    // Cache results
    ids = results.map(|r| r.chunk.id)
    cache.put(query, ids)

    return results
```

#### Approximate Search

For large indexes, approximate search trades accuracy for speed:

```
function approximate_search(query: Vec<f32>, hnsw: HNSWIndex, k: usize):
    // Reduce ef for faster search
    hnsw.set_ef(20)  // Lower than default 50
    results = hnsw.search(query, k)
    hnsw.set_ef(50)  // Restore
    return results
```

## Desired Properties

### 1. Determinism

**Property**: Identical queries MUST produce identical results.

**Verification**:
```
∀ query: search(query) = search(query)
```

### 2. Relevance

**Property**: Results SHOULD be semantically related to the query.

**Metric**: Precision@K, Recall@K, NDCG

### 3. Diversity

**Property**: Results SHOULD cover different aspects of the query.

**Mechanism**: RRF naturally promotes diversity by combining different signals.

### 4. Efficiency

**Property**: Search SHOULD complete in sub-second time.

**Target**: < 100ms for typical queries on < 1M embeddings.

## Evaluation Metrics

### Precision@K

```
precision_at_k = relevant_retrieved / k
```

### Recall@K

```
recall_at_k = relevant_retrieved / total_relevant
```

### Normalized Discounted Cumulative Gain (NDCG)

```
DCG = sum(relevance[i] / log2(i + 2) for i in 0..k)
NDCG = DCG / ideal_DCG
```

## Test Vectors

### Hybrid Search Test

```
Query: "merkle tree verification"

Semantic Top 3:
1. "Merkle roots enable cryptographic verification..."
2. "Hash trees provide data integrity..."
3. "Blockchain uses merkle proofs..."

Lexical Top 3:
1. "Merkle tree structure allows verification..."
2. "The merkle root is computed..."
3. "Binary tree for merkle proofs..."

RRF Fused Top 3:
1. "Merkle tree structure allows verification..." (in both)
2. "Merkle roots enable cryptographic verification..."
3. "The merkle root is computed..."
```

### Exact Match Test

```
Query: "function calculate_hyper_parameter_v7"

Semantic: Poor results (no semantic meaning)
Lexical: Exact match found

Hybrid: Lexical result ranked first due to RRF
```

## References

- [Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods](https://dl.acm.org/doi/10.1145/1571941.1572114)
- [Approximate Nearest Neighbors in HNSW](https://arxiv.org/abs/1603.09320)
- [BM25: The Next Generation](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf)
