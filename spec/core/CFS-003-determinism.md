# CFS-003: Determinism & Math

> **Spec Version**: 1.0.0-draft
> **Status**: Draft
> **Category**: Core
> **Requires**: CFS-001, CFS-002

## Synopsis

This specification defines the mechanisms by which CFS achieves **true determinism**—the property that identical inputs always produce identical outputs across all devices, platforms, and time. This is the foundational guarantee that enables trustless verification of the semantic substrate.

## Motivation

Determinism is the **core invariant** upon which all CFS guarantees depend:

| Guarantee | Depends On |
|-----------|------------|
| State verification | Same inputs → same Merkle root |
| Multi-device sync | Same operations → same state |
| Audit trail | Reproducible execution |
| Intelligence accountability | Same context → traceable response |

Without true determinism, devices will silently diverge, Merkle roots will mismatch, and the entire verification model collapses.

### The Problem

Achieving determinism in a semantic system is hard because:

1. **Machine Learning is Non-Deterministic by Default**: Floating-point operations, parallel execution, and model updates all introduce variation
2. **Platforms Differ**: ARM and x86 handle IEEE 754 edge cases differently
3. **Time is Relative**: Wall clocks drift, network delays vary
4. **Concurrency is Chaotic**: Without coordination, parallel operations interleave unpredictably

CFS solves these problems through a combination of:

- **Canonical representations** (eliminate ambiguity)
- **Content-addressed artifacts** (pin exact versions)
- **Deterministic algorithms** (reproducible execution)
- **Logical ordering** (remove wall-clock dependency)
- **Verified execution** (detect divergence immediately)

## Technical Specification

### 1. The Determinism Invariant

**Definition**: CFS is deterministic if and only if:

```
∀ device_A, device_B:
    Apply(State_0, [Op_1, Op_2, ..., Op_n]) on device_A
    = Apply(State_0, [Op_1, Op_2, ..., Op_n]) on device_B
```

Where:
- `State_0` is the genesis state (empty substrate)
- `Op_i` is the i-th operation (add document, delete chunk, etc.)
- `Apply` is the state transition function
- Equality means **byte-identical Merkle roots**

This is the **same model used by blockchains**: deterministic state machines with content-addressed operations.

### 2. Canonical Data Representation

All data in CFS MUST have a single, unambiguous binary representation.

#### 2.1 Text Canonicalization

```
function canonicalize_text(text: String) -> CanonicalText:
    // 1. Unicode normalization (NFC)
    text = unicode_nfc_normalize(text)

    // 2. Line ending normalization (LF only)
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    // 3. Whitespace normalization
    text = text.lines()
              .map(|line| line.trim_end())  // Remove trailing whitespace
              .join("\n")

    // 4. Ensure single trailing newline
    text = text.trim_end() + "\n"

    // 5. Encode as UTF-8
    return CanonicalText {
        bytes: text.as_bytes(),
        hash: BLAKE3(text.as_bytes()),
    }
```

**Guarantees**:
- Identical semantic content → identical bytes
- Hash is computed over canonical bytes
- No platform-dependent encoding

#### 2.2 Strict Integer/SoftFloat Representation

CFS operates on a strictly **integer-only substrate** or requires **Software-Defined Floating Point (SoftFloat)**. Hardware floating-point units (FPU) are **PROHIBITED** for any state-affecting calculation due to IEEE 754 inconsistencies across architectures (x86 EXTended precision, ARM NEON flushing behaviors, FMA differences).

**Fixed-Point Schema**:
- **Embeddings**: `i16` (Range: -32768 to 32767) representing values in `[-1.0, 1.0]`
- **Scores**: `i32` scaled by `1,000,000` (6 decimal precision)
- **Probabilities**: `u32` scaled by `1,000,000` (Range: 0 to 1,000,000)

**Normalization Rule**:
Normalization MUST NOT use hardware `sqrt` or division. It MUST use:
1.  **SoftFloat Library**: A bit-exact software emulation of IEEE 754 (e.g., `berkeley-softfloat`, `rust-softfloat`).
2.  **OR Integer-Scaled Math**: Newton-Raphson integer square root with fixed scaling.

**Robust Quantization**:
To handle model output jitter (e.g., `0.5000001` vs `0.4999999`), we employ a "Dead-Zone" optimization before committing to the canonical state.

```rust
/// Canonical embedding is a vector of i16 values
struct CanonicalEmbedding {
    values: Vec<i16>,     // The source of truth
    hash: [u8; 32],       // BLAKE3(cast_to_bytes(values))
    dimension: u16,
}

function canonicalize_embedding(raw_vector: Vec<SoftFloat64>) -> CanonicalEmbedding:
    // 1. L2 Normalize (using Software Float or Integer Math)
    // HARDWARE FPU IS BANNED HERE
    norm = soft_sqrt(soft_sum(v * v for v in raw_vector))
    normalized = raw_vector.map(|v| soft_div(v, norm))

    // 2. Robust Quantization with Dead-Zone
    quantized = normalized.map(|v| {
        scaled = soft_mul(v, 32767.0)
        // If within epsilon of a rounding boundary, snap to lower magnitude
        // This prevents 0.5 toggling between 0 and 1 across architectures
        if is_near_boundary(scaled, EPSILON):
            return floor(scaled)
        
        return soft_round_half_to_even(scaled) as i16
    })

    // 3. Hash the i16 bytes directly (Little-Endian)
    bytes = quantized.flat_map(|v| v.to_le_bytes())
    
    return CanonicalEmbedding {
        values: quantized,
        hash: BLAKE3(bytes),
        dimension: raw_vector.len(),
    }
```

**Key Insight**: By forbidding hardware FPU for the normalization and quantization step, we guarantee that if two devices start with the same raw model logits, they derive the exact same `i16` vector.

#### 2.3 Serialization Order

All collections MUST be serialized in canonical order:

```
function canonicalize_collection<T: Hashable>(items: Vec<T>) -> Vec<T>:
    // Sort by content hash (BLAKE3 of canonical representation)
    return items.sort_by(|a, b| {
        hash_a = BLAKE3(a.to_canonical_bytes())
        hash_b = BLAKE3(b.to_canonical_bytes())
        return hash_a.cmp(hash_b)
    })
```

**No Insertion Order Dependency**: The canonical order is always derivable from content.

### 3. Content-Addressed Artifacts

All external dependencies MUST be pinned by content hash.

#### 3.1 Model Manifest

```
struct ModelManifest {
    model_id: String,                    // e.g., "cfs/minilm-v1"
    version: String,                     // Semantic version
    weights_hash: [u8; 32],              // BLAKE3 of weights file
    tokenizer_hash: [u8; 32],            // BLAKE3 of tokenizer.json
    config_hash: [u8; 32],               // BLAKE3 of config.json
    manifest_hash: [u8; 32],             // BLAKE3 of this manifest

    // Reproducibility metadata
    training_seed: u64,
    training_data_hash: [u8; 32],
    architecture: String,
    quantization: Option<QuantizationSpec>,
}
```

**Model Verification**:

```
function verify_model(model_path: Path, expected: ModelManifest) -> Result<()>:
    // 1. Hash all model files
    actual_weights_hash = BLAKE3(read_file(model_path / "model.safetensors"))
    actual_tokenizer_hash = BLAKE3(read_file(model_path / "tokenizer.json"))
    actual_config_hash = BLAKE3(read_file(model_path / "config.json"))

    // 2. Verify against manifest
    if actual_weights_hash != expected.weights_hash:
        return Err(ModelMismatch::Weights)
    if actual_tokenizer_hash != expected.tokenizer_hash:
        return Err(ModelMismatch::Tokenizer)
    if actual_config_hash != expected.config_hash:
        return Err(ModelMismatch::Config)

    return Ok(())
```

**Model Distribution**:

Models are distributed via content-addressed storage:

```
// Model URL is derived from manifest hash
model_url = f"ipfs://{manifest.manifest_hash}"
// OR
model_url = f"https://models.cfs.dev/v1/{manifest.manifest_hash}"
```

This ensures all devices download **exactly the same bytes**.

#### 3.2 Embedding Identity

Embeddings are uniquely identified by their inputs:

```
function compute_embedding_id(
    text_hash: [u8; 32],
    model_manifest_hash: [u8; 32],
    embedding_version: u32
) -> UUID:
    // Embedding ID = hash of (text, model, version)
    input = text_hash || model_manifest_hash || embedding_version.to_le_bytes()
    return UUIDv5(EMBEDDING_NAMESPACE, BLAKE3(input))
```

**Key Property**: If two devices compute embeddings for the same text with the same model, they MUST get the same embedding ID.

### 4. Deterministic Embedding Generation

The embedding pipeline MUST be fully deterministic.

#### 4.1 Tokenization

```
function deterministic_tokenize(text: CanonicalText, tokenizer: Tokenizer) -> TokenIds:
    // 1. Apply canonical tokenization
    tokens = tokenizer.encode(text.bytes)

    // 2. Verify tokenizer version
    assert(BLAKE3(tokenizer.vocab) == expected_tokenizer_hash)

    // 3. Add special tokens in fixed positions
    tokens = [CLS_ID] + tokens + [SEP_ID]

    // 4. Truncate deterministically (from end)
    if tokens.len() > MAX_SEQ_LEN:
        tokens = tokens[..MAX_SEQ_LEN-1] + [SEP_ID]

    return TokenIds {
        ids: tokens,
        hash: BLAKE3(tokens.to_le_bytes()),
    }
```

#### 4.2 Canonical vs. Fast Inference

We acknowledge two modes of inference with distinct guarantees.

**Mode A: Canonical Inference (Verification & Publishing)**
To produce embeddings that enter the **OpsLog** and **Merkle Tree**, inference MUST be bit-exact.
- **Requirement**: MUST run on **WASM (WebAssembly)** or a **Software-Defined float/integer kernel**.
- **Hardware Acceleration**: DISALLOWED (No AVX, No NEON, No Metal/CUDA) unless strict bit-exactness is proven (which is rarely possible).
- **Use Case**: Indexing new content, verifying remote ops.
- **Performance**: Slower (10-50x), but correct.

**Mode B: Fast Inference (Local User Experience)**
To generate chat responses or ephemeral search queries.
- **Requirement**: None. Use Metal, CUDA, CoreML.
- **Hardware Acceleration**: ALLOWED.
- **Use Case**: Chatting with the bot, local-only search queries.
- **Constraint**: These outputs **NEVER** enter the Merkle tree directly.

```rust
function generate_canonical_embedding(
    text: CanonicalText,
    model: WasmModelBox
) -> CanonicalEmbedding:
    // 1. Run inference in WASM sandbox
    // WASM guarantees stack machine behavior is identical bit-for-bit
    raw_vector = model.run_wasm_inference(text)
    
    // 2. Canonicalize (SoftFloat normalization)
    return canonicalize_embedding(raw_vector)
```

**Implementation Strategy**:
Compile a small embedding model (e.g., `all-MiniLM-L6-v2` or `bg-base-en-v1.5`) to WASM. This 30MB-100MB blob is part of the protocol verification kit.

#### 4.3 Post-Processing

```
function deterministic_postprocess(raw: RawEmbedding) -> CanonicalEmbedding:
    // 1. L2 normalize (deterministic)
    norm = sqrt(kahan_sum(raw.vector.map(|v| v * v)))
    normalized = raw.vector.map(|v| v / norm)

    // 2. Quantize to fixed-point i16
    pub fn to_i16_quantized(val: f32) -> i16 {
        let scaled = val * 32767.0;
        let rounded = scaled.round_ties_even(); // Deterministic rounding
        rounded.clamp(-32767.0, 32767.0) as i16
    }
    
    quantized = normalized.map(to_i16_quantized)

    // 3. Compute canonical hash
    bytes = quantized.flat_map(|v| v.to_le_bytes())

    return CanonicalEmbedding {
        values_i16: quantized,
        bytes: bytes,
        hash: BLAKE3(bytes),
    }
```

### 5. Deterministic Similarity Search

Vector similarity MUST be computed deterministically.

#### 5.1 Fixed-Point Dot Product

```
function deterministic_dot_product(a: CanonicalEmbedding, b: CanonicalEmbedding) -> i64:
    assert(a.values_i16.len() == b.values_i16.len())

    // Integer dot product (no floating-point)
    sum: i64 = 0
    for i in 0..a.values_i16.len():
        sum += (a.values_i16[i] as i64) * (b.values_i16[i] as i64)

    return sum
```

**Similarity Score**:

```
function deterministic_similarity(a: CanonicalEmbedding, b: CanonicalEmbedding) -> i64:
    // Since vectors are normalized, dot product ≈ cosine similarity
    // Range: [-32767² × dim, +32767² × dim]
    return deterministic_dot_product(a, b)
```

**Ranking**: Higher dot product = more similar. No floating-point division needed for ranking.

**Note**: Since embeddings are `i16` (max 32767), the dot product of two vectors of dimension `d` can reach `32767 * 32767 * d`. For `d=768` (BERT), this is `~8.2 * 10^11`, which fits comfortably in `i64` (max `9 * 10^18`).

#### 5.2 Transient Indexing (HNSW)

**CRITICAL ARCHITECTURAL CHANGE**: The HNSW Graph is **NOT** part of the canonical state.

**Rationale**: HNSW construction is inherently sensitive to insertion order and random seed. While "Deterministic HNSW" is possible, it is fragile and expensive to verify (you must rebuild the whole graph from scratch to verify the root).

**New Design**:
1.  **Canonical State**: A **Flat, Sorted List** of embeddings.
    -   `Vec<(EmbeddingId, Vector)>`
    -   Sorted by `EmbeddingId`.
    -   Merkle Tree hashes this list.
2.  **Runtime Index**: An HNSW index (or other ANN index) built *locally* from the canonical list.

```rust
struct CanonicalState {
    // The SOURCE OF TRUTH
    embeddings: Vec<CanonicalEmbedding>, // Sorted by ID
    merkle_root: [u8; 32],
}

struct RuntimeIndex {
    // The PERFORMANCE CACHE (Ephemeral)
    // Can be blown away and rebuilt at any time
    hnsw: HNSW<CanonicalEmbedding>,
}

function sync_step(local_state: &mut CanonicalState, diff: Diff) {
    // 1. Apply diff to valid sorted list
    local_state.apply(diff)
    
    // 2. Update Merkle Root (Cheap, log(N) for leaves)
    local_state.recompute_root()
    
    // 3. Async Background Task: Update Runtime Index
    spawn_background_worker(|| {
        local_hnsw.insert(diff.new_embeddings)
        local_hnsw.remove(diff.removed_ids)
    })
}
```

**Verification**:
To verify state `S`, a device simply checks if it holds the same **Set** of embeddings. It does NOT need to check if its HNSW graph has the exact same link structure. This creates a robust separation between **Truth** (State) and **Acceleration** (Index).

#### 5.3 Deterministic Search

```
function deterministic_search(
    query: CanonicalEmbedding,
    index: HNSWIndex,
    k: usize
) -> Vec<(EmbeddingId, i64)>:
    // 1. Search with fixed ef
    candidates = index.search(query.values_i16, k, ef=50)

    // 2. Sort results deterministically
    // Primary: similarity (descending)
    // Secondary: embedding hash (ascending) for tiebreaker
    sorted = candidates.sort_by(|a, b| {
        cmp = b.similarity.cmp(a.similarity)
        if cmp == Equal:
            return a.id.cmp(b.id)
        return cmp
    })

    return sorted
```

### 6. Deterministic Lexical Search

FTS ranking MUST be deterministic across SQLite versions.

#### 6.1 Integer BM25

BM25 ranking MUST use fixed-point arithmetic.

```
function integer_bm25(
    term_freq: u32,
    doc_len: u32,
    avg_doc_len_scaled: u32, // scaled by 1000
    doc_freq: u32,
    total_docs: u32
) -> i32: // Scaled by 1,000,000
    k1 = 1200    // 1.2 * 1000
    b = 750      // 0.75 * 1000
    
    // IDF: uses integer log approximation
    // idf(q) = log((N - n + 0.5) / (n + 0.5) + 1)
    idf_num = (total_docs - doc_freq) * 1000 + 500
    idf_denom = doc_freq * 1000 + 500
    idf_val = integer_log(idf_num * 1000 / idf_denom) // returns scaled log
    
    // TF component
    // tf = ((k1 + 1) * freq) / (k1 * (1 - b + b * L/avgL) + freq)
    num = term_freq * (k1 + 1000)
    
    // Denom calculation in fixed point
    len_ratio = (doc_len * 1000) / avg_doc_len_scaled
    b_part = (b * len_ratio) / 1000
    k_part = (k1 * (1000 - b + b_part)) / 1000
    denom = k_part + term_freq * 1000
    
    tf_val = (num * 1000) / denom
    
    return (idf_val * tf_val) / 1000
```

#### 6.2 Deterministic FTS Index

```
function build_deterministic_fts(chunks: Vec<Chunk>) -> FTSIndex:
    index = FTSIndex::new()

    // 1. Canonical tokenization
    for chunk in chunks.sort_by(|c| c.id):
        tokens = canonical_tokenize_for_fts(chunk.text)
        index.add(chunk.id, tokens)

    // 2. Precompute corpus statistics
    index.compute_statistics()  // avg_doc_len, doc_freq, etc.

    return index

function canonical_tokenize_for_fts(text: CanonicalText) -> Vec<String>:
    // 1. Lowercase
    lower = text.to_lowercase()

    // 2. Split on non-alphanumeric
    tokens = lower.split(|c| !c.is_alphanumeric())
                  .filter(|t| t.len() > 0)

    // 3. Remove stop words (fixed list)
    tokens = tokens.filter(|t| !STOP_WORDS.contains(t))

    // 4. Apply Porter stemming (deterministic algorithm)
    tokens = tokens.map(|t| porter_stem(t))

    return tokens.collect()
```

### 7. Deterministic Context Assembly

Context assembly MUST produce identical results for identical queries.

```
function deterministic_context_assembly(
    query: CanonicalText,
    semantic_results: Vec<(ChunkId, i64)>,  // Integer similarities
    lexical_results: Vec<(ChunkId, i64)>,   // Integer BM25 scores
    graph: GraphStore,
    budget: TokenBudget
) -> CanonicalContext:
    // 1. RRF fusion with integer arithmetic
    scores: HashMap<ChunkId, i64> = HashMap::new()

    for (rank, (id, _)) in semantic_results.enumerate():
        // RRF score scaled by 1000000 for precision
        rrf_score = 1000000 / (60 + rank as i64 + 1)
        scores[id] = scores.get(id, 0) + rrf_score

    for (rank, (id, _)) in lexical_results.enumerate():
        rrf_score = 1000000 / (60 + rank as i64 + 1)
        scores[id] = scores.get(id, 0) + rrf_score

    // 2. Sort by score (descending), then by ID (ascending) for ties
    sorted = scores.entries()
        .sort_by(|(id_a, score_a), (id_b, score_b)| {
            cmp = score_b.cmp(score_a)
            if cmp == Equal:
                return id_a.cmp(id_b)  // Deterministic tiebreaker
            return cmp
        })

    // 3. Greedy packing
    context = CanonicalContext::new()
    for (chunk_id, score) in sorted:
        chunk = graph.get_chunk(chunk_id)
        tokens = count_tokens(chunk.text)

        if context.total_tokens + tokens <= budget.available:
            context.add(chunk, score)
        else:
            break

    // 4. Compute context hash
    context.hash = BLAKE3(context.chunks.map(|c| c.id).concat())

    return context
```

### 8. Logical Time and Ordering

CFS uses **Hybrid Logical Clocks (HLC)** instead of wall clocks.

#### 8.1 HLC Structure

```
struct HybridTimestamp {
    wall_time: u64,      // Physical time (milliseconds since epoch)
    logical: u32,        // Logical counter
    device_id: UUID,     // Originating device
}
```

**Comparison**:

```
function compare_hlc(a: HybridTimestamp, b: HybridTimestamp) -> Ordering:
    // 1. Compare wall time
    if a.wall_time != b.wall_time:
        return a.wall_time.cmp(b.wall_time)

    // 2. Compare logical counter
    if a.logical != b.logical:
        return a.logical.cmp(b.logical)

    // 3. Compare device ID (deterministic tiebreaker)
    return a.device_id.cmp(b.device_id)
```

#### 8.2 HLC Operations

```
function hlc_now(clock: &mut HLC, device_id: UUID) -> HybridTimestamp:
    wall = system_time_millis()

    if wall > clock.last_wall_time:
        clock.last_wall_time = wall
        clock.logical = 0
    else:
        clock.logical += 1

    return HybridTimestamp {
        wall_time: clock.last_wall_time,
        logical: clock.logical,
        device_id: device_id,
    }

function hlc_receive(clock: &mut HLC, remote: HybridTimestamp, device_id: UUID) -> HybridTimestamp:
    wall = system_time_millis()

    if wall > clock.last_wall_time && wall > remote.wall_time:
        clock.last_wall_time = wall
        clock.logical = 0
    else if clock.last_wall_time > remote.wall_time:
        clock.logical += 1
    else if remote.wall_time > clock.last_wall_time:
        clock.last_wall_time = remote.wall_time
        clock.logical = remote.logical + 1
    else:
        clock.logical = max(clock.logical, remote.logical) + 1

    return HybridTimestamp {
        wall_time: clock.last_wall_time,
        logical: clock.logical,
        device_id: device_id,
    }
```

**Property**: HLC provides a **total ordering** of all events across all devices, independent of wall clock drift.

**Drift Protection**: If `wall > system_time + MAX_DRIFT`, the clock MUST panic or block until system time catches up. This prevents "time travel" attacks.

### 9. Deterministic State Transitions

Every state change is an **Operation** that transitions the substrate deterministically.

#### 9.1 Operation Types

```
enum Operation {
    AddDocument {
        document: Document,
        chunks: Vec<Chunk>,
        embeddings: Vec<CanonicalEmbedding>,
        timestamp: HybridTimestamp,
    },

    RemoveDocument {
        document_id: UUID,
        timestamp: HybridTimestamp,
    },

    UpdateDocument {
        document_id: UUID,
        new_chunks: Vec<Chunk>,
        new_embeddings: Vec<CanonicalEmbedding>,
        timestamp: HybridTimestamp,
    },
}
```

#### 9.2 Operation Hashing

Each operation has a deterministic hash:

```
function hash_operation(op: Operation) -> [u8; 32]:
    canonical_bytes = match op:
        AddDocument { document, chunks, embeddings, timestamp } =>
            serialize_canonical([
                "ADD_DOCUMENT",
                document.to_canonical_bytes(),
                chunks.sort_by(|c| c.id).map(|c| c.to_canonical_bytes()),
                embeddings.sort_by(|e| e.hash).map(|e| e.bytes),
                timestamp.to_bytes(),
            ])

        RemoveDocument { document_id, timestamp } =>
            serialize_canonical([
                "REMOVE_DOCUMENT",
                document_id.to_bytes(),
                timestamp.to_bytes(),
            ])

        UpdateDocument { ... } =>
            // Similar pattern

    return BLAKE3(canonical_bytes)
```

#### 9.3 Deterministic Apply

```
function apply_operation(state: SubstrateState, op: Operation) -> SubstrateState:
    // 1. Verify operation is valid for current state
    validate_operation(state, op)?

    // 2. Apply changes
    new_state = match op:
        AddDocument { document, chunks, embeddings, .. } =>
            state
                .with_document(document)
                .with_chunks(chunks)
                .with_embeddings(embeddings)
                .with_edges(derive_edges(document, chunks, embeddings))

        RemoveDocument { document_id, .. } =>
            state.without_document(document_id)  // Cascades

        UpdateDocument { ... } =>
            // Remove then add (atomic)

    // 3. Recompute Merkle root
    new_state.merkle_root = compute_merkle_root(new_state)

    // 4. Record operation in log
    new_state.operation_log.push(OperationLogEntry {
        op_hash: hash_operation(op),
        prev_root: state.merkle_root,
        new_root: new_state.merkle_root,
        timestamp: op.timestamp(),
    })

    return new_state
```

### 10. Verified Execution

Devices MUST be able to verify each other's operations.

#### 10.1 Execution Trace

Each operation produces an execution trace:

```
struct ExecutionTrace {
    operation: Operation,
    prev_state_root: [u8; 32],
    new_state_root: [u8; 32],

    // Intermediate hashes for debugging divergence
    document_subtree_root: [u8; 32],
    chunk_subtree_root: [u8; 32],
    embedding_subtree_root: [u8; 32],
    edge_subtree_root: [u8; 32],

    // Embedding computation proof
    embedding_inputs: Vec<EmbeddingInput>,  // (text_hash, model_hash)
    embedding_outputs: Vec<[u8; 32]>,       // Output hashes

    timestamp: HybridTimestamp,
    device_id: UUID,
    signature: [u8; 64],
}
```

#### 10.2 Trace Verification

```
function verify_execution_trace(
    trace: ExecutionTrace,
    trusted_model_manifest: ModelManifest
) -> Result<()>:
    // 1. Verify signature
    if !verify_signature(trace, trace.device_id):
        return Err(InvalidSignature)

    // 2. Verify model was correct
    for (input, output) in zip(trace.embedding_inputs, trace.embedding_outputs):
        if input.model_hash != trusted_model_manifest.manifest_hash:
            return Err(UntrustedModel)

    // 3. Re-execute operation locally (optional, for high-security)
    local_result = apply_operation(load_state(trace.prev_state_root), trace.operation)

    if local_result.merkle_root != trace.new_state_root:
        return Err(DeterminismViolation {
            expected: trace.new_state_root,
            got: local_result.merkle_root,
            divergence_point: find_divergence(trace, local_result),
        })

    return Ok(())
```

#### 10.3 Divergence Detection

When Merkle roots don't match, identify where they diverged:

```
function find_divergence(
    expected_trace: ExecutionTrace,
    actual_state: SubstrateState
) -> DivergencePoint:
    // Compare subtree roots to narrow down
    if expected_trace.document_subtree_root != actual_state.document_subtree_root:
        return compare_documents(expected_trace, actual_state)

    if expected_trace.chunk_subtree_root != actual_state.chunk_subtree_root:
        return compare_chunks(expected_trace, actual_state)

    if expected_trace.embedding_subtree_root != actual_state.embedding_subtree_root:
        // This is the most likely source of divergence
        return compare_embeddings(expected_trace, actual_state)

    if expected_trace.edge_subtree_root != actual_state.edge_subtree_root:
        return compare_edges(expected_trace, actual_state)

    return DivergencePoint::Unknown
```

### 11. Multi-Device Conflict Resolution

When devices operate concurrently, conflicts MUST be resolved deterministically.

#### 11.1 Conflict Detection

```
function detect_conflicts(
    local_ops: Vec<Operation>,
    remote_ops: Vec<Operation>
) -> Vec<Conflict>:
    conflicts = []

    for local_op in local_ops:
        for remote_op in remote_ops:
            if operations_conflict(local_op, remote_op):
                conflicts.push(Conflict {
                    local: local_op,
                    remote: remote_op,
                })

    return conflicts

function operations_conflict(a: Operation, b: Operation) -> bool:
    // Same document touched by different operations
    match (a, b):
        (AddDocument { document: d1, .. }, AddDocument { document: d2, .. }) =>
            d1.path == d2.path  // Same path = conflict

        (UpdateDocument { document_id: id1, .. }, UpdateDocument { document_id: id2, .. }) =>
            id1 == id2  // Same document = conflict

        (RemoveDocument { document_id: id1, .. }, UpdateDocument { document_id: id2, .. }) =>
            id1 == id2  // Remove vs update = conflict

        _ => false
```

#### 11.2 Deterministic Resolution

CFS uses **Last-Writer-Wins with HLC** for conflict resolution:

```
function resolve_conflict(conflict: Conflict) -> Operation:
    // Compare timestamps using HLC ordering
    local_ts = conflict.local.timestamp()
    remote_ts = conflict.remote.timestamp()

    winner = if compare_hlc(remote_ts, local_ts) == Greater:
        conflict.remote
    else if compare_hlc(local_ts, remote_ts) == Greater:
        conflict.local
    else:
        // Timestamps equal (extremely rare): use operation hash as tiebreaker
        if hash_operation(conflict.remote) < hash_operation(conflict.local):
            conflict.remote
        else:
            conflict.local

    return winner
```

**Alternative: CRDT-based Resolution**

For richer semantics, document content can use CRDTs:

```
struct CRDTDocument {
    // Each character has a unique ID
    characters: Vec<(CharacterId, char)>,
    tombstones: HashSet<CharacterId>,
}

struct CharacterId {
    timestamp: HybridTimestamp,
    position_hint: u64,  // For ordering
}

// Merge is deterministic and commutative
function merge_crdt(a: CRDTDocument, b: CRDTDocument) -> CRDTDocument:
    merged_chars = union(a.characters, b.characters)
    merged_tombstones = union(a.tombstones, b.tombstones)

    // Remove tombstoned characters
    live_chars = merged_chars.filter(|(id, _)| !merged_tombstones.contains(id))

    // Sort by CharacterId for deterministic order
    sorted = live_chars.sort_by(|(id_a, _), (id_b, _)| compare_char_id(id_a, id_b))

    return CRDTDocument { characters: sorted, tombstones: merged_tombstones }
```

### 12. Intelligence Scope: Context vs. Reasoning

We strictly distinguish between **Deterministic Context** (Input) and **Non-Deterministic Reasoning** (Output).

#### 12.1 The Promise: Exact Context
CFS guarantees that for a given query, **every device will assemble the EXACT same context**.

```
Query: "Fiscal Policy 2024"
Device A Context: [Chunk #123 (Score 98), Chunk #456 (Score 85)]
Device B Context: [Chunk #123 (Score 98), Chunk #456 (Score 85)]
```
This is achieved via the deterministic semantic/lexical search and ranking pipelines defined in Sections 5 and 6.

#### 12.2 The Reality: Ephemeral Reasoning
The LLM response itself (the "Reasoning") is **NOT** part of the canonical state and is **NOT** synced.

**Why?**
1.  **Hardware Divergence**: Even with fixed seeds, `Model(X)` varies on different GPUs.
2.  **Ephemerality**: The "Answer" is a transient view for the user. The "Knowledge" is in the chunks.

**Protocol**:
1.  **Hash the Input**: The `ContextHash` is recorded.
2.  **Trust the Input**: We verify that the AI was *fed* the correct data.
3.  **Display the Output**: The generated text is shown to the user but never effectively "signed" as a global truth.

```rust
struct IntelligenceEvent {
    query: String,
    context_hash: [u8; 32], // VERIFIED GLOBAL TRUTH
    generated_text: String, // Ephemeral, Local-Only
}
```

This pivot acknowledges that while we can control the *library* (CFS), we cannot strictly control the *oracle* (The LLM) without massive performance penalties. We choose to make the **Substrate** deterministic, and the **Intelligence** accountable (via Context Hashing).

### 13. Desired Properties

### 13.1 Byte-Level Determinism (State)

**Property**: Identical inputs produce byte-identical **Substrate State**.
```
∀ input: Apply(State, input) -> State' 
// State' is bit-exact across all devices
```

### 2. Cross-Platform Consistency

**Property**: The same operations produce the same results on any platform.

```
∀ platform ∈ {x86, ARM, WASM}:
    execute(op, platform_x86) = execute(op, platform_ARM) = execute(op, platform_WASM)
```

### 3. Temporal Consistency

**Property**: Operations can be replayed in the future with identical results.

```
∀ op, t1, t2 where t2 > t1:
    replay(op, at_time=t1) = replay(op, at_time=t2)
```

### 4. Verified Reproducibility

**Property**: Any device can verify any other device's operations.

```
∀ device_A, device_B, op:
    verify(device_B, trace_from(device_A, op)) = Ok
```

## Implementation Checklist

| Component | Determinism Mechanism | Priority |
|-----------|----------------------|----------|
| Text canonicalization | NFC + LF + trim | P0 |
| Embedding quantization | i16 fixed-point | P0 |
| Model verification | Content-addressed manifest | P0 |
| Similarity computation | Integer dot product | P0 |
| HNSW construction | Sorted insertion + seeded PRNG | P0 |
| BM25 scoring | Canonical integer formula | P1 |
| Context assembly | Integer RRF + sorted tiebreakers | P0 |
| Timestamp ordering | Hybrid Logical Clocks | P0 |
| Conflict resolution | LWW with HLC | P1 |
| LLM sampling | Seeded PRNG | P2 |

## Test Vectors

### Embedding Determinism Test

```
Input:
    text = "The quick brown fox"
    model = ModelManifest { hash: 0xabc123... }

Expected (on all platforms):
    embedding.hash = 0x7f8a9b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a
    embedding.values_i16[0..4] = [7823, -5124, 12456, -8901]
```

### Cross-Platform Verification

```
// Run on x86
result_x86 = apply_operation(genesis_state, add_document("test.md", "Hello"))

// Run on ARM
result_arm = apply_operation(genesis_state, add_document("test.md", "Hello"))

// Verify
assert(result_x86.merkle_root == result_arm.merkle_root)
```

## References

- [IEEE 754-2019: Floating-Point Arithmetic](https://ieeexplore.ieee.org/document/8766229)
- [Hybrid Logical Clocks](https://cse.buffalo.edu/tech-reports/2014-04.pdf)
- [CRDTs: Conflict-free Replicated Data Types](https://hal.inria.fr/inria-00609399/document)
- [Kahan Summation Algorithm](https://en.wikipedia.org/wiki/Kahan_summation_algorithm)
- [Content-Addressed Storage](https://en.wikipedia.org/wiki/Content-addressable_storage)
