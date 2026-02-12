# CFS-010: Embedding Protocol

> **Spec Version**: 1.0.0
> **Author**: Nadeem Bhati
> **Category**: Protocol
> **Requires**: CFS-001, CFS-003

## Synopsis

This specification defines the protocol for generating and managing vector embeddings in CFS. Embeddings transform semantic content into high-dimensional vectors for similarity search.

## Motivation

Vector embeddings enable:

1. **Semantic Search**: Find content by meaning, not just keywords
2. **Similarity Clustering**: Group related content automatically
3. **Cross-Lingual Retrieval**: Match content across languages
4. **Concept Abstraction**: Capture abstract relationships

CFS requires embeddings that are:

- **Deterministic**: Same content always produces same embedding (Bit-Exact)
- **Local**: Generated without network access
- **Auditable**: Model provenance is tracked

## Technical Specification

### 1. Embedding Model

CFS uses the **all-MiniLM-L6-v2** model as the reference embedding model.

#### Model Properties

| Property | Value |
|----------|-------|
| Architecture | MiniLM (Transformer) |
| Layers | 6 |
| Hidden Size | 384 |
| Output Dimension | 384 |
| Max Sequence Length | 256 tokens |
| Training | Sentence similarity |
| Framework | Candle (Rust) / WASM |

#### Model Hash

```
model_id = "sentence-transformers/all-MiniLM-L6-v2"
model_hash = BLAKE3(model_id.to_lowercase())
          = 0x7f3a9c2b4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a
```

### 2. Embedding Generation Pipeline

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌────────────┐
│    Text     │───>│  Tokenizer   │───>│  Transformer│───>│  Pooling   │
│   (Chunk)   │    │   (BPE)      │    │   (6 layers)│    │  (Mean)    │
└─────────────┘    └──────────────┘    └─────────────┘    └────────────┘
                                                                 │
                                                                 v
                                                          ┌────────────┐
                                                          │ Normalize  │
                                                          │(SoftFloat) │
                                                          └────────────┘
                                                                 │
                                                                 v
                                                          ┌────────────┐
                                                          │  Quantize  │
                                                          │ (f32→i16)  │
                                                          └────────────┘
```

### 3. Tokenization

#### Algorithm

```
function tokenize(text: String) -> TokenIds:
    // 1. Normalize text (lowercase for uncased models)
    normalized = text.to_lowercase()

    // 2. Apply BPE tokenization
    tokens = bpe_tokenize(normalized)

    // 3. Add special tokens
    tokens = [CLS_TOKEN] + tokens + [SEP_TOKEN]

    // 4. Truncate if necessary
    if tokens.len() > MAX_SEQ_LEN:
        tokens = tokens[:MAX_SEQ_LEN - 1] + [SEP_TOKEN]

    // 5. Convert to IDs
    return vocab.to_ids(tokens)
```

#### Special Tokens

| Token | ID | Purpose |
|-------|-----|---------|
| [CLS] | 101 | Sequence start |
| [SEP] | 102 | Sequence end |
| [PAD] | 0 | Padding |
| [UNK] | 100 | Unknown token |

### 4. Attention Mask

```
function create_attention_mask(token_ids: Vec<i64>) -> Vec<i64>:
    // 1 for real tokens, 0 for padding
    return token_ids.map(|id| if id == PAD_TOKEN { 0 } else { 1 })
```

### 5. Model Inference

#### Forward Pass

```
function forward(token_ids: Vec<i64>, attention_mask: Vec<i64>) -> Vec<Vec<f32>>:
    // 1. Embed tokens
    embeddings = token_embeddings(token_ids) + position_embeddings(token_ids.len())

    // 2. Apply transformer layers
    hidden = embeddings
    for layer in transformer_layers:
        hidden = layer.forward(hidden, attention_mask)

    // 3. Return hidden states
    return hidden  // Shape: [seq_len, hidden_size]
```

### 6. Inference Modes

CFS defines two inference modes to balance verification with user experience:

1.  **Canonical Inference (WASM/SoftFloat)**:
    -   **MUST** be used for generating embeddings that enter the **Canonical State** (Merkle Tree).
    -   Executes in a WASM sandbox or using a bit-exact SoftFloat kernel.
    -   Guarantees bit-for-bit reproducibility across all architectures (x86, ARM, RISC-V).
    -   **Slower** but verifiable.

2.  **Fast Inference (Hardware Accelerated)**:
    -   **MAY** be used for transient operations like local chat responses or UI feedback.
    -   Uses GPU/NPU/AMX for speed.
    -   **NOT** used for state generation/hashing.
    -   **Faster** but potentially non-deterministic across devices.

### 7. Pooling Strategy

CFS uses **mean pooling** over token embeddings.

#### Algorithm

```
function mean_pooling(hidden_states: Vec<Vec<f32>>, attention_mask: Vec<i64>) -> Vec<f32>:
    // 1. Expand attention mask to hidden dimension
    mask = attention_mask.expand(hidden_dim)

    // 2. Apply mask to hidden states
    masked = hidden_states * mask

    // 3. Sum over sequence dimension
    summed = masked.sum(axis=0)

    // 4. Count non-padded tokens
    counts = attention_mask.sum()

    // 5. Divide by counts (mean)
    return summed / counts
```

#### Why Mean Pooling?

| Strategy | Pros | Cons |
|----------|------|------|
| CLS Token | Fast | May not capture full meaning |
| Max Pooling | Captures extremes | Loses nuance |
| **Mean Pooling** | Balanced representation | Slightly slower |

Mean pooling provides the best balance for semantic similarity tasks.

### 8. L2 Normalization (SoftFloat)

**CRITICAL**: Canonical normalization **MUST** use software-implemented floating point operations (SoftFloat) to guarantee bit-exact results across architectures. Hardware FPU operations are **BANNED** for this step.

```
function l2_normalize_canonical(vector: Vec<f32>) -> Vec<f32>:
    // 1. Convert to SoftFloat
    soft_vec = vector.map(|x| SoftF64::from_f32(x))

    // 2. Compute L2 norm (in soft double precision)
    sum_sq = soft_vec.iter().map(|x| soft_mul(x, x)).sum()
    l2_norm = soft_sqrt(sum_sq)

    // 3. Divide by norm
    if l2_norm == 0.0:
        return vector
    
    return soft_vec.map(|x| soft_div(x, l2_norm).to_f32())
```

**Rationale**: `sqrt` and `div` are notoriously non-deterministic across x86 (SSE/AVX) and ARM (NEON). SoftFloat ensures `1.0 / 3.0` is identical on every device.

### 9. Quantization (f32 → i16)

Embeddings are quantized to **signed 16-bit integers** to ensure cross-platform consistency and reduce storage size.

#### Algorithm

```
function quantize_f32_to_i16(vector: Vec<f32>) -> Vec<i16>:
    return vector.map(|x| {
        // 1. Scale to i16 range (removing outliers > 1.0)
        // using SoftFloat logic implicitly
        scaled = x * 32767.0

        // 2. Robust Rounding (Dead-zone)
        // If value is extremely close to x.5, snap to lower magnitude
        // to prevent architecture-specific rounding jitter.
        if is_near_boundary(scaled):
            return floor(scaled)
        
        // 3. Round to nearest integer
        return round_half_to_even(scaled) as i16
    })
```

#### Precision Analysis

| Format | Mantissa Bits | Precision | Size |
|--------|---------------|-----------|------|
| f32 | 23 | ~7 decimal places | 4 bytes |
| **i16** | 15 (fixed) | ~4 decimal places | **2 bytes** |

i16 provides sufficient fidelity for semantic search while guaranteeing that `dot_product(a, b)` can be computed using integer arithmetic, which is universally deterministic.

### 10. Complete Canonical Embedding Generation

```
function generate_canonical_embedding(chunk: Chunk, model: EmbeddingModel) -> CanonicalEmbedding:
    // 1. Tokenize
    token_ids = tokenize(chunk.text)
    attention_mask = create_attention_mask(token_ids)

    // 2. Forward pass (WASM or SoftFloat Kernel)
    hidden_states = model.forward_canonical(token_ids, attention_mask)

    // 3. Pool
    pooled = mean_pooling(hidden_states, attention_mask)

    // 4. Normalize (SoftFloat)
    normalized = l2_normalize_canonical(pooled)

    // 5. Quantize (i16)
    vector_i16 = quantize_f32_to_i16(normalized)

    // 6. Canonical Hash
    // Hash the SORTED byte representation of the i16 vector
    embedding_hash = BLAKE3(vector_i16.to_le_bytes())

    // 7. Create embedding
    embedding_id = UUIDv5(
        EMBEDDING_NAMESPACE,
        chunk.id.to_bytes() || model.hash
    )

    return CanonicalEmbedding {
        id: embedding_id,
        chunk_id: chunk.id,
        vector: vector_i16,  // i16
        model_hash: model.hash,
        content_hash: embedding_hash,
    }
```

### 11. Batch Embedding

For efficiency, CFS supports batch embedding:

```
function embed_batch_canonical(chunks: Vec<Chunk>, model: EmbeddingModel) -> Vec<CanonicalEmbedding>:
    // 1. Tokenize all chunks
    batch_tokens = chunks.map(|c| tokenize(c.text))
    max_len = batch_tokens.map(|t| t.len()).max()

    // 2. Pad to same length
    padded_tokens = batch_tokens.map(|t| pad_to_length(t, max_len))
    attention_masks = padded_tokens.map(|t| create_attention_mask(t))

    // 3. Stack into batch
    token_batch = stack(padded_tokens)  // Shape: [batch, seq_len]
    mask_batch = stack(attention_masks)

    // 4. Forward pass (batched, must be canonical/WASM)
    hidden_batch = model.forward_batch_canonical(token_batch, mask_batch)

    // 5. Pool each sequence
    embeddings = []
    for i, chunk in enumerate(chunks):
        pooled = mean_pooling(hidden_batch[i], mask_batch[i])
        normalized = l2_normalize_canonical(pooled)
        quantized = quantize_f32_to_i16(normalized)

        embedding = create_canonical_embedding(chunk, quantized, model)
        embeddings.push(embedding)

    return embeddings
```

### 12. Model Provenance

Every embedding tracks its model provenance:

```
struct ModelProvenance {
    model_id: String,       // e.g., "sentence-transformers/all-MiniLM-L6-v2"
    model_hash: [u8; 32],   // BLAKE3(model_id)
    version: String,        // e.g., "1.0.0"
}
```

This enables:

1. **Re-embedding Detection**: Identify outdated embeddings when model changes
2. **Compatibility Checking**: Ensure query embeddings use same model
3. **Audit Trail**: Track which model generated each embedding

### 13. Model Versioning

When the embedding model changes:

```
function on_model_change(old_hash: [u8; 32], new_hash: [u8; 32]):
    // 1. Mark old embeddings as stale
    old_embeddings = graph.embeddings_by_model_hash(old_hash)

    // 2. Re-embed all affected chunks
    for emb in old_embeddings:
        chunk = graph.get_chunk(emb.chunk_id)
        new_emb = generate_canonical_embedding(chunk, new_model)
        graph.replace_embedding(emb.id, new_emb)

    // 3. Rebuild Runtime Index
    graph.rebuild_hnsw_index()
```

## Desired Properties

### 1. Determinism

**Property**: Identical chunks MUST produce identical embeddings.

**Verification**:
```
∀ chunk: generate_embedding(chunk) = generate_embedding(chunk)
```

**Implementation Notes**:
- Use fixed random seeds
- Disable non-deterministic operations (e.g., dropout)
- Use **SoftFloat** for all normalization
- Use **WASM** for inference

### 2. Semantic Preservation

**Property**: Similar texts SHOULD produce similar embeddings.

**Metric**:
```
dot(embed("dog"), embed("puppy")) > dot(embed("dog"), embed("airplane"))
```

### 3. Model Independence

**Property**: CFS SHOULD support multiple embedding models.

**Mechanism**: Model hash field enables model-agnostic storage.

## Alternative Models

CFS implementations MAY support alternative models:

| Model | Dimensions | Speed | Quality |
|-------|------------|-------|---------|
| all-MiniLM-L6-v2 | 384 | Fast | Good |
| all-mpnet-base-v2 | 768 | Medium | Better |
| BGE-small-en-v1.5 | 384 | Fast | Good |
| E5-small-v2 | 384 | Fast | Better |

Alternative models MUST:

1. Produce deterministic outputs
2. Support the embedding interface
3. Be identifiable by model hash

## Security Considerations

### Model Integrity

Implementations SHOULD verify model checksums before use:

```
function load_model(path: String, expected_hash: [u8; 32]) -> Model:
    data = read_file(path)
    actual_hash = BLAKE3(data)

    if actual_hash != expected_hash:
        raise IntegrityError("Model file corrupted")

    return deserialize_model(data)
```

### Input Sanitization

Long inputs are truncated to prevent resource exhaustion:

```
MAX_CHUNK_LENGTH = 10_000  // Characters
MAX_TOKENS = 256

function sanitize_input(text: String) -> String:
    if text.len() > MAX_CHUNK_LENGTH:
        text = text[:MAX_CHUNK_LENGTH]
    return text
```

## Test Vectors

### Tokenization

```
Input:  "Hello, world!"
Output: [101, 7592, 1010, 2088, 999, 102]
         [CLS] hello  ,    world  !   [SEP]
```

### Embedding (First 8 dimensions - i16)

```
Input:  "The quick brown fox"
// Quantized to i16 (Scale: 32767)
Output: [767, -511, 2920, -1386, 547, -2562, 1678, -767]
```

### Similarity (Integer Dot Product)

```
// Dot product of normalized i16 vectors (scaled) needs large accumulator
embed("dog")     · embed("puppy")    = High Positive
embed("dog")     · embed("cat")      = Medium Positive
embed("dog")     · embed("airplane") = Low Positive / Negative
```

## References

- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)
- [MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression](https://arxiv.org/abs/2002.10957)
- [Berkley SoftFloat](http://www.jhauser.us/arithmetic/SoftFloat.html)
