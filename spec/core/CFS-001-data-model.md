# CFS-001: Data Model & Identity

> **Spec Version**: 1.1.0-draft
> **Status**: Draft
> **Category**: Core
> **Requires**: None

## Synopsis

This specification defines the canonical data types that comprise the CFS substrate. All CFS implementations MUST support these data types with the exact semantics described herein.

## Motivation

A well-defined data model is essential for:

1. **Interoperability**: Different implementations can exchange data
2. **Determinism**: Identical representations produce identical hashes
3. **Auditability**: Clear semantics enable verification and debugging
4. **Extensibility**: Future versions can add fields without breaking compatibility

## Technical Specification

### 1. Cryptographic Primitives

CFS uses specific cryptographic algorithms to ensure determinism and security.

#### Hash Function: BLAKE3
CFS uses **BLAKE3** for all content hashing and Merkle tree construction.
- **Output**: 256 bits (32 bytes)
- **Properties**: Deterministic, parallelizable, collision-resistant (128-bit).

#### Identity Generation: UUIDv5
CFS uses **UUID Version 5** (SHA-1 based) for all entity identifiers.
- **Namespace**: Deterministic constant per entity type.
- **Input**: Content hash or unique attributes.

### 2. Document

A **Document** represents an ingested file in the substrate.

#### Schema

```
Document {
    id:              UUID        // Deterministic identifier (UUIDv5)
    path:            String      // Original filesystem path
    content_hash:    [u8; 32]    // BLAKE3 hash of file contents
    hierarchical_hash: [u8; 32]  // BLAKE3 hash of all chunk hashes
    mtime:           i64         // Unix timestamp (milliseconds)
    size_bytes:      u64         // File size in bytes
    mime_type:       String      // MIME type (e.g., "text/markdown")
}
```

#### Identity Derivation

The Document ID MUST be derived deterministically:

```
document_id = UUIDv5(
    namespace = DOCUMENT_NAMESPACE,  // Fixed UUID for documents
    name      = content_hash         // BLAKE3 hash of file contents
)
```

**Namespaces**:

| Entity | Namespace UUID |
|--------|---------------|
| Document | `6ba7b810-9dad-11d1-80b4-00c04fd430c8` |
| Chunk | `6ba7b811-9dad-11d1-80b4-00c04fd430c8` |
| Embedding | `6ba7b812-9dad-11d1-80b4-00c04fd430c8` |
| Device | `6ba7b813-9dad-11d1-80b4-00c04fd430c8` |
| StateRoot | `6ba7b814-9dad-11d1-80b4-00c04fd430c8` |

#### Hierarchical Hash

The hierarchical hash provides a Merkle commitment to all chunks belonging to this document:

```
hierarchical_hash = BLAKE3(
    sorted_chunk_hashes.join()  // Chunks sorted by sequence number
)
```

This enables verification that chunk content has not been modified.

### 2. Chunk

A **Chunk** represents a semantic unit of text extracted from a document.

#### Schema

```
Chunk {
    id:              UUID        // Deterministic identifier (UUIDv5)
    document_id:     UUID        // Parent document reference
    text:            String      // Chunk content (UTF-8)
    text_hash:       [u8; 32]    // BLAKE3 hash of text
    byte_offset:     u64         // Start position in source document
    byte_length:     u64         // Length in bytes
    sequence:        u32         // Order within document (0-indexed)
}
```

#### Identity Derivation

```
chunk_id = UUIDv5(
    namespace = CHUNK_NAMESPACE,     // Fixed UUID for chunks
    name      = text_hash            // BLAKE3 hash of chunk text
)
```

Where `CHUNK_NAMESPACE` is the fixed UUID: `6ba7b811-9dad-11d1-80b4-00c04fd430c8`.

#### Overlap Semantics

Chunks MAY overlap to preserve semantic context across boundaries:

```
Document: "The quick brown fox jumps over the lazy dog."
                                    ^^^^
                    Chunk 1 ends here ─┘└─ Chunk 2 begins here
                    (overlapping region)
```

The `byte_offset` and `byte_length` fields precisely define each chunk's extent, allowing reconstruction of overlap relationships.

### 3. Embedding

An **Embedding** represents the vector representation of a chunk.

#### Schema

```
Embedding {
    id:              UUID        // Deterministic identifier (UUIDv5)
    chunk_id:        UUID        // Source chunk reference
    vector:          [i16; N]    // Embedding vector (i16 quantization)
    model_hash:      [u8; 32]    // BLAKE3 hash of model identifier
    l2_norm:         f32         // Precomputed L2 norm for similarity
}
```

#### Identity Derivation

```
embedding_id = UUIDv5(
    namespace = EMBEDDING_NAMESPACE,
    name      = chunk_id || model_hash    // Concatenation
)
```

Where `EMBEDDING_NAMESPACE` is the fixed UUID: `6ba7b812-9dad-11d1-80b4-00c04fd430c8`.

#### Vector Format

Embeddings are stored in **i16 (signed 16-bit integer)** format:

- **Rationale**: Strict bit-level determinism across architectures (avoids FPU inconsistencies)
- **Precision**: Quantized from SoftFloat (range -32767 to 32767)
- **Dimension**: Implementation-defined (typically 384 for MiniLM-L6-v2)

#### Model Provenance

The `model_hash` field provides cryptographic binding to the embedding model:

```
model_hash = BLAKE3(model_identifier_string)
```

This ensures embeddings generated by different models are not conflated.

### 4. Edge

An **Edge** represents a relationship between entities in the substrate.

#### Schema

```
Edge {
    source_id:       UUID        // Source entity
    target_id:       UUID        // Target entity
    kind:            EdgeKind    // Relationship type
    weight:          Option<f32> // Optional relationship strength
}
```

#### Edge Kinds

```
enum EdgeKind {
    DocToChunk,        // Document contains Chunk
    ChunkToEmbedding,  // Chunk has Embedding
    ChunkToChunk,      // Semantic similarity between chunks
    Custom(String),    // User-defined relationship
}
```

#### Canonicalization

Edges are uniquely identified by the tuple `(source_id, target_id, kind)`. No duplicate edges are permitted.

### 5. StateRoot

A **StateRoot** represents a cryptographic commitment to the entire substrate state.

#### Schema

```
StateRoot {
    hash:            [u8; 32]    // BLAKE3 Merkle root
    parent_hash:     Option<[u8; 32]>  // Previous state root (chain)
    timestamp:       i64         // Unix timestamp (milliseconds)
    device_id:       UUID        // Originating device
    signature:       [u8; 64]    // Ed25519 signature
    sequence:        u64         // Monotonic sequence number
}
```

#### Merkle Root Computation

State roots are computed using a BLAKE3 Merkle Tree over all sorted entities.

1.  **Sort**: All entities are sorted by their deterministic ID.
2.  **Leaf Hash**: Each entity is hashed with its type-specific fields.
3.  **Tree Build**: Leaves are hashed in pairs (BLAKE3) up to a single root.
4.  **State Composition**: The final State Root hash is `BLAKE3(doc_root || chunk_root || emb_root || edge_root)`.

See `CFS-002` (formerly `CFS-002`, now merged) logic for precise node construction.

#### State Chain

State roots form a hash chain, enabling:

1. **Lineage Verification**: Trace state evolution over time
2. **Fork Detection**: Identify divergent state branches
3. **Rollback Points**: Return to previous verified states

### 6. CognitiveDiff

A **CognitiveDiff** represents an atomic unit of state change.

#### Schema

```
CognitiveDiff {
    // Added or updated entities
    documents:       Vec<Document>
    chunks:          Vec<Chunk>
    embeddings:      Vec<Embedding>
    edges:           Vec<Edge>

    // Removed entity IDs
    removed_documents:   Vec<UUID>
    removed_chunks:      Vec<UUID>
    removed_embeddings:  Vec<UUID>
    removed_edges:       Vec<(UUID, UUID, EdgeKind)>

    // Metadata
    prev_root:       [u8; 32]    // State root before diff
    new_root:        [u8; 32]    // State root after diff
    timestamp:       i64         // Creation timestamp
    device_id:       UUID        // Originating device
    sequence:        u64         // Diff sequence number
}
```

#### Serialization

CognitiveDiffs are serialized using:

1. **CBOR**: Canonical Binary Object Representation (RFC 8949)
2. **zstd**: Compression for network transmission

## Desired Properties

### 1. Determinism

**Property**: Given identical inputs, any conformant implementation MUST produce identical data structures.

**Verification**:
```
∀ input: hash(process(input)) = hash(process(input))
```

### 2. Referential Integrity

**Property**: All foreign key references MUST point to existing entities.

**Constraints**:
- `Chunk.document_id` MUST reference an existing `Document`
- `Embedding.chunk_id` MUST reference an existing `Chunk`
- `Edge.source_id` and `Edge.target_id` MUST reference existing entities

### 3. Immutability of Identity

**Property**: Entity IDs MUST NOT change after creation.

**Rationale**: Since IDs are derived from content hashes, changing content creates a new entity rather than modifying the existing one.

### 4. Cascade Semantics

**Property**: Deleting an entity MUST delete all dependent entities.

**Cascade Rules**:
- Deleting a `Document` deletes all its `Chunk`s
- Deleting a `Chunk` deletes all its `Embedding`s
- Deleting an entity removes all `Edge`s referencing it

## Backwards Compatibility

New fields MAY be added to data types in minor versions. Implementations MUST:

1. Ignore unknown fields when deserializing
2. Preserve unknown fields when re-serializing (round-trip safety)
3. Provide sensible defaults for missing optional fields

## Test Vectors

### Document ID Generation

```
Input:
    content_hash = 0xaf1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262

Expected:
    document_id = UUID("a7f3b2c1-4d5e-5f6a-8b9c-0d1e2f3a4b5c")
```

### Chunk ID Generation

```
Input:
    text = "The quick brown fox jumps over the lazy dog."
    text_hash = BLAKE3(text)

Expected:
    chunk_id = UUID("b8c4d3e2-5f6a-5b7c-9d0e-1f2a3b4c5d6e")
```

## References

- [RFC 4122: UUID URN Namespace](https://tools.ietf.org/html/rfc4122)
- [BLAKE3 Specification](https://github.com/BLAKE3-team/BLAKE3-specs)
- [RFC 8949: CBOR](https://tools.ietf.org/html/rfc8949)
