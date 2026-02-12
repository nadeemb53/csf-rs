# Cognitive Filesystem (CFS)

![CFS macOS and iOS Apps](images/llm-cfs.png)

Cognitive Filesystem (CFS) is a **local-first, deterministic semantic storage and retrieval system** written in rust, designed to index personal documents, keep them continuously up to date, and make them searchable across devices with cryptographic verification.

It is a **semantic substrate** intended to be stable, inspectable, and correct before any LLM-based intelligence is layered on top.

CFS currently provides a complete end-to-end pipeline for:

* Local document ingestion
* Incremental semantic indexing
* Hybrid (lexical + vector) retrieval
* Deterministic context assembly
* Encrypted, verifiable synchronization
* 100% Local Inference on desktop and mobile
* Desktop and mobile parity

No cloud inference or external AI APIs are required.

---

## Core Principles

* **Local-first**
  All ingestion, indexing, and retrieval run locally.

* **Deterministic**
  Identical inputs produce identical state roots, chunk IDs, and retrieval results. This is enforced via **SoftFloat** arithmetic and **Canonical Inference**.

* **Mutable but stable**
  File changes trigger incremental updates without reindexing unrelated content.

* **Inspectable**
  Internal state (chunks, hashes, state roots) is visible and debuggable.

---

## Architecture & Specifications

CFS is defined by a rigorous set of specifications that ensure interoperability and determinism.

### Core Layer
*   **[CFS-001: Data Model](./spec/core/CFS-001-data-model.md)** — Canonical data types (Document, Chunk, Embedding), UUIDv5 identity generation, and Merkle tree construction.
*   **[CFS-002: Storage Engine](./spec/core/CFS-002-storage-engine.md)** — Hybrid persistence using SQLite for metadata and HNSW for transient vector indices.
*   **[CFS-003: Determinism](./spec/core/CFS-003-determinism.md)** — Mathematical rules for SoftFloat arithmetic and Canonical Inference to guarantee bit-exact synchronization.

### Protocol Layer
*   **[CFS-010: Embedding](./spec/protocol/CFS-010-embedding-protocol.md)** — Deterministic vector generation using quantization and provenance tracking.
*   **[CFS-011: Indexing](./spec/protocol/CFS-011-indexing-protocol.md)** — Document ingestion, chunking strategies, and index maintenance.
*   **[CFS-012: Retrieval](./spec/protocol/CFS-012-retrieval-protocol.md)** — Hybrid search (Semantic + Lexical), RRF fusion, and Integer Dot Product scoring.
*   **[CFS-013: Synchronization](./spec/protocol/CFS-013-sync-protocol.md)** — Encrypted, blind synchronization using Merkle diffs and signatures.

### Application Layer
*   **[CFS-020: Intelligence Interface](./spec/application/CFS-020-intelligence-interface.md)** — strict read-only contract for LLM integration.
*   **[CFS-021: Context Assembly](./spec/application/CFS-021-context-assembly.md)** — Deterministic context window construction for RAG.

---

## Applications

### macOS App

* Folder selection and live ingestion
* State root and graph statistics display
* Hybrid query interface
* Chunk browser (debug surface)
* Manual sync controls

### iOS App

* Read-only local graph (V0)
* Encrypted state pull with Merkle-root verification
* Hybrid query support (Semantic + Lexical)
* On-device inference via `llama.cpp`
* "End-to-End Private" badge confirms local work
* State Verification view explains deterministic state roots
* Verified latency metrics for substrate-to-intelligence synthesis

Both apps are intentionally minimal and expose internal state rather than hiding it.

---

## Repository Structure

### Core Crates (`/crates`)
* `cfs-core` — Canonical data models, hashing, cryptographic primitives
* `cfs-parser` — Document parsing and chunking (PDF, Markdown, Text)
* `cfs-embeddings` — Local embedding generation (CPU-only)
* `cfs-graph` — SQLite + HNSW hybrid storage engine
* `cfs-query` — Hybrid retrieval, RRF fusion, and context assembly
* `cfs-inference-mobile` — Local LLM inference engine (`llama.cpp` + GGUF)
* `cfs-sync` — Merkle tree diffing, encryption, and state convergence
* `cfs-relay-client` — HTTP client for encrypted blob synchronization
* `cfs-desktop` — Desktop-specific ingestion and watcher logic
* `cfs-mobile` — C FFI for iOS/Android integration
* `cfs-desktop-cli` — Command-line interface for graph inspection
* `cfs-tests` — End-to-end and cross-platform validation tests

### Applications (`/apps`)
* `apps/macos` — macOS Tauri UI wrapper
* `apps/ios` — iOS SwiftUI application

### Relay Server (`/relay`)
* `relay/cfs-relay-server` — Blind Axum-based encrypted blob storage

### Infrastructure & Tools
* `scripts/` — Build and deployment scripts (e.g., iOS cross-compilation)
* `test_corpus/` — Curated dataset for system validation and RAG testing



