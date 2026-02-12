# CFS-011: Indexing Protocol

> **Spec Version**: 1.0.0
> **Author**: Nadeem Bhati
> **Category**: Protocol
> **Requires**: CFS-001, CFS-002, CFS-003

## Synopsis

This specification defines the document ingestion and indexing protocol for CFS. It covers file parsing, chunking strategies, and index construction.

## Motivation

Effective indexing is critical for:

1. **Retrieval Quality**: Well-chunked content produces better search results
2. **Determinism**: Same content must produce same index entries
3. **Efficiency**: Incremental updates minimize reprocessing
4. **Semantic Preservation**: Chunks maintain meaningful context

## Technical Specification

### 1. Ingestion Pipeline

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌────────────┐
│    File     │───>│   Parser     │───>│  Chunker    │───>│  Embedder  │
│  (Raw)      │    │  (Extract)   │    │  (Split)    │    │  (Vector)  │
└─────────────┘    └──────────────┘    └─────────────┘    └────────────┘
                                                                 │
                                                                 v
                                                          ┌────────────┐
                                                          │  Indexer   │
                                                          │ 1. State   │
                                                          │(Sorted List)
                                                          │            │
                                                          │ 2. Cache   │
                                                          │  (HNSW)    │
                                                          └────────────┘
```

### 2. File Detection

CFS detects file types for appropriate parsing:

```
function detect_file_type(path: String) -> MimeType:
    // 1. Check file extension
    extension = path.extension().to_lowercase()

    // 2. Map to MIME type
    match extension:
        "md" | "markdown" => "text/markdown"
        "txt"             => "text/plain"
        "pdf"             => "application/pdf"
        "html" | "htm"    => "text/html"
        "json"            => "application/json"
        "rs"              => "text/x-rust"
        "py"              => "text/x-python"
        "js"              => "text/javascript"
        "ts"              => "text/typescript"
        _                 => "application/octet-stream"
```

### 3. Parsing

#### 3.1 Text Parser

```
function parse_text(content: bytes) -> String:
    // 1. Detect encoding
    encoding = detect_encoding(content)  // UTF-8, UTF-16, etc.

    // 2. Decode to string
    text = decode(content, encoding)

    // 3. Canonicalize
    return canonicalize_text(text)
```

#### 3.2 Markdown Parser

```
function parse_markdown(content: bytes) -> ParsedDocument:
    text = parse_text(content)

    // 1. Extract frontmatter (if present)
    if text.starts_with("---"):
        (frontmatter, body) = extract_frontmatter(text)
    else:
        frontmatter = None
        body = text

    // 2. Parse markdown structure
    ast = markdown_parse(body)

    // 3. Extract sections
    sections = []
    for node in ast:
        match node:
            Heading(level, text) =>
                sections.push(Section { level, title: text, content: "" })
            Paragraph(text) =>
                sections.last().content += text + "\n"
            CodeBlock(lang, code) =>
                sections.last().content += f"```{lang}\n{code}\n```\n"
            List(items) =>
                for item in items:
                    sections.last().content += f"- {item}\n"

    return ParsedDocument { frontmatter, sections }
```

#### 3.3 PDF Parser

```
function parse_pdf(content: bytes) -> String:
    // 1. Load PDF document
    pdf = pdfium_load(content)

    // 2. Extract text from each page
    pages = []
    for page_num in 0..pdf.page_count():
        page = pdf.get_page(page_num)
        text = page.extract_text()
        pages.push(text)

    // 3. Join pages with form feed separator
    combined = pages.join("\f")  // Form feed = page break

    // 4. Canonicalize
    return canonicalize_text(combined)
```

#### 3.4 Code Parser

```
function parse_code(content: bytes, language: String) -> String:
    text = parse_text(content)

    // Preserve code structure (no canonicalization that affects semantics)
    return text
```

### 4. Chunking

#### 4.1 Chunking Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| target_chunk_size | 256 tokens | Target size for each chunk |
| max_chunk_size | 512 tokens | Maximum allowed chunk size |
| overlap | 32 tokens | Overlap between adjacent chunks |
| separator_priority | ["\n\n", "\n", ". ", " "] | Split preference order |

#### 4.2 Semantic Chunking Algorithm

```
function chunk_text(text: String, params: ChunkParams) -> Vec<ChunkData>:
    chunks = []
    current_offset = 0
    sequence = 0

    while current_offset < text.len():
        // 1. Find chunk boundary
        chunk_end = find_chunk_boundary(
            text,
            current_offset,
            params.target_chunk_size,
            params.max_chunk_size,
            params.separator_priority
        )

        // 2. Extract chunk text
        chunk_text = text[current_offset:chunk_end]

        // 3. Create chunk
        chunks.push(ChunkData {
            text: chunk_text,
            byte_offset: current_offset,
            byte_length: chunk_end - current_offset,
            sequence: sequence,
        })

        // 4. Move to next position (with overlap)
        next_offset = chunk_end - params.overlap
        if next_offset <= current_offset:
            next_offset = chunk_end  // Avoid infinite loop

        current_offset = next_offset
        sequence += 1

    return chunks
```

#### 4.3 Boundary Detection

```
function find_chunk_boundary(
    text: String,
    start: usize,
    target_size: usize,
    max_size: usize,
    separators: Vec<String>
) -> usize:
    end = min(start + max_size, text.len())
    target = start + target_size

    // Try each separator in priority order
    for separator in separators:
        // Look for separator near target
        search_start = max(target - 64, start)
        search_end = min(target + 64, end)
        search_region = text[search_start:search_end]

        if let Some(pos) = search_region.rfind(separator):
            return search_start + pos + separator.len()

    // No separator found, hard break at max size
    return min(start + max_size, text.len())
```

#### 4.4 Section-Aware Chunking (Markdown)

```
function chunk_markdown(doc: ParsedDocument, params: ChunkParams) -> Vec<ChunkData>:
    chunks = []
    global_offset = 0
    sequence = 0

    for section in doc.sections:
        // Prepend section header for context
        header_prefix = f"# {section.title}\n\n"

        // Chunk section content
        section_chunks = chunk_text(section.content, params)

        for chunk in section_chunks:
            // Add header to first chunk of each section
            if chunk.sequence == 0:
                chunk.text = header_prefix + chunk.text

            chunk.byte_offset += global_offset
            chunk.sequence = sequence
            chunks.push(chunk)
            sequence += 1

        global_offset += section.content.len()

    return chunks
```

### 5. Document Creation

```
function create_document(path: String, content: bytes) -> Document:
    // 1. Compute content hash
    content_hash = BLAKE3(content)

    // 2. Generate document ID
    doc_id = UUIDv5(DOCUMENT_NAMESPACE, content_hash)

    // 3. Detect MIME type
    mime_type = detect_file_type(path)

    // 4. Get file metadata
    stat = fs::metadata(path)
    mtime = stat.modified().timestamp_millis()
    size_bytes = stat.size()

    // 5. Create document (hierarchical_hash computed after chunking)
    return Document {
        id: doc_id,
        path: path,
        content_hash: content_hash,
        hierarchical_hash: [0; 32],  // Placeholder
        mtime: mtime,
        size_bytes: size_bytes,
        mime_type: mime_type,
    }
```

### 6. Chunk Creation

```
function create_chunks(doc: Document, chunk_data: Vec<ChunkData>) -> Vec<Chunk>:
    chunks = []

    for data in chunk_data:
        // 1. Compute text hash
        text_hash = BLAKE3(data.text.as_bytes())

        // 2. Generate chunk ID
        chunk_id = UUIDv5(CHUNK_NAMESPACE, text_hash)

        // 3. Create chunk
        chunks.push(Chunk {
            id: chunk_id,
            document_id: doc.id,
            text: data.text,
            text_hash: text_hash,
            byte_offset: data.byte_offset,
            byte_length: data.byte_length,
            sequence: data.sequence,
        })

    return chunks
```

### 7. Hierarchical Hash Computation

```
function compute_hierarchical_hash(chunks: Vec<Chunk>) -> [u8; 32]:
    // Sort chunks by sequence
    sorted = chunks.sort_by(|c| c.sequence)

    // Collect text hashes
    hashes = sorted.map(|c| c.text_hash)

    // Compute Merkle root
    return compute_merkle_root(hashes)
```

### 8. Complete Ingestion Workflow

```
function ingest_file(path: String, graph: GraphStore) -> Result<IngestResult>:
    // 1. Read file
    content = fs::read(path)?

    // 2. Check if already indexed
    content_hash = BLAKE3(content)
    if let Some(existing) = graph.get_document_by_content_hash(content_hash):
        return Ok(IngestResult::AlreadyExists(existing.id))

    // 3. Create document
    doc = create_document(path, content)

    // 4. Parse content
    mime = doc.mime_type
    parsed = match mime:
        "text/markdown" => parse_markdown(content)
        "application/pdf" => parse_pdf(content)
        _ => parse_text(content)

    // 5. Chunk content
    params = ChunkParams::default()
    chunk_data = match mime:
        "text/markdown" => chunk_markdown(parsed, params)
        _ => chunk_text(parsed.text, params)

    // 6. Create chunks
    chunks = create_chunks(doc, chunk_data)

    // 7. Compute hierarchical hash
    doc.hierarchical_hash = compute_hierarchical_hash(chunks)

    // 8. Generate embeddings
    embeddings = embed_batch(chunks, model)

    // 9. Begin transaction (Canonical State Update)
    tx = graph.begin_transaction()

    try:
        // 10. Insert document
        graph.insert_document(doc)

        // 11. Insert chunks
        graph.insert_chunks(chunks)

        // 12. Insert embeddings (Canonical State: Sorted List in DB)
        graph.insert_embeddings(embeddings)

        // 13. Create edges
        for (chunk, emb) in zip(chunks, embeddings):
            graph.insert_edge(Edge::DocToChunk(doc.id, chunk.id))
            graph.insert_edge(Edge::ChunkToEmbedding(chunk.id, emb.id))

        // 14. Commit to Canonical State
        // This makes the data durable and verifies the Merkle root
        graph.commit_transaction(tx)

        // 15. Update Runtime Index (Transient)
        // This updates the local HNSW cache for fast retrieval.
        // It is NOT part of the atomic state commit.
        graph.update_runtime_index(embeddings)

        return Ok(IngestResult::Ingested(doc.id, chunks.len()))

    except error:
        graph.rollback_transaction(tx)
        raise error
```

### 9. Incremental Updates

When a file is modified:

```
function update_file(path: String, graph: GraphStore) -> Result<UpdateResult>:
    // 1. Read new content
    new_content = fs::read(path)?
    new_hash = BLAKE3(new_content)

    // 2. Get existing document
    existing = graph.get_document_by_path(path)?

    if existing.content_hash == new_hash:
        return Ok(UpdateResult::Unchanged)

    // 3. Delete old document (cascades)
    graph.delete_document(existing.id)

    // 4. Ingest as new
    return ingest_file(path, graph)
```

### 10. File Watching

CFS supports live file monitoring:

```
function watch_directory(path: String, graph: GraphStore):
    watcher = FileWatcher::new(path)

    for event in watcher.events():
        match event:
            Created(path) =>
                ingest_file(path, graph)

            Modified(path) =>
                update_file(path, graph)

            Deleted(path) =>
                if let Some(doc) = graph.get_document_by_path(path):
                    graph.delete_document(doc.id)

            Renamed(old_path, new_path) =>
                if let Some(doc) = graph.get_document_by_path(old_path):
                    doc.path = new_path
                    graph.update_document(doc)
```

### 11. Runtime Index Management

The **Runtime Index** (HNSW) is a transient performance cache. It is **NOT** part of the canonical state.

#### Properties

| Property | Canonical State | Runtime Index |
|----------|-----------------|---------------|
| Structure | Sorted List (B-Tree) | Graph (HNSW) |
| Durability | Persistent (Event Log) | Ephemeral (Cache) |
| Determinism | Strict (Bit-Exact) | Loose (Graph construction varies) |
| Usage | Verification & Sync | Fast Retrieval |

#### Reconstruction

The Runtime Index can be deleted and rebuilt from the Canonical State at any time:

```
function rebuild_runtime_index(graph: GraphStore):
    // 1. Clear existing HNSW
    graph.hnsw.clear()

    // 2. Load all canonical embeddings (sorted)
    embeddings = graph.get_all_embeddings_sorted()

    // 3. Batch insert into HNSW
    // Note: Insertion order affects graph structure but NOT search correctness
    graph.hnsw.build(embeddings)
```

## Desired Properties

### 1. Determinism

**Property**: Identical files MUST produce identical documents, chunks, and embeddings.

**Verification**:
```
∀ content: ingest(content).state_root = ingest(content).state_root
```

### 2. Chunk Coherence

**Property**: Chunks SHOULD preserve semantic coherence.

**Heuristics**:
- Prefer breaking at paragraph boundaries
- Keep code blocks intact when possible
- Maintain header context

### 3. Overlap Continuity

**Property**: Overlapping regions MUST be identical in adjacent chunks.

**Verification**:
```
∀ i: chunk[i].text[end-overlap:end] = chunk[i+1].text[0:overlap]
```

### 4. Complete Coverage

**Property**: The union of all chunks MUST cover the entire document.

**Verification**:
```
∀ doc: union(chunks).text = doc.text
```

## Chunking Strategies

### Strategy Comparison

| Strategy | Pros | Cons |
|----------|------|------|
| Fixed Size | Simple, predictable | Breaks mid-sentence |
| Sentence | Natural boundaries | Varying sizes |
| Paragraph | Semantic units | Can be too large |
| Section | Document structure | Markdown-specific |
| Semantic | Best retrieval | Complex, slower |

CFS default uses **hybrid semantic chunking** with separator priority.

### Custom Chunking

Implementations MAY provide custom chunking strategies:

```
interface Chunker:
    fn chunk(text: String, params: ChunkParams) -> Vec<ChunkData>

// Example: Code-aware chunker
struct CodeChunker:
    fn chunk(code: String, params: ChunkParams) -> Vec<ChunkData>:
        // Chunk at function boundaries
        functions = parse_functions(code)
        return functions.map(|f| ChunkData {
            text: f.source,
            byte_offset: f.start,
            byte_length: f.end - f.start,
            sequence: f.index,
        })
```

## Performance Optimization

### Batch Processing

```
function ingest_batch(paths: Vec<String>, graph: GraphStore) -> Vec<Result>:
    // 1. Parse all files in parallel
    parsed = paths.par_map(|p| parse_file(p))

    // 2. Chunk all documents in parallel
    chunked = parsed.par_map(|p| chunk_document(p))

    // 3. Batch embed all chunks
    all_chunks = chunked.flatten()
    embeddings = embed_batch(all_chunks, model)

    // 4. Insert all in single transaction
    tx = graph.begin_transaction()
    // ... insert all entities
    graph.commit_transaction(tx)
```

### Memory Management

For large files:

```
function ingest_large_file(path: String, graph: GraphStore):
    // Stream file in chunks
    file = fs::open(path)
    buffer_size = 1_000_000  // 1MB

    while let Some(buffer) = file.read(buffer_size):
        // Process incrementally
        chunks = chunk_text(buffer, params)
        embeddings = embed_batch(chunks, model)
        graph.insert_batch(chunks, embeddings)
```

## Test Vectors

### Chunking Test

```
Input:
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    params = ChunkParams { target_size: 50, overlap: 10 }

Output:
    chunks = [
        "First paragraph.\n\n",
        "\n\nSecond paragraph.\n\n",
        "\n\nThird paragraph."
    ]
```

### Hierarchical Hash Test

```
Input:
    chunks = [
        Chunk { text_hash: 0x1111...1111 },
        Chunk { text_hash: 0x2222...2222 },
    ]

Output:
    hierarchical_hash = BLAKE3(0x1111...1111 || 0x2222...2222)
```

## References

- [LangChain Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
- [Semantic Chunking Strategies](https://www.pinecone.io/learn/chunking-strategies/)
- [PDF.js Text Extraction](https://mozilla.github.io/pdf.js/)
