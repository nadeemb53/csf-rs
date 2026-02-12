//! CFS Graph - Knowledge graph and vector store
//!
//! Combines SQLite for structured data with HNSW for vector search.

use cfs_core::{Chunk, CfsError, Document, Edge, EdgeKind, Embedding, Result};
use hnsw_rs::prelude::*;
use rusqlite::{params, Connection, OptionalExtension};
use std::sync::{Arc, RwLock};
use tracing::info;
use uuid::Uuid;

/// Distance metric for HNSW - cosine distance
#[derive(Clone)]
struct CosineDistance;

impl Distance<f32> for CosineDistance {
    fn eval(&self, a: &[f32], b: &[f32]) -> f32 {
        // Returns 1 - cosine_similarity (so lower is more similar)
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            return 1.0;
        }
        (1.0 - (dot / (norm_a * norm_b))).max(0.0)
    }
}

/// HNSW index wrapper that owns its data
struct HnswIndex {
    /// The index itself (stores references to data)
    hnsw: Hnsw<'static, f32, CosineDistance>,
    /// Stored vectors (we keep ownership here)
    vectors: Vec<Vec<f32>>,
    /// Mapping from vector index to embedding UUID
    id_map: std::collections::HashMap<usize, Uuid>,
}

impl HnswIndex {
    fn new() -> Self {
        // Create HNSW index with reasonable defaults
        // max_nb_connection = 16 (M parameter), max_elements = 100000, max_layer = 16, ef_construction = 200
        let hnsw = Hnsw::new(16, 100000, 16, 200, CosineDistance);
        Self {
            hnsw,
            vectors: Vec::new(),
            id_map: std::collections::HashMap::new(),
        }
    }

    fn insert(&mut self, emb_id: Uuid, vector: Vec<f32>) {
        let idx = self.vectors.len();
        self.vectors.push(vector);
        self.id_map.insert(idx, emb_id);
        
        // Get a reference to the stored vector (now 'static since we own it)
        // Note: This is safe because we never remove from vectors
        let vec_ref: &'static [f32] = unsafe {
            std::slice::from_raw_parts(
                self.vectors[idx].as_ptr(),
                self.vectors[idx].len(),
            )
        };
        self.hnsw.insert((vec_ref, idx));
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<(Uuid, f32)> {
        let results = self.hnsw.search(query, k, 32);
        results
            .iter()
            .filter_map(|neighbor| {
                self.id_map.get(&neighbor.d_id).map(|id| {
                    // Convert distance to similarity (1 - cosine_distance)
                    (*id, 1.0 - neighbor.distance)
                })
            })
            .collect()
    }
}

/// Combined graph and vector store
pub struct GraphStore {
    /// SQLite connection for structured data
    db: Connection,
    /// HNSW index for vector search
    hnsw: Arc<RwLock<HnswIndex>>,
    /// Vector dimensionality
    _dim: usize,
}

impl GraphStore {
    /// Open or create a graph store at the given path
    pub fn open(db_path: &str) -> Result<Self> {
        let db = if db_path == ":memory:" {
            Connection::open_in_memory()
        } else {
            Connection::open(db_path)
        }
        .map_err(|e| CfsError::Database(e.to_string()))?;

        Self::init_schema(&db)?;
        
        let store = Self {
            db,
            hnsw: Arc::new(RwLock::new(HnswIndex::new())),
            _dim: 384, // Default for gte-small
        };

        store.load_hnsw_index()?;
        
        info!("Opened graph store at {} and loaded index", db_path);

        Ok(store)
    }

    /// Open an in-memory graph store (for testing)
    pub fn in_memory() -> Result<Self> {
        Self::open(":memory:")
    }

    /// Initialize the database schema
    fn init_schema(db: &Connection) -> Result<()> {
        db.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS documents (
                id BLOB PRIMARY KEY,
                path TEXT NOT NULL,
                hash BLOB NOT NULL,
                mtime INTEGER NOT NULL,
                size INTEGER NOT NULL,
                mime_type TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id BLOB PRIMARY KEY,
                doc_id BLOB NOT NULL,
                text TEXT NOT NULL,
                offset INTEGER NOT NULL,
                len INTEGER NOT NULL,
                seq INTEGER NOT NULL,
                text_hash BLOB NOT NULL,
                FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS embeddings (
                id BLOB PRIMARY KEY,
                chunk_id BLOB NOT NULL,
                vector BLOB NOT NULL,
                model_hash BLOB NOT NULL,
                dim INTEGER NOT NULL,
                norm REAL NOT NULL,
                FOREIGN KEY (chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS edges (
                source BLOB NOT NULL,
                target BLOB NOT NULL,
                kind INTEGER NOT NULL,
                weight INTEGER,
                PRIMARY KEY (source, target, kind)
            );

            CREATE TABLE IF NOT EXISTS state_roots (
                hash BLOB PRIMARY KEY,
                parent BLOB,
                timestamp INTEGER NOT NULL,
                device_id BLOB NOT NULL,
                signature BLOB NOT NULL,
                seq INTEGER NOT NULL DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
            CREATE INDEX IF NOT EXISTS idx_embeddings_chunk_id ON embeddings(chunk_id);
            CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source);
            CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target);

            -- FTS5 Virtual Table for lexical search
            CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(
                id,
                text,
                content='chunks',
                content_rowid='rowid'
            );

            -- Triggers to keep FTS in sync with chunks table
            CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                INSERT INTO fts_chunks(rowid, id, text) VALUES (new.rowid, new.id, new.text);
            END;
            CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                INSERT INTO fts_chunks(fts_chunks, rowid, id, text) VALUES('delete', old.rowid, old.id, old.text);
            END;
            CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
                INSERT INTO fts_chunks(fts_chunks, rowid, id, text) VALUES('delete', old.rowid, old.id, old.text);
                INSERT INTO fts_chunks(rowid, id, text) VALUES (new.rowid, new.id, new.text);
            END;
            "#,
        )
        .map_err(|e| CfsError::Database(e.to_string()))?;

        Ok(())
    }

    // ========== Document operations ==========

    /// Insert a document
    pub fn insert_document(&mut self, doc: &Document) -> Result<()> {
        self.db
            .execute(
                "INSERT OR REPLACE INTO documents (id, path, hash, mtime, size, mime_type) 
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![
                    doc.id.as_bytes().as_slice(),
                    doc.path.to_string_lossy().as_ref(),
                    doc.hash.as_slice(),
                    doc.mtime,
                    doc.size as i64,
                    &doc.mime_type,
                ],
            )
            .map_err(|e| CfsError::Database(e.to_string()))?;

        Ok(())
    }

    /// Get a document by ID
    pub fn get_document(&self, id: Uuid) -> Result<Option<Document>> {
        self.db
            .query_row(
                "SELECT id, path, hash, mtime, size, mime_type FROM documents WHERE id = ?1",
                params![id.as_bytes().as_slice()],
                |row| {
                    let id_bytes: Vec<u8> = row.get(0)?;
                    let path_str: String = row.get(1)?;
                    let hash_bytes: Vec<u8> = row.get(2)?;
                    let mtime: i64 = row.get(3)?;
                    let size: i64 = row.get(4)?;
                    let mime_type: String = row.get(5)?;

                    let id = Uuid::from_slice(&id_bytes).expect("invalid uuid");
                    let mut hash = [0u8; 32];
                    hash.copy_from_slice(&hash_bytes);

                    Ok(Document {
                        id,
                        path: std::path::PathBuf::from(path_str),
                        hash,
                        mtime,
                        size: size as u64,
                        mime_type,
                    })
                },
            )
            .optional()
            .map_err(|e| CfsError::Database(e.to_string()))
    }

    /// Get a document by path
    pub fn get_document_by_path(&self, path: &std::path::Path) -> Result<Option<Document>> {
        self.db
            .query_row(
                "SELECT id, path, hash, mtime, size, mime_type FROM documents WHERE path = ?1",
                params![path.to_string_lossy().as_ref()],
                |row| {
                    let id_bytes: Vec<u8> = row.get(0)?;
                    let path_str: String = row.get(1)?;
                    let hash_bytes: Vec<u8> = row.get(2)?;
                    let mtime: i64 = row.get(3)?;
                    let size: i64 = row.get(4)?;
                    let mime_type: String = row.get(5)?;

                    let id = Uuid::from_slice(&id_bytes).expect("invalid uuid");
                    let mut hash = [0u8; 32];
                    hash.copy_from_slice(&hash_bytes);

                    Ok(Document {
                        id,
                        path: std::path::PathBuf::from(path_str),
                        hash,
                        mtime,
                        size: size as u64,
                        mime_type,
                    })
                },
            )
            .optional()
            .map_err(|e| CfsError::Database(e.to_string()))
    }

    /// Delete a document and its associated chunks/embeddings
    pub fn delete_document(&mut self, id: Uuid) -> Result<()> {
        // Cascade delete (chunks and embeddings deleted via manual cleanup)
        self.db
            .execute(
                "DELETE FROM embeddings WHERE chunk_id IN (SELECT id FROM chunks WHERE doc_id = ?1)",
                params![id.as_bytes().as_slice()],
            )
            .map_err(|e| CfsError::Database(e.to_string()))?;

        self.db
            .execute(
                "DELETE FROM chunks WHERE doc_id = ?1",
                params![id.as_bytes().as_slice()],
            )
            .map_err(|e| CfsError::Database(e.to_string()))?;

        self.db
            .execute(
                "DELETE FROM documents WHERE id = ?1",
                params![id.as_bytes().as_slice()],
            )
            .map_err(|e| CfsError::Database(e.to_string()))?;

        Ok(())
    }

    /// Get all documents
    pub fn get_all_documents(&self) -> Result<Vec<Document>> {
        let mut stmt = self
            .db
            .prepare("SELECT id, path, hash, mtime, size, mime_type FROM documents")
            .map_err(|e| CfsError::Database(e.to_string()))?;

        let docs = stmt
            .query_map([], |row| {
                let id_bytes: Vec<u8> = row.get(0)?;
                let path_str: String = row.get(1)?;
                let hash_bytes: Vec<u8> = row.get(2)?;
                let mtime: i64 = row.get(3)?;
                let size: i64 = row.get(4)?;
                let mime_type: String = row.get(5)?;

                let id = Uuid::from_slice(&id_bytes).expect("invalid uuid");
                let mut hash = [0u8; 32];
                hash.copy_from_slice(&hash_bytes);

                Ok(Document {
                    id,
                    path: std::path::PathBuf::from(path_str),
                    hash,
                    mtime,
                    size: size as u64,
                    mime_type,
                })
            })
            .map_err(|e| CfsError::Database(e.to_string()))?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| CfsError::Database(e.to_string()))?;

        Ok(docs)
    }

    // ========== Chunk operations ==========

    /// Insert a chunk
    pub fn insert_chunk(&mut self, chunk: &Chunk) -> Result<()> {
        self.db
            .execute(
                "INSERT OR REPLACE INTO chunks (id, doc_id, text, offset, len, seq, text_hash) 
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                params![
                    chunk.id.as_bytes().as_slice(),
                    chunk.doc_id.as_bytes().as_slice(),
                    &chunk.text,
                    chunk.offset,
                    chunk.len,
                    chunk.seq,
                    chunk.text_hash.as_slice(),
                ],
            )
            .map_err(|e| CfsError::Database(e.to_string()))?;

        Ok(())
    }

    /// Get chunks for a document
    pub fn get_chunks_for_doc(&self, doc_id: Uuid) -> Result<Vec<Chunk>> {
        let mut stmt = self
            .db
            .prepare(
                "SELECT id, doc_id, text, offset, len, seq, text_hash 
                 FROM chunks WHERE doc_id = ?1 ORDER BY seq",
            )
            .map_err(|e| CfsError::Database(e.to_string()))?;

        let chunks = stmt
            .query_map(params![doc_id.as_bytes().as_slice()], |row| {
                let id_bytes: Vec<u8> = row.get(0)?;
                let doc_id_bytes: Vec<u8> = row.get(1)?;
                let text: String = row.get(2)?;
                let offset: u32 = row.get(3)?;
                let len: u32 = row.get(4)?;
                let seq: u32 = row.get(5)?;
                let text_hash_bytes: Vec<u8> = row.get(6)?;

                let id = Uuid::from_slice(&id_bytes).expect("invalid uuid");
                let doc_id = Uuid::from_slice(&doc_id_bytes).expect("invalid uuid");
                let mut text_hash = [0u8; 32];
                text_hash.copy_from_slice(&text_hash_bytes);

                Ok(Chunk {
                    id,
                    doc_id,
                    text,
                    offset,
                    len,
                    seq,
                    text_hash,
                })
            })
            .map_err(|e| CfsError::Database(e.to_string()))?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| CfsError::Database(e.to_string()))?;

        Ok(chunks)
    }

    /// Get a chunk by ID
    pub fn get_chunk(&self, id: Uuid) -> Result<Option<Chunk>> {
        self.db
            .query_row(
                "SELECT id, doc_id, text, offset, len, seq, text_hash FROM chunks WHERE id = ?1",
                params![id.as_bytes().as_slice()],
                |row| {
                    let id_bytes: Vec<u8> = row.get(0)?;
                    let doc_id_bytes: Vec<u8> = row.get(1)?;
                    let text: String = row.get(2)?;
                    let offset: u32 = row.get(3)?;
                    let len: u32 = row.get(4)?;
                    let seq: u32 = row.get(5)?;
                    let text_hash_bytes: Vec<u8> = row.get(6)?;

                    let id = Uuid::from_slice(&id_bytes).expect("invalid uuid");
                    let doc_id = Uuid::from_slice(&doc_id_bytes).expect("invalid uuid");
                    let mut text_hash = [0u8; 32];
                    text_hash.copy_from_slice(&text_hash_bytes);

                    Ok(Chunk {
                        id,
                        doc_id,
                        text,
                        offset,
                        len,
                        seq,
                        text_hash,
                    })
                },
            )
            .optional()
            .map_err(|e| CfsError::Database(e.to_string()))
    }

    // ========== Embedding operations ==========

    /// Insert an embedding and add it to the HNSW index
    pub fn insert_embedding(&mut self, emb: &Embedding) -> Result<()> {
        // Store in SQLite
        let vector_bytes: Vec<u8> = emb
            .vector
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        self.db
            .execute(
                "INSERT OR REPLACE INTO embeddings (id, chunk_id, vector, model_hash, dim, norm) 
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![
                    emb.id.as_bytes().as_slice(),
                    emb.chunk_id.as_bytes().as_slice(),
                    &vector_bytes,
                    emb.model_hash.as_slice(),
                    emb.dim as i32,
                    emb.norm,
                ],
            )
            .map_err(|e| CfsError::Database(e.to_string()))?;

        // Add to HNSW index
        let vector_f32 = emb.to_f32();
        {
            let mut hnsw = self.hnsw.write().unwrap();
            hnsw.insert(emb.id, vector_f32);
        }

        Ok(())
    }

    /// Get embedding by chunk ID
    pub fn get_embedding_for_chunk(&self, chunk_id: Uuid) -> Result<Option<Embedding>> {
        self.db
            .query_row(
                "SELECT id, chunk_id, vector, model_hash, dim, norm 
                 FROM embeddings WHERE chunk_id = ?1",
                params![chunk_id.as_bytes().as_slice()],
                |row| self.row_to_embedding(row),
            )
            .optional()
            .map_err(|e| CfsError::Database(e.to_string()))
    }

    /// Get embedding by ID
    pub fn get_embedding(&self, id: Uuid) -> Result<Option<Embedding>> {
        self.db
            .query_row(
                "SELECT id, chunk_id, vector, model_hash, dim, norm 
                 FROM embeddings WHERE id = ?1",
                params![id.as_bytes().as_slice()],
                |row| self.row_to_embedding(row),
            )
            .optional()
            .map_err(|e| CfsError::Database(e.to_string()))
    }

    fn row_to_embedding(&self, row: &rusqlite::Row) -> rusqlite::Result<Embedding> {
        let id_bytes: Vec<u8> = row.get(0)?;
        let chunk_id_bytes: Vec<u8> = row.get(1)?;
        let vector_bytes: Vec<u8> = row.get(2)?;
        let model_hash_bytes: Vec<u8> = row.get(3)?;
        let dim: i32 = row.get(4)?;
        let norm: f32 = row.get(5)?;

        let id = Uuid::from_slice(&id_bytes).expect("invalid uuid");
        let chunk_id = Uuid::from_slice(&chunk_id_bytes).expect("invalid uuid");
        let mut model_hash = [0u8; 32];
        model_hash.copy_from_slice(&model_hash_bytes);

        // Convert bytes back to f16 vector
        let vector: Vec<half::f16> = vector_bytes
            .chunks(2)
            .map(|bytes| half::f16::from_le_bytes([bytes[0], bytes[1]]))
            .collect();

        Ok(Embedding {
            id,
            chunk_id,
            vector,
            model_hash,
            dim: dim as u16,
            norm,
        })
    }

    /// Get chunk ID for an embedding ID
    pub fn get_chunk_id_for_embedding(&self, embedding_id: Uuid) -> Result<Option<Uuid>> {
        self.db
            .query_row(
                "SELECT chunk_id FROM embeddings WHERE id = ?1",
                params![embedding_id.as_bytes().as_slice()],
                |row| {
                    let id_bytes: Vec<u8> = row.get(0)?;
                    Ok(Uuid::from_slice(&id_bytes).expect("invalid uuid"))
                },
            )
            .optional()
            .map_err(|e| CfsError::Database(e.to_string()))
    }

    /// Get all embeddings
    pub fn get_all_embeddings(&self) -> Result<Vec<Embedding>> {
        let mut stmt = self
            .db
            .prepare("SELECT id, chunk_id, vector, model_hash, dim, norm FROM embeddings")
            .map_err(|e| CfsError::Database(e.to_string()))?;

        let embs = stmt
            .query_map([], |row| self.row_to_embedding(row))
            .map_err(|e| CfsError::Database(e.to_string()))?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| CfsError::Database(e.to_string()))?;

        Ok(embs)
    }

    /// Load the HNSW index from the database
    fn load_hnsw_index(&self) -> Result<()> {
        let embs = self.get_all_embeddings()?;
        let mut hnsw = self.hnsw.write().unwrap();
        
        for emb in embs {
            hnsw.insert(emb.id, emb.to_f32());
        }
        
        Ok(())
    }

    // ========== Edge operations ==========

    /// Add an edge to the graph
    pub fn add_edge(&mut self, edge: &Edge) -> Result<()> {
        self.db
            .execute(
                "INSERT OR REPLACE INTO edges (source, target, kind, weight) VALUES (?1, ?2, ?3, ?4)",
                params![
                    edge.source.as_bytes().as_slice(),
                    edge.target.as_bytes().as_slice(),
                    edge.kind as i32,
                    edge.weight.map(|w| w as i32),
                ],
            )
            .map_err(|e| CfsError::Database(e.to_string()))?;

        Ok(())
    }

    /// Get edges from a node
    pub fn get_edges(&self, node_id: Uuid) -> Result<Vec<Edge>> {
        let mut stmt = self
            .db
            .prepare("SELECT source, target, kind, weight FROM edges WHERE source = ?1")
            .map_err(|e| CfsError::Database(e.to_string()))?;

        let edges = stmt
            .query_map(params![node_id.as_bytes().as_slice()], |row| {
                let source_bytes: Vec<u8> = row.get(0)?;
                let target_bytes: Vec<u8> = row.get(1)?;
                let kind: i32 = row.get(2)?;
                let weight: Option<i32> = row.get(3)?;

                let source = Uuid::from_slice(&source_bytes).expect("invalid uuid");
                let target = Uuid::from_slice(&target_bytes).expect("invalid uuid");
                let kind = EdgeKind::from_u8(kind as u8).unwrap_or(EdgeKind::DocToChunk);

                Ok(Edge { 
                    source, 
                    target, 
                    kind,
                    weight: weight.map(|w| w as u16),
                })
            })
            .map_err(|e| CfsError::Database(e.to_string()))?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| CfsError::Database(e.to_string()))?;

        Ok(edges)
    }

    // ========== Vector search ==========

    /// Search for similar embeddings
    pub fn search(&self, query_vec: &[f32], k: usize) -> Result<Vec<(Uuid, f32)>> {
        let hnsw = self.hnsw.read().unwrap();
        Ok(hnsw.search(query_vec, k))
    }

    /// Lexical search using FTS5
    /// Get graph statistics
    pub fn search_lexical(&self, query: &str, k: usize) -> Result<Vec<(Uuid, f32)>> {
        let mut stmt = self
            .db
            .prepare(
                "SELECT id, rank FROM fts_chunks 
                 WHERE fts_chunks MATCH ?1 
                 ORDER BY rank LIMIT ?2",
            )
            .map_err(|e| CfsError::Database(e.to_string()))?;

        let results = stmt
            .query_map(params![query, k as i64], |row| {
                let id_bytes: Vec<u8> = row.get(0)?;
                let rank: f64 = row.get(1)?;
                
                let id = Uuid::from_slice(&id_bytes).expect("invalid uuid");
                // Convert rank to a similarity score (approximate)
                // SQLite FTS5 rank: lower is better (usually negative).
                // We'll just return it raw for now, or normalize it superficially.
                Ok((id, -rank as f32))
            })
            .map_err(|e| CfsError::Database(e.to_string()))?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| CfsError::Database(e.to_string()))?;

        Ok(results)
    }

    // ========== State ==========

    /// Compute Merkle root of the current state
    /// 
    /// This is strictly semantic and content-addressable. It does NOT include
    /// timestamps, UUIDs (except as IDs for relationships), or device-specific metadata.
    pub fn compute_merkle_root(&self) -> Result<[u8; 32]> {
        let mut hasher = blake3::Hasher::new();

        // 1. Documents (sorted by ID)
        let mut docs = self.get_all_documents()?;
        docs.sort_by_key(|d| d.id);
        for doc in docs {
            hasher.update(doc.id.as_bytes());
            hasher.update(&doc.hash);
            // We ignore mtime/path as they are metadata, but hash is semantic content.
        }

        // 2. Chunks (sorted by ID)
        // We use a query to get them sorted to avoid huge Vec in memory if possible, 
        // but for MVP we get all and sort.
        let mut stmt = self.db.prepare("SELECT id, text_hash FROM chunks ORDER BY id")
            .map_err(|e| CfsError::Database(e.to_string()))?;
        let chunk_iter = stmt.query_map([], |row| {
            let id_bytes: Vec<u8> = row.get(0)?;
            let hash_bytes: Vec<u8> = row.get(1)?;
            Ok((id_bytes, hash_bytes))
        }).map_err(|e| CfsError::Database(e.to_string()))?;

        for chunk in chunk_iter {
            let (id, hash) = chunk.map_err(|e| CfsError::Database(e.to_string()))?;
            hasher.update(&id);
            hasher.update(&hash);
        }

        // 3. Embeddings (sorted by ID)
        let mut stmt = self.db.prepare("SELECT id, vector, model_hash FROM embeddings ORDER BY id")
            .map_err(|e| CfsError::Database(e.to_string()))?;
        let emb_iter = stmt.query_map([], |row| {
            let id_bytes: Vec<u8> = row.get(0)?;
            let vec_bytes: Vec<u8> = row.get(1)?;
            let model_hash: Vec<u8> = row.get(2)?;
            Ok((id_bytes, vec_bytes, model_hash))
        }).map_err(|e| CfsError::Database(e.to_string()))?;

        for emb in emb_iter {
            let (id, vec, model) = emb.map_err(|e| CfsError::Database(e.to_string()))?;
            hasher.update(&id);
            hasher.update(&vec);
            hasher.update(&model);
        }

        // 4. Edges (sorted by source, target, kind)
        let mut stmt = self.db.prepare("SELECT source, target, kind, weight FROM edges ORDER BY source, target, kind")
            .map_err(|e| CfsError::Database(e.to_string()))?;
        let edge_iter = stmt.query_map([], |row| {
            let s: Vec<u8> = row.get(0)?;
            let t: Vec<u8> = row.get(1)?;
            let k: i32 = row.get(2)?;
            let w: Option<i32> = row.get(3)?;
            Ok((s, t, k, w))
        }).map_err(|e| CfsError::Database(e.to_string()))?;

        for edge in edge_iter {
            let (s, t, k, w) = edge.map_err(|e| CfsError::Database(e.to_string()))?;
            hasher.update(&s);
            hasher.update(&t);
            hasher.update(&k.to_le_bytes());
            if let Some(weight) = w {
                hasher.update(&weight.to_le_bytes());
            }
        }

        Ok(*hasher.finalize().as_bytes())
    }

    /// Get the latest state root
    pub fn get_latest_root(&self) -> Result<Option<cfs_core::StateRoot>> {
        use rusqlite::OptionalExtension;
        self.db
            .query_row(
                "SELECT hash, parent, timestamp, device_id, signature, seq 
                 FROM state_roots ORDER BY seq DESC LIMIT 1",
                [],
                |row| {
                    let hash_bytes: Vec<u8> = row.get(0)?;
                    let parent_bytes: Option<Vec<u8>> = row.get(1)?;
                    let timestamp: i64 = row.get(2)?;
                    let device_id_bytes: Vec<u8> = row.get(3)?;
                    let signature_bytes: Vec<u8> = row.get(4)?;
                    let seq: i64 = row.get(5)?; // Read as i64 (sqlite INTEGER)

                    let mut hash = [0u8; 32];
                    hash.copy_from_slice(&hash_bytes);

                    let parent = parent_bytes.map(|b| {
                        let mut p = [0u8; 32];
                        p.copy_from_slice(&b);
                        p
                    });

                    let device_id = Uuid::from_slice(&device_id_bytes).expect("invalid uuid");
                    
                    let mut signature = [0u8; 64];
                    signature.copy_from_slice(&signature_bytes);

                    Ok(cfs_core::StateRoot {
                        hash,
                        parent,
                        timestamp,
                        device_id,
                        signature,
                        seq: seq as u64,
                    })
                },
            )
            .optional()
            .map_err(|e| CfsError::Database(e.to_string()))
    }

    /// Set the latest state root
    pub fn set_latest_root(&mut self, root: &cfs_core::StateRoot) -> Result<()> {
        self.db
            .execute(
                "INSERT OR REPLACE INTO state_roots (hash, parent, timestamp, device_id, signature, seq) 
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![
                    root.hash.as_slice(),
                    root.parent.as_ref().map(|p| p.as_slice()),
                    root.timestamp,
                    root.device_id.as_bytes().as_slice(),
                    root.signature.as_slice(),
                    root.seq as i64,
                ],
            )
            .map_err(|e| CfsError::Database(e.to_string()))?;

        Ok(())
    }

    /// Apply a cognitive diff to the graph store
    pub fn apply_diff(&mut self, diff: &cfs_core::CognitiveDiff) -> Result<()> {
        let tx = self.db.transaction().map_err(|e| CfsError::Database(e.to_string()))?;

        // 1. Remove items (Delete)
        for id in &diff.removed_doc_ids {
            tx.execute("DELETE FROM documents WHERE id = ?1", params![id.as_bytes().as_slice()])
                .map_err(|e| CfsError::Database(e.to_string()))?;
        }
        for id in &diff.removed_chunk_ids {
            tx.execute("DELETE FROM chunks WHERE id = ?1", params![id.as_bytes().as_slice()])
                 .map_err(|e| CfsError::Database(e.to_string()))?;
        }
        for id in &diff.removed_embedding_ids {
            tx.execute("DELETE FROM embeddings WHERE id = ?1", params![id.as_bytes().as_slice()])
                 .map_err(|e| CfsError::Database(e.to_string()))?;
        }
        for (source, target) in &diff.removed_edges {
            tx.execute(
                "DELETE FROM edges WHERE source = ?1 AND target = ?2",
                params![source.as_bytes().as_slice(), target.as_bytes().as_slice()],
            )
            .map_err(|e| CfsError::Database(e.to_string()))?;
        }

        // 2. Add/Update items (Insert or Replace)
        for doc in &diff.added_docs {
             tx.execute(
                "INSERT OR REPLACE INTO documents (id, path, hash, mtime, size, mime_type) 
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![
                    doc.id.as_bytes().as_slice(),
                    doc.path.to_string_lossy().as_ref(), // Store as string
                    doc.hash.as_slice(),
                    doc.mtime,
                    doc.size as i64,
                    &doc.mime_type,
                ],
            ).map_err(|e| CfsError::Database(e.to_string()))?;
        }
        for doc in &diff.updated_docs {
             tx.execute(
                "INSERT OR REPLACE INTO documents (id, path, hash, mtime, size, mime_type) 
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![
                    doc.id.as_bytes().as_slice(),
                    doc.path.to_string_lossy().as_ref(),
                    doc.hash.as_slice(),
                    doc.mtime,
                    doc.size as i64,
                    &doc.mime_type,
                ],
            ).map_err(|e| CfsError::Database(e.to_string()))?;
        }

        for chunk in &diff.added_chunks {
            tx.execute(
                "INSERT OR REPLACE INTO chunks (id, doc_id, text, offset, len, seq, text_hash) 
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                params![
                    chunk.id.as_bytes().as_slice(),
                    chunk.doc_id.as_bytes().as_slice(),
                    &chunk.text,
                    chunk.offset,
                    chunk.len,
                    chunk.seq,
                    chunk.text_hash.as_slice(),
                ],
            ).map_err(|e| CfsError::Database(e.to_string()))?;
        }

        for emb in &diff.added_embeddings {
             let vector_bytes: Vec<u8> = emb.vector.iter().flat_map(|f| f.to_le_bytes()).collect();
             tx.execute(
                "INSERT OR REPLACE INTO embeddings (id, chunk_id, vector, model_hash, dim, norm) 
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![
                    emb.id.as_bytes().as_slice(),
                    emb.chunk_id.as_bytes().as_slice(),
                    &vector_bytes,
                    emb.model_hash.as_slice(),
                    emb.dim as i32,
                    emb.norm,
                ],
            ).map_err(|e| CfsError::Database(e.to_string()))?;
        }

        for edge in &diff.added_edges {
            tx.execute(
                "INSERT OR REPLACE INTO edges (source, target, kind, weight) VALUES (?1, ?2, ?3, ?4)",
                params![
                    edge.source.as_bytes().as_slice(),
                    edge.target.as_bytes().as_slice(),
                    edge.kind as i32,
                    edge.weight.map(|w| w as i32),
                ],
            ).map_err(|e| CfsError::Database(e.to_string()))?;
        }
        
        // 3. Update State Root
        tx.execute(
             "INSERT OR REPLACE INTO state_roots (hash, parent, timestamp, device_id, signature, seq) 
              VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
             params![
                 diff.metadata.new_root.as_slice(),
                 if diff.metadata.prev_root == [0u8; 32] { None } else { Some(diff.metadata.prev_root.as_slice()) },
                 diff.metadata.timestamp,
                 diff.metadata.device_id.as_bytes().as_slice(),
                 [0u8; 64].as_slice(), // Dummy signature as we don't have it in Diff
                 diff.metadata.seq as i64,
             ],
         ).map_err(|e| CfsError::Database(e.to_string()))?;

        tx.commit().map_err(|e| CfsError::Database(e.to_string()))?;
        
        // 4. Update HNSW Index (Best effort for added embeddings)
        {
            let mut hnsw = self.hnsw.write().unwrap();
            for emb in &diff.added_embeddings {
                let vec_f32 = emb.to_f32();
                hnsw.insert(emb.id, vec_f32);
            }
        }

        Ok(())
    }

    /// Clear all data from the graph store (for fresh sync)
    pub fn clear_all(&mut self) -> Result<()> {
        self.db.execute_batch(
            r#"
            DELETE FROM embeddings;
            DELETE FROM chunks;
            DELETE FROM documents;
            DELETE FROM edges;
            DELETE FROM state_roots;
            "#,
        )
        .map_err(|e| CfsError::Database(e.to_string()))?;

        // Reset HNSW index
        *self.hnsw.write().unwrap() = HnswIndex::new();

        info!("Cleared all data from graph store for fresh sync");
        Ok(())
    }

    /// Get statistics about the graph store
    pub fn stats(&self) -> Result<GraphStats> {
        let doc_count: i64 = self
            .db
            .query_row("SELECT COUNT(*) FROM documents", [], |row| row.get(0))
            .map_err(|e| CfsError::Database(e.to_string()))?;

        let chunk_count: i64 = self
            .db
            .query_row("SELECT COUNT(*) FROM chunks", [], |row| row.get(0))
            .map_err(|e| CfsError::Database(e.to_string()))?;

        let embedding_count: i64 = self
            .db
            .query_row("SELECT COUNT(*) FROM embeddings", [], |row| row.get(0))
            .map_err(|e| CfsError::Database(e.to_string()))?;

        let edge_count: i64 = self
            .db
            .query_row("SELECT COUNT(*) FROM edges", [], |row| row.get(0))
            .map_err(|e| CfsError::Database(e.to_string()))?;

        Ok(GraphStats {
            documents: doc_count as usize,
            chunks: chunk_count as usize,
            embeddings: embedding_count as usize,
            edges: edge_count as usize,
        })
    }
}

/// Statistics about the graph store
#[derive(Debug, Clone)]
pub struct GraphStats {
    pub documents: usize,
    pub chunks: usize,
    pub embeddings: usize,
    pub edges: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_document_roundtrip() {
        let mut store = GraphStore::in_memory().unwrap();
        
        let doc = Document::new(PathBuf::from("test.md"), b"Hello, world!", 12345);
        store.insert_document(&doc).unwrap();
        
        let retrieved = store.get_document(doc.id).unwrap().unwrap();
        assert_eq!(retrieved.id, doc.id);
        assert_eq!(retrieved.path, doc.path);
        assert_eq!(retrieved.hash, doc.hash);
    }

    #[test]
    fn test_chunk_operations() {
        let mut store = GraphStore::in_memory().unwrap();
        
        let doc = Document::new(PathBuf::from("test.md"), b"Content", 0);
        store.insert_document(&doc).unwrap();
        
        let chunk = Chunk::new(doc.id, "Test chunk text".to_string(), 0, 0);
        store.insert_chunk(&chunk).unwrap();
        
        let chunks = store.get_chunks_for_doc(doc.id).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "Test chunk text");
    }

    #[test]
    fn test_embedding_and_search() {
        let mut store = GraphStore::in_memory().unwrap();
        
        let doc = Document::new(PathBuf::from("test.md"), b"Content", 0);
        store.insert_document(&doc).unwrap();
        
        let chunk = Chunk::new(doc.id, "Test text".to_string(), 0, 0);
        store.insert_chunk(&chunk).unwrap();
        
        // Create a simple embedding
        let vector = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let emb = Embedding::new(chunk.id, &vector, [0u8; 32]);
        store.insert_embedding(&emb).unwrap();
        
        // Search with same vector should return it
        let results = store.search(&vector, 1).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].1 > 0.99); // High similarity
    }

    #[test]
    fn test_edge_operations() {
        let mut store = GraphStore::in_memory().unwrap();
        
        let source = Uuid::new_v4();
        let target = Uuid::new_v4();
        let edge = Edge::new(source, target, EdgeKind::DocToChunk);
        
        store.add_edge(&edge).unwrap();
        
        let edges = store.get_edges(source).unwrap();
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].target, target);
    }

    #[test]
    fn test_merkle_root() {
        let mut store = GraphStore::in_memory().unwrap();
        
        let root1 = store.compute_merkle_root().unwrap();
        
        let doc = Document::new(PathBuf::from("test.md"), b"Hello", 0);
        store.insert_document(&doc).unwrap();
        
        let root2 = store.compute_merkle_root().unwrap();
        
        // Roots should be different after adding content
        assert_ne!(root1, root2);
    }

    #[test]
    fn test_stats() {
        let mut store = GraphStore::in_memory().unwrap();
        
        let doc = Document::new(PathBuf::from("test.md"), b"Content", 0);
        store.insert_document(&doc).unwrap();
        
        let stats = store.stats().unwrap();
        assert_eq!(stats.documents, 1);
        assert_eq!(stats.chunks, 0);
    }
}
