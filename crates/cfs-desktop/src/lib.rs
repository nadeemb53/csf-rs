//! CFS Desktop - Tauri desktop application
//!
//! This crate will be the Tauri backend for the desktop app.
//! For now, it provides a library with the core logic.

use cfs_core::{CfsError, Document, Embedding, Result};
use cfs_embeddings::EmbeddingEngine;
use cfs_graph::GraphStore;
use cfs_parser::{Chunker, ChunkConfig};
use tracing::info;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use uuid::Uuid;

mod watcher;
mod sync;

pub use watcher::FileWatcher;
use sync::SyncManager;

#[derive(Clone)]
pub struct DesktopApp {
    watch_dirs: Vec<PathBuf>,
    graph: Arc<Mutex<GraphStore>>,
    sync_manager: Arc<SyncManager>,
    status: Arc<Mutex<String>>,
}

/// Worker that processes files
struct IngestionWorker {
    graph: Arc<Mutex<GraphStore>>,
    embedder: Arc<EmbeddingEngine>,
    chunker: Chunker,
    sync_manager: Arc<SyncManager>,
    status: Arc<Mutex<String>>,
}

impl IngestionWorker {


    fn process_event(&self, event: notify::Event) {
        use notify::EventKind;

        // Log all events for debugging
        tracing::debug!("FS Event: {:?}", event);

        match event.kind {
            EventKind::Create(_) | EventKind::Modify(_) => {
                for path in event.paths {
                    if path.is_file() {
                        let filename = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                        if filename == ".DS_Store" || filename.starts_with('.') {
                            continue;
                        }
                        if let Err(e) = self.process_file(&path) {
                            tracing::error!("Failed to process file {:?}: {}", path, e);
                        }
                    }
                }
            }
            EventKind::Remove(_) => {
                info!("Remove event detected: {:?}", event.paths);
                for path in event.paths {
                    if let Err(e) = self.remove_file(&path) {
                        tracing::error!("Failed to remove file {:?}: {}", path, e);
                    }
                }
            }
            _ => {}
        }
    }

    fn process_file(&self, path: &PathBuf) -> Result<()> {
        let filename = path.file_name().and_then(|n| n.to_str()).unwrap_or("unknown");
        *self.status.lock().unwrap() = format!("Parsing {}", filename);
        let start_total = std::time::Instant::now();
        
        // 1. Read metadata
        let metadata = std::fs::metadata(path).map_err(|e| CfsError::Io(e))?;
        let mtime = metadata.modified().unwrap_or(std::time::SystemTime::UNIX_EPOCH)
            .duration_since(std::time::UNIX_EPOCH).unwrap().as_secs() as i64;

        // 2. Parse
        let start_parse = std::time::Instant::now();
        let text = cfs_parser::parse_file(path)?;
        let parse_duration = start_parse.elapsed();

        // 3. Create Document
        let doc = Document::new(path.clone(), text.as_bytes(), mtime);
        
        // 4. Chunk
        *self.status.lock().unwrap() = format!("Chunking {}", filename);
        let start_chunk = std::time::Instant::now();
        let chunks = self.chunker.chunk(doc.id, &text)?;
        let chunk_duration = start_chunk.elapsed();

        // 5. Embed (with Hierarchical Hash calculation)
        *self.status.lock().unwrap() = format!("Embedding {}", filename);
        let start_embed = std::time::Instant::now();
        let chunk_hashes: Vec<[u8; 32]> = chunks.iter().map(|c| c.text_hash).collect();
        let h_hash = Document::compute_hierarchical_hash(&chunk_hashes);
        
        let chunk_texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
        let embeddings_vec = self.embedder.embed_batch(&chunk_texts)?;
        let embed_duration = start_embed.elapsed();

        // 6. Store
        let start_store = std::time::Instant::now();
        {
            let mut graph = self.graph.lock().unwrap();
            
            // Insert Doc (could update with hierarchical hash)
            graph.insert_document(&doc)?;
            self.sync_manager.record_add_doc(doc.clone());

            // Insert Chunks & Embeddings
            for (i, chunk) in chunks.iter().enumerate() {
                graph.insert_chunk(chunk)?;
                self.sync_manager.record_add_chunk(chunk.clone());

                if let Some(vec) = embeddings_vec.get(i) {
                     let emb = Embedding::new(
                         chunk.id,
                         vec,
                         self.embedder.model_hash(),
                     );
                     graph.insert_embedding(&emb)?;
                     self.sync_manager.record_add_embedding(emb.clone());
                }
            }
        }
        let store_duration = start_store.elapsed();
        let total_duration = start_total.elapsed();
        *self.status.lock().unwrap() = "Idle".to_string();
        
        info!(
            "Processed {:?}: chunks={}, h_hash={}, parse={:?}, chunk={:?}, embed={:?}, store={:?}, total={:?}",
            path,
            chunks.len(),
            hex::encode(&h_hash[..4]),
            parse_duration,
            chunk_duration,
            embed_duration,
            store_duration,
            total_duration
        );
        Ok(())
    }

    fn remove_file(&self, path: &PathBuf) -> Result<()> {
        let filename = path.file_name().and_then(|n| n.to_str()).unwrap_or("unknown");
        *self.status.lock().unwrap() = format!("Removing {}", filename);

        info!("Remove event received for: {:?}", path);

        let mut graph = self.graph.lock().unwrap();
        if let Some(doc) = graph.get_document_by_path(path)? {
            info!("Removing document from substrate: {:?}", path);

            // 1. Get associated IDs for sync
            let chunks = graph.get_chunks_for_doc(doc.id)?;
            let chunk_ids: Vec<Uuid> = chunks.iter().map(|c| c.id).collect();
            let mut embedding_ids = Vec::new();
            for cid in &chunk_ids {
                if let Some(emb) = graph.get_embedding_for_chunk(*cid)? {
                    embedding_ids.push(emb.id);
                }
            }

            // 2. Perform deletion in DB
            graph.delete_document(doc.id)?;

            // 3. Record for sync
            self.sync_manager.record_remove_doc(doc.id, chunk_ids, embedding_ids);

            info!("Document removed successfully: {:?}", path);
        } else {
            // Document not found - might be a path mismatch
            tracing::warn!("Remove event for unknown document: {:?}", path);
        }

        *self.status.lock().unwrap() = "Idle".to_string();
        Ok(())
    }
}

impl DesktopApp {
    pub fn new(data_dir: PathBuf) -> Result<Self> {
        std::fs::create_dir_all(&data_dir).map_err(|e| CfsError::Io(e))?;

        let db_path = data_dir.join("graph.db");
        let graph = GraphStore::open(db_path.to_str().unwrap())?;
        let graph_arc = Arc::new(Mutex::new(graph));

        // Init SyncManager
        let key_path = data_dir.join("cfs.key");
        // Use local relay for dev
        let sync_manager = Arc::new(SyncManager::new(
            "http://localhost:8080",
            "dummy_token",
            graph_arc.clone(),
            &key_path,
        )?);

        Ok(Self {
            watch_dirs: Vec::new(),
            graph: graph_arc,
            sync_manager,
            status: Arc::new(Mutex::new("Idle".to_string())),
        })
    }

    pub fn add_watch_dir(&mut self, path: PathBuf) -> Result<()> {
        if !path.exists() {
            return Err(CfsError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Path not found: {:?}", path),
            )));
        }
        self.watch_dirs.push(path);
        Ok(())
    }

    pub async fn start(&self) -> Result<()> {
        let (tx, rx) = std::sync::mpsc::channel();
        let mut watcher = FileWatcher::new(tx)?;

        for dir in &self.watch_dirs {
            watcher.watch(dir)?;
        }

        let embedder = Arc::new(EmbeddingEngine::new()?);
        
        let worker = IngestionWorker {
            graph: self.graph.clone(),
            embedder,
            chunker: Chunker::new(ChunkConfig::default()),
            sync_manager: self.sync_manager.clone(),
            status: self.status.clone(),
        };

        // Initial crawl
        let mut found_paths = std::collections::HashSet::new();
        for dir in &self.watch_dirs {
            for entry in std::fs::read_dir(dir).map_err(|e| CfsError::Io(e))? {
                let entry = entry.map_err(|e| CfsError::Io(e))?;
                let path = entry.path();
                if path.is_file() {
                    let filename = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                    if filename == ".DS_Store" || filename.starts_with('.') {
                        continue;
                    }
                    found_paths.insert(path.clone());
                    if let Err(e) = worker.process_file(&path) {
                        tracing::error!("Failed to process existing file {:?}: {}", path, e);
                    }
                }
            }
        }

        // Pruning: Remove docs from DB that are no longer in watch dirs
        {
            let mut graph = self.graph.lock().unwrap();
            let all_docs = graph.get_all_documents()?;
            for doc in all_docs {
                // Only prune if the doc is in one of the watch dirs but not found in the crawl
                let in_watch_dir = self.watch_dirs.iter().any(|d| doc.path.starts_with(d));
                if in_watch_dir && !found_paths.contains(&doc.path) {
                    info!("Pruning stale document from substrate: {:?}", doc.path);
                    // Get associated IDs for sync
                    let chunks = graph.get_chunks_for_doc(doc.id)?;
                    let chunk_ids: Vec<Uuid> = chunks.iter().map(|c| c.id).collect();
                    let mut embedding_ids = Vec::new();
                    for cid in &chunk_ids {
                        if let Some(emb) = graph.get_embedding_for_chunk(*cid)? {
                            embedding_ids.push(emb.id);
                        }
                    }
                    graph.delete_document(doc.id)?;
                    // Record removal diff so the relay (and other devices) know it's gone
                    self.sync_manager.record_remove_doc(doc.id, chunk_ids, embedding_ids);
                }
            }
        }

        // Start sync loop with periodic prune
        let sync_mgr = self.sync_manager.clone();
        let app_clone = self.clone();
        tokio::spawn(async move {
            info!("Starting background sync loop (every 5s)");
            let mut prune_counter = 0u32;
            loop {
                tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;

                // Prune stale files every 30 seconds (6 cycles)
                // FSEvents on macOS often misses Remove events
                prune_counter += 1;
                if prune_counter >= 6 {
                    prune_counter = 0;
                    match app_clone.prune_stale_documents() {
                        Ok(count) if count > 0 => {
                            info!("Auto-pruned {} stale documents", count);
                        }
                        Err(e) => {
                            tracing::error!("Auto-prune failed: {}", e);
                        }
                        _ => {}
                    }
                }

                // Pull first
                if let Err(e) = sync_mgr.pull().await {
                    tracing::error!("Sync pull failed: {}", e);
                }

                // Periodic push
                if let Err(e) = sync_mgr.push().await {
                    tracing::error!("Sync push failed: {}", e);
                }
            }
        });

        // Run ingestion loop (blocking)
        tokio::task::spawn_blocking(move || {
            worker.run(rx);
        }).await.map_err(|e| CfsError::Io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?;

        Ok(())
    }

    pub fn graph(&self) -> Arc<Mutex<GraphStore>> {
        self.graph.clone()
    }

    pub fn status(&self) -> String {
        self.status.lock().unwrap().clone()
    }

    /// Prune stale documents (files that no longer exist) and record for sync
    pub fn prune_stale_documents(&self) -> Result<usize> {
        let mut graph = self.graph.lock().unwrap();
        let all_docs = graph.get_all_documents()?;

        let mut removed = 0;
        for doc in all_docs {
            if !doc.path.exists() {
                info!("Pruning stale document: {:?}", doc.path);

                // Get associated IDs for sync
                let chunks = graph.get_chunks_for_doc(doc.id)?;
                let chunk_ids: Vec<Uuid> = chunks.iter().map(|c| c.id).collect();
                let mut embedding_ids = Vec::new();
                for cid in &chunk_ids {
                    if let Some(emb) = graph.get_embedding_for_chunk(*cid)? {
                        embedding_ids.push(emb.id);
                    }
                }

                // Delete from DB
                graph.delete_document(doc.id)?;

                // Record for sync
                self.sync_manager.record_remove_doc(doc.id, chunk_ids, embedding_ids);

                removed += 1;
            }
        }

        Ok(removed)
    }
}

// Need to update IngestionWorker::run signature
impl IngestionWorker {
    fn run(&self, rx: std::sync::mpsc::Receiver<notify::Result<notify::Event>>) {
        loop {
            match rx.recv() {
                Ok(res) => {
                    match res {
                        Ok(event) => {
                            self.process_event(event);
                        }
                        Err(e) => {
                             tracing::error!("Watch error event: {}", e);
                        }
                    }
                }
                Err(e) => {
                    tracing::error!("Watch channel closed: {}", e);
                    break;
                }
            }
        }
    }
}
