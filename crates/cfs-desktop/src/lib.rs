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
        // Simplified event handling
        match event.kind {
            EventKind::Create(_) | EventKind::Modify(_) => {
                for path in event.paths {
                    if path.is_file() {
                        if let Err(e) = self.process_file(&path) {
                            tracing::error!("Failed to process file {:?}: {}", path, e);
                        }
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
        for dir in &self.watch_dirs {
            for entry in std::fs::read_dir(dir).map_err(|e| CfsError::Io(e))? {
                let entry = entry.map_err(|e| CfsError::Io(e))?;
                let path = entry.path();
                if path.is_file() {
                    if let Err(e) = worker.process_file(&path) {
                        tracing::error!("Failed to process existing file {:?}: {}", path, e);
                    }
                }
            }
        }

        // Start sync loop
        let sync_mgr = self.sync_manager.clone();
        tokio::spawn(async move {
            info!("Starting background sync loop (every 5s)");
            loop {
                tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
                
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
