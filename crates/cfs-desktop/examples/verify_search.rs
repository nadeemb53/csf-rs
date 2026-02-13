use cfs_graph::GraphStore;
use cfs_query::QueryEngine;
use cfs_embeddings::EmbeddingEngine;
use std::sync::{Arc, Mutex};

fn main() -> cfs_core::Result<()> {
    let db_path = "/Users/nadeem/dev/CFS/apps/macos/src-tauri/.cfs/graph.db";
    let graph = GraphStore::open(db_path)?;
    let graph_arc = Arc::new(Mutex::new(graph));
    let embedder = Arc::new(EmbeddingEngine::new()?);
    let qe = QueryEngine::new(graph_arc, embedder);
    
    let query = "ethereum";
    println!("Querying: '{}'", query);
    
    let results = qe.search(query, 5)?;
    println!("Results found: {}", results.len());
    for (i, res) in results.iter().enumerate() {
        println!("  {}. [Score: {:.4}] {} -> {}", 
            i + 1, res.score, res.doc_path, res.chunk.text.trim());
    }
    
    Ok(())
}
