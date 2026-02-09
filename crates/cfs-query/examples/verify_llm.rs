use cfs_query::{QueryEngine, GenerationResult};
use cfs_core::{Document, Chunk};
use std::sync::{Arc, Mutex};
use tempfile::TempDir;
use uuid::Uuid;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- Phase 2: LLM RAG Verification ---");

    // 1. Setup mock graph with ZK document
    let temp = TempDir::new()?;
    let db_path = temp.path().join("verify_phase2.db");
    let mut graph = cfs_graph::GraphStore::open(db_path.to_str().unwrap())?;
    
    let content = "Zero-knowledge proofs (ZKP) allow a prover to convince a verifier that a statement is true without revealing any information beyond the validity of the statement itself. Common types include zk-SNARKs and zk-STARKs. They are used in privacy blockchains like Zcash and for scaling via ZK-rollups.";
    let doc = Document::new("zk_info.md".into(), content.as_bytes(), 0);
    graph.insert_document(&doc)?;
    
    let chunk = Chunk {
        id: Uuid::new_v4(),
        doc_id: doc.id,
        text: content.to_string(),
        offset: 0,
        len: content.len() as u32,
        seq: 0,
        text_hash: [0; 32],
    };
    graph.insert_chunk(&chunk)?;
    
    // 2. Add embedding
    println!("Embedding knowledge substrate...");
    let embedder = Arc::new(cfs_embeddings::EmbeddingEngine::new()?);
    let vec = embedder.embed(&chunk.text)?;
    let emb = cfs_core::Embedding::new(chunk.id, &vec, embedder.model_hash());
    graph.insert_embedding(&emb)?;
    
    let qe = QueryEngine::new(Arc::new(Mutex::new(graph)), embedder);
    
    // 3. Run RAG Query
    let query = "What are the common types of zero-knowledge proofs and where are they used?";
    println!("Querying: '{}'", query);
    
    match qe.generate_answer(query).await {
        Ok(res) => {
            println!("\n[AI RESPONSE]\n{}", res.answer);
            println!("\n[ASSEMBLED CONTEXT]\n{}", res.context);
            println!("\n[STATS] Latency: {}ms", res.latency_ms);
            
            if res.answer.to_lowercase().contains("snark") || res.answer.to_lowercase().contains("zcash") {
                println!("\n✅ VERIFICATION SUCCESS: AI response is grounded in context.");
            } else {
                println!("\n⚠️ VERIFICATION WARNING: AI response may not be fully grounded.");
            }
        },
        Err(e) => {
            println!("\n❌ VERIFICATION FAILED: {}", e);
            if e.to_string().contains("reqwest") || e.to_string().contains("Ollama") {
                println!("Hint: Is Ollama running?");
            }
        }
    }
    
    Ok(())
}
