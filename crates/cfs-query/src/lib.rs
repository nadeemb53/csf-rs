//! CFS Query - RAG query engine
//!
//! Provides semantic search and chat over the knowledge graph.

use cfs_core::{Chunk, CfsError, Result};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tracing::{info, warn};
use uuid::Uuid;
use std::collections::HashMap;

/// Search result with relevance score
#[derive(Debug, Clone, serde::Serialize)]
pub struct SearchResult {
    /// The matching chunk
    pub chunk: Chunk,
    /// Similarity score (0.0-1.0)
    pub score: f32,
    /// Path to the source document
    pub doc_path: PathBuf,
}

/// Result of an LLM generation
#[derive(Debug, Clone, serde::Serialize)]
pub struct GenerationResult {
    /// The generated answer
    pub answer: String,
    /// The context used to generate the answer
    pub context: String,
    /// Latency in milliseconds
    pub latency_ms: u64,
}

/// Trait for the CFS Intelligence Module (IM)
/// 
/// An IntelligenceEngine is a read-only consumer of the semantic substrate.
/// It synthesizes information from retrieved context into human-readable answers.
#[async_trait::async_trait]
pub trait IntelligenceEngine: Send + Sync {
    /// Generate a synthesized answer from the provided context and query.
    /// This is a read-only operation and cannot mutate the underlying graph.
    async fn generate(&self, context: &str, query: &str) -> Result<String>;
}

/// Ollama-based LLM generator (for desktop/server use)
pub struct OllamaGenerator {
    base_url: String,
    model: String,
}

impl OllamaGenerator {
    /// Create a new Ollama generator
    pub fn new(base_url: String, model: String) -> Self {
        Self { base_url, model }
    }
}

#[async_trait::async_trait]
impl IntelligenceEngine for OllamaGenerator {
    async fn generate(&self, context: &str, query: &str) -> Result<String> {
        // Engineered for consistency with mobile prompt
        let prompt = format!(
            "<|im_start|>system\nYou are a context reading machine. You do not have knowledge of the outside world.\n- Read the Context below carefully.\n- If the answer to the Query is in the Context, output it.\n- If the answer is NOT in the Context, say 'Information is missing from the substrate.' and nothing else.\n- Do not make up facts.\n<|im_end|>\n<|im_start|>user\nContext:\n{}\n\nQuery: {}<|im_end|>\n<|im_start|>assistant\n",
            context, query
        );

        let client = reqwest::Client::new();
        let payload = serde_json::json!({
            "model": self.model,
            "prompt": prompt,
            "stream": false
        });

        let url = format!("{}/api/generate", self.base_url);
        let res = client
            .post(&url)
            .json(&payload)
            .send()
            .await
            .map_err(|e| CfsError::Embedding(format!("Ollama request failed: {}", e)))?;

        let json: serde_json::Value = res
            .json()
            .await
            .map_err(|e| CfsError::Parse(e.to_string()))?;

        let answer = json["response"]
            .as_str()
            .ok_or_else(|| CfsError::Parse("Invalid Ollama response".into()))?
            .to_string();

        Ok(answer)
    }
}

/// Query engine for semantic search
pub struct QueryEngine {
    graph: Arc<Mutex<cfs_graph::GraphStore>>,
    embedder: Arc<cfs_embeddings::EmbeddingEngine>,
    intelligence: Option<Box<dyn IntelligenceEngine>>,
}

impl QueryEngine {
    /// Create a new query engine
    pub fn new(
        graph: Arc<Mutex<cfs_graph::GraphStore>>,
        embedder: Arc<cfs_embeddings::EmbeddingEngine>,
    ) -> Self {
        Self {
            graph,
            embedder,
            intelligence: None,
        }
    }

    /// Set the intelligence engine for RAG
    pub fn with_intelligence(mut self, intelligence: Box<dyn IntelligenceEngine>) -> Self {
        self.intelligence = Some(intelligence);
        self
    }

    /// Search for relevant chunks using a hybrid (semantic + lexical) approach
    pub fn search(&self, query: &str, k: usize) -> Result<Vec<SearchResult>> {
        info!("Hybrid search for: '{}'", query);

        // 1. Semantic Search (Vector)
        let query_vec = self
            .embedder
            .embed(query)
            .map_err(|e| CfsError::Embedding(format!("Failed to embed query: {}", e)))?;

        let semantic_results = {
            let graph = self.graph.lock().unwrap();
            graph.search(&query_vec, k)?
        };

        // 2. Lexical Search (FTS5)
        let lexical_results = {
            let graph = self.graph.lock().unwrap();
            // Surround query with quotes for better FTS performance if it has spaces
            let fts_query = if query.contains(' ') {
                format!("\"{}\"", query.replace('"', ""))
            } else {
                query.to_string()
            };
            graph.search_lexical(&fts_query, k).unwrap_or_else(|e| {
                warn!("Lexical search failed: {}. Falling back to semantic only.", e);
                Vec::new()
            })
        };

        // 3. Merge Results using Reciprocal Rank Fusion (RRF)
        // Score = \sum_{r \in R} \frac{1}{60 + rank(r)}
        // IMPORTANT: Standardize on Chunk IDs before merging to avoid duplicates
        let mut scores: HashMap<Uuid, f32> = HashMap::new();

        {
            let graph = self.graph.lock().unwrap();
            
            for (i, (emb_id, _)) in semantic_results.iter().enumerate() {
                if let Ok(Some(chunk_id)) = graph.get_chunk_id_for_embedding(*emb_id) {
                    let score = 1.0 / (60.0 + i as f32);
                    *scores.entry(chunk_id).or_insert(0.0) += score;
                }
            }

            for (i, (chunk_id, _)) in lexical_results.iter().enumerate() {
                let score = 1.0 / (60.0 + i as f32);
                *scores.entry(*chunk_id).or_insert(0.0) += score;
            }
        }

        // Sort by fused score
        let mut fused: Vec<(Uuid, f32)> = scores.into_iter().collect();
        fused.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        fused.truncate(k);

        // 4. Retrieve chunks and docs
        let mut search_results = Vec::with_capacity(fused.len());
        let graph = self.graph.lock().unwrap();

        for (chunk_id, fused_score) in fused {
            let chunk = match graph.get_chunk(chunk_id)? {
                Some(c) => c,
                None => continue,
            };

            let doc = match graph.get_document(chunk.doc_id)? {
                Some(d) => d,
                None => continue,
            };

            search_results.push(SearchResult {
                chunk,
                score: fused_score,
                doc_path: doc.path,
            });
        }

        Ok(search_results)
    }

    /// Get all chunks for a specific document
    pub fn get_chunks_for_document(&self, doc_id: Uuid) -> Result<Vec<SearchResult>> {
        let graph = self.graph.lock().unwrap();
        
        let doc = graph
            .get_document(doc_id)?
            .ok_or_else(|| CfsError::Database(format!("Doc {} not found", doc_id)))?;
            
        let chunks = graph.get_chunks_for_doc(doc_id)?;
        
        Ok(chunks.into_iter().map(|c| SearchResult {
            chunk: c,
            score: 1.0, // Browsing doesn't have a score
            doc_path: doc.path.clone(),
        }).collect())
    }

    /// Access the underlying graph store (for testing/debugging)
    pub fn graph(&self) -> Arc<Mutex<cfs_graph::GraphStore>> {
        self.graph.clone()
    }

    /// Generate an answer using the knowledge graph and an LLM
    pub async fn generate_answer(&self, query: &str) -> Result<GenerationResult> {
        let start = std::time::Instant::now();
        info!("Generating answer for: '{}'", query);

        // 1. Search for relevant chunks
        let results = self.search(query, 5)?;

        // 2. Assemble context
        use cfs_core::context_assembler::{ContextAssembler, ScoredChunk};
        let assembler = ContextAssembler::new(2000); // 2000 approx tokens budget
        let scored_chunks: Vec<ScoredChunk> = results
            .iter()
            .map(|r| ScoredChunk {
                chunk: r.chunk.clone(),
                score: r.score,
            })
            .collect();

        let context = assembler.assemble(scored_chunks);

        // 3. Generate answer using configured intelligence engine (no fallback)
        let answer = if let Some(ref engine) = self.intelligence {
            engine.generate(&context, query).await?
        } else {
            return Err(CfsError::NotFound("Intelligence engine not configured".into()));
        };

        Ok(GenerationResult {
            answer,
            context,
            latency_ms: start.elapsed().as_millis() as u64,
        })
    }

    /// Chat with context from the knowledge graph
    pub fn chat(&self, _query: &str, _history: &[Message]) -> Result<String> {
        // This will be expanded later in Phase 2
        Err(CfsError::NotFound("Use generate_answer for Phase 2 initial integration".into()))
    }
}

/// A chat message
#[derive(Debug, Clone)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

/// Chat message role
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
    System,
}

#[cfg(test)]
mod tests {
    use super::*;
    use cfs_core::{Document, Chunk};
    use std::sync::{Arc, Mutex};
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_get_chunks_for_document() {
        let temp = TempDir::new().unwrap();
        let db_path = temp.path().join("test.db");
        let mut graph = cfs_graph::GraphStore::open(db_path.to_str().unwrap()).unwrap();
        
        let doc = Document::new("test.md".into(), b"Hello world", 100);
        graph.insert_document(&doc).unwrap();
        
        let chunk = Chunk {
            id: Uuid::new_v4(),
            doc_id: doc.id,
            text: "Hello world".to_string(),
            offset: 0,
            len: 11,
            seq: 0,
            text_hash: [0; 32],
        };
        graph.insert_chunk(&chunk).unwrap();
        
        let embedder = Arc::new(cfs_embeddings::EmbeddingEngine::new().unwrap());
        let qe = QueryEngine::new(Arc::new(Mutex::new(graph)), embedder);
        
        let results = qe.get_chunks_for_document(doc.id).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].chunk.text, "Hello world");
    }

    #[tokio::test]
    async fn test_hybrid_search() {
        let temp = TempDir::new().unwrap();
        let db_path = temp.path().join("test_hybrid.db");
        let mut graph = cfs_graph::GraphStore::open(db_path.to_str().unwrap()).unwrap();
        
        let doc = Document::new("test.md".into(), b"The quick brown fox jumps over the lazy dog", 100);
        graph.insert_document(&doc).unwrap();
        
        let chunk = Chunk {
            id: Uuid::new_v4(),
            doc_id: doc.id,
            text: "The quick brown fox jumps over the lazy dog".to_string(),
            offset: 0,
            len: 43,
            seq: 0,
            text_hash: [0; 32],
        };
        graph.insert_chunk(&chunk).unwrap();
        
        // Add embedding for semantic search
        let embedder = Arc::new(cfs_embeddings::EmbeddingEngine::new().unwrap());
        let vec = embedder.embed(&chunk.text).unwrap();
        let emb = cfs_core::Embedding::new(chunk.id, &vec, embedder.model_hash());
        graph.insert_embedding(&emb).unwrap();
        
        let qe = QueryEngine::new(Arc::new(Mutex::new(graph)), embedder);
        
        // Test lexical priority
        let results = qe.search("quick brown fox", 5).unwrap();
        assert!(!results.is_empty());
        assert!(results[0].chunk.text.contains("quick brown fox"));
        
        // Test semantic priority (using synonym/related terms)
        let results_sem = qe.search("fast auburn canine", 5).unwrap();
        assert!(!results_sem.is_empty());
        assert!(results_sem[0].chunk.text.contains("quick brown fox"));
    }

    #[tokio::test]
    async fn test_search_comparison_proof() {
        let temp = TempDir::new().unwrap();
        let db_path = temp.path().join("comparison_proof.db");
        let mut graph = cfs_graph::GraphStore::open(db_path.to_str().unwrap()).unwrap();
        
        // Setup Corpus
        let t1 = "The quick brown fox jumps over the lazy dog"; // Keyword: fox
        let t2 = "Artificial intelligence is transforming the modern world"; // Keyword: modern
        let t3 = "A fast auburn canine leaps across an idle hound"; // Keyword: canine, Semantic match for fox
        
        let texts = vec![t1, t2, t3];
        for (i, text) in texts.iter().enumerate() {
            let doc = Document::new(format!("doc_{}.md", i).into(), text.as_bytes(), 100);
            graph.insert_document(&doc).unwrap();
            let chunk = Chunk {
                id: Uuid::new_v4(),
                doc_id: doc.id,
                text: text.to_string(),
                offset: 0,
                len: text.len() as u32,
                seq: 0,
                text_hash: [0; 32],
            };
            graph.insert_chunk(&chunk).unwrap();
            let embedder = Arc::new(cfs_embeddings::EmbeddingEngine::new().unwrap());
            let vec = embedder.embed(text).unwrap();
            let emb = cfs_core::Embedding::new(chunk.id, &vec, embedder.model_hash());
            graph.insert_embedding(&emb).unwrap();
        }
        
        let embedder = Arc::new(cfs_embeddings::EmbeddingEngine::new().unwrap());
        let qe = QueryEngine::new(Arc::new(Mutex::new(graph)), embedder);
        
        // Query: "fox"
        let query = "fox";
        
        // 1. Lexical Results
        let lexical = {
            let graph_lock = qe.graph();
            let g = graph_lock.lock().unwrap();
            g.search_lexical(query, 5).unwrap()
        };
        // Expect result: t1 (direct hit)
        assert_eq!(lexical.len(), 1); 

        // 2. Vector Results
        let vector = {
            let graph_lock = qe.graph();
            let g = graph_lock.lock().unwrap();
            let e = cfs_embeddings::EmbeddingEngine::new().unwrap();
            let q_vec = e.embed(query).unwrap();
            g.search(&q_vec, 5).unwrap()
        };
        // Expect results: t1 and t3 (canine)
        assert!(vector.len() >= 2);

        // 3. Hybrid Results
        let hybrid = qe.search(query, 5).unwrap();
        
        println!("\n--- SEARCH PROOF FOR '{}' ---", query);
        println!("LEXICAL HITS: {}", lexical.len());
        println!("VECTOR HITS:  {}", vector.len());
        println!("HYBRID HITS:  {}", hybrid.len());
        
        // Proof: Hybrid should contain both direct hits and semantic relatives
        let texts_found: Vec<String> = hybrid.iter().map(|r| r.chunk.text.clone()).collect();
        assert!(texts_found.contains(&t1.to_string())); // Direct lexical + semantic
        assert!(texts_found.contains(&t3.to_string())); // Semantic only
    }

    #[tokio::test]
    async fn test_real_corpus_proof() {
        let temp = TempDir::new().unwrap();
        let db_path = temp.path().join("real_corpus.db");
        let mut graph = cfs_graph::GraphStore::open(db_path.to_str().unwrap()).unwrap();
        
        // 1. Ingest actual test_corpus files
        let corpus_dir = std::path::PathBuf::from("/Users/nadeem/dev/CFS/test_corpus");
        let embedder = Arc::new(cfs_embeddings::EmbeddingEngine::new().unwrap());
        
        let files = vec!["zk.md", "ethereum.md", "random.md", "zksnark.md", "lexical_gap.md", "adversarial.md"];
        
        for file_name in files {
            let path = corpus_dir.join(file_name);
            if !path.exists() { continue; }
            
            let content = std::fs::read_to_string(&path).unwrap();
            let doc = Document::new(path.clone(), content.as_bytes(), 0);
            graph.insert_document(&doc).unwrap();
            
            // Simple chunking (for brevity in test)
            let chunk = Chunk {
                id: Uuid::new_v4(),
                doc_id: doc.id,
                text: content.clone(),
                offset: 0,
                len: content.len() as u32,
                seq: 0,
                text_hash: [0; 32],
            };
            graph.insert_chunk(&chunk).unwrap();
            
            let vec = embedder.embed(&content).unwrap();
            let emb = cfs_core::Embedding::new(chunk.id, &vec, embedder.model_hash());
            graph.insert_embedding(&emb).unwrap();
        }
        
        let qe = QueryEngine::new(Arc::new(Mutex::new(graph)), embedder);
        
        // Test 1: Lexical Gap (Strict identifier)
        let q1 = "calculate_hyper_parameter_v7";
        let res1 = qe.search(q1, 1).unwrap();
        println!("\n--- QUERY: '{}' ---", q1);
        for (i, r) in res1.iter().enumerate() {
            println!("Rank {}: [Score: {:.4}] {}", i+1, r.score, r.doc_path.file_name().unwrap().to_string_lossy());
        }
        assert!(res1[0].doc_path.to_string_lossy().contains("lexical_gap.md"));

        // Test 2: Semantic (ZKP concept)
        let q2 = "cryptographic privacy statement validity"; // No "ZKP" or "SNARK" words
        let res2 = qe.search(q2, 3).unwrap();
        println!("\n--- QUERY: '{}' ---", q2);
        for (i, r) in res2.iter().enumerate() {
            println!("Rank {}: [Score: {:.4}] {}", i+1, r.score, r.doc_path.file_name().unwrap().to_string_lossy());
        }
        // Should find zk.md or zksnark.md
        let found_zk = res2.iter().any(|r| r.doc_path.to_string_lossy().contains("zk"));
        assert!(found_zk);

        // Test 3: Hybrid (Privacy blockchain)
        let q3 = "Ethereum privacy zksnark";
        let res3 = qe.search(q3, 5).unwrap();
        println!("\n--- QUERY: '{}' ---", q3);
        for (i, r) in res3.iter().enumerate() {
             println!("Rank {}: [Score: {:.4}] {}", i+1, r.score, r.doc_path.file_name().unwrap().to_string_lossy());
        }
    }
}
