//! CFS Query - RAG query engine
//!
//! Provides semantic search and chat over the knowledge graph.
//!
//! Per CFS-012/CFS-020: Supports filtered retrieval and query caching.

use cfs_core::{Chunk, CfsError, Result, ContextAssembler, ScoredChunk, AssembledContext};
use glob::Pattern;
use lru::LruCache;
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex, RwLock};
use tracing::{info, warn};
use uuid::Uuid;
use std::collections::HashMap;

/// Filter for search queries
///
/// Per CFS-012: Supports filtering by document path, MIME type, and modification time.
#[derive(Debug, Clone)]
pub enum Filter {
    /// Filter by document path glob pattern (e.g., "docs/*.md")
    DocumentPath(String),
    /// Filter by MIME type (e.g., "text/markdown")
    MimeType(String),
    /// Filter by modification time (Unix timestamp, documents modified after this time)
    ModifiedAfter(i64),
    /// Filter by modification time (Unix timestamp, documents modified before this time)
    ModifiedBefore(i64),
}

impl Filter {
    /// Check if a document matches this filter
    pub fn matches(&self, doc: &cfs_core::Document) -> bool {
        match self {
            Filter::DocumentPath(pattern) => {
                if let Ok(glob) = Pattern::new(pattern) {
                    glob.matches(doc.path.to_string_lossy().as_ref())
                } else {
                    false
                }
            }
            Filter::MimeType(mime) => doc.mime_type == *mime,
            Filter::ModifiedAfter(ts) => doc.mtime > *ts,
            Filter::ModifiedBefore(ts) => doc.mtime < *ts,
        }
    }
}

/// Query cache for storing search results
///
/// Per CFS-020: Caches query results keyed by query hash, invalidated on state change.
pub struct QueryCache {
    /// LRU cache: query_hash -> chunk IDs
    cache: RwLock<LruCache<[u8; 32], Vec<Uuid>>>,
    /// State root when cache was last valid
    state_root: RwLock<[u8; 32]>,
}

impl QueryCache {
    /// Create a new query cache with specified capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            cache: RwLock::new(LruCache::new(
                NonZeroUsize::new(capacity).unwrap_or(NonZeroUsize::new(100).unwrap()),
            )),
            state_root: RwLock::new([0u8; 32]),
        }
    }

    /// Get cached results for a query
    pub fn get(&self, query: &str) -> Option<Vec<Uuid>> {
        let hash = Self::hash_query(query);
        self.cache.write().ok()?.get(&hash).cloned()
    }

    /// Store results for a query
    pub fn put(&self, query: &str, results: Vec<Uuid>) {
        let hash = Self::hash_query(query);
        if let Ok(mut cache) = self.cache.write() {
            cache.put(hash, results);
        }
    }

    /// Check if cache is valid for current state root
    pub fn is_valid(&self, current_root: &[u8; 32]) -> bool {
        if let Ok(root) = self.state_root.read() {
            *root == *current_root
        } else {
            false
        }
    }

    /// Invalidate cache and update state root
    pub fn invalidate(&self, new_root: [u8; 32]) {
        if let Ok(mut cache) = self.cache.write() {
            cache.clear();
        }
        if let Ok(mut root) = self.state_root.write() {
            *root = new_root;
        }
    }

    /// Hash a query string
    fn hash_query(query: &str) -> [u8; 32] {
        *blake3::hash(query.as_bytes()).as_bytes()
    }
}

impl Default for QueryCache {
    fn default() -> Self {
        Self::new(100)
    }
}

/// Search result with relevance score
#[derive(Debug, Clone, serde::Serialize)]
pub struct SearchResult {
    /// The matching chunk
    pub chunk: Chunk,
    /// Similarity score (0.0-1.0)
    pub score: f32,
    /// Path to the source document
    pub doc_path: String,
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

/// A citation linking response text to source chunks
///
/// Per CFS-020: Tracks which parts of a response are grounded in context.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Citation {
    /// ID of the source chunk
    pub chunk_id: Uuid,
    /// Byte span in the response (start, end)
    pub span: (usize, usize),
    /// Confidence score (0.0-1.0) based on overlap ratio
    pub confidence: f32,
}

/// Result of response validation
///
/// Per CFS-020: Detects potential hallucinations and measures citation coverage.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ValidationResult {
    /// Whether the response is considered valid (well-grounded)
    pub is_valid: bool,
    /// Warning messages about potential issues
    pub warnings: Vec<String>,
    /// Percentage of response covered by citations (0.0-1.0)
    pub citation_coverage: f32,
    /// Extracted citations
    pub citations: Vec<Citation>,
}

/// Phrases that often indicate hallucination
const HALLUCINATION_PHRASES: &[&str] = &[
    "from my knowledge",
    "i recall that",
    "as far as i know",
    "i believe that",
    "in my experience",
    "typically",
    "generally speaking",
    "it's commonly known",
    "as everyone knows",
    "i think that",
    "probably",
    "most likely",
    "i assume",
    "based on my understanding",
    "from what i've learned",
];

/// Extract citations by finding n-gram overlaps between response and context
///
/// Per CFS-020: Uses 5-gram overlap detection to identify grounded text.
pub fn extract_citations(response: &str, context: &AssembledContext) -> Vec<Citation> {
    let mut citations = Vec::new();
    let response_lower = response.to_lowercase();
    let response_words: Vec<&str> = response_lower.split_whitespace().collect();

    if response_words.len() < 5 {
        return citations;
    }

    for chunk in &context.chunks {
        let chunk_lower = chunk.text.to_lowercase();
        let chunk_words: Vec<&str> = chunk_lower.split_whitespace().collect();

        if chunk_words.len() < 5 {
            continue;
        }

        // Find 5-gram overlaps
        let mut overlap_count = 0;
        let mut matched_positions: Vec<usize> = Vec::new();

        for i in 0..=response_words.len().saturating_sub(5) {
            let response_ngram: Vec<&str> = response_words[i..i + 5].to_vec();

            for j in 0..=chunk_words.len().saturating_sub(5) {
                let chunk_ngram: Vec<&str> = chunk_words[j..j + 5].to_vec();

                if response_ngram == chunk_ngram {
                    overlap_count += 1;
                    matched_positions.push(i);
                    break;
                }
            }
        }

        if overlap_count > 0 {
            // Calculate confidence as ratio of matched n-grams
            let max_ngrams = (response_words.len().saturating_sub(4)).max(1);
            let confidence = (overlap_count as f32) / (max_ngrams as f32);

            // Find byte span from word positions
            let start_pos = matched_positions.first().copied().unwrap_or(0);
            let end_pos = matched_positions.last().copied().unwrap_or(0) + 5;

            // Convert word positions to byte offsets (approximate)
            let mut byte_start = 0;
            let mut byte_end = response.len();

            let mut word_idx = 0;
            for (i, c) in response.char_indices() {
                if c.is_whitespace() {
                    word_idx += 1;
                    if word_idx == start_pos {
                        byte_start = i + 1;
                    }
                    if word_idx == end_pos.min(response_words.len()) {
                        byte_end = i;
                        break;
                    }
                }
            }

            citations.push(Citation {
                chunk_id: chunk.chunk_id,
                span: (byte_start, byte_end),
                confidence,
            });
        }
    }

    // Sort by confidence descending
    citations.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));

    citations
}

/// Validate a response for potential hallucinations
///
/// Per CFS-020: Checks for hallucination phrases and low citation coverage.
pub fn validate_response(response: &str, context: &AssembledContext) -> ValidationResult {
    let mut warnings = Vec::new();

    // Extract citations
    let citations = extract_citations(response, context);

    // Calculate citation coverage
    let total_response_len = response.len() as f32;
    let mut covered_bytes = 0usize;

    for citation in &citations {
        covered_bytes += citation.span.1.saturating_sub(citation.span.0);
    }

    let citation_coverage = if total_response_len > 0.0 {
        (covered_bytes as f32 / total_response_len).min(1.0)
    } else {
        0.0
    };

    // Check for hallucination phrases
    let response_lower = response.to_lowercase();
    for phrase in HALLUCINATION_PHRASES {
        if response_lower.contains(phrase) {
            warnings.push(format!("Response contains hallucination indicator: '{}'", phrase));
        }
    }

    // Check for low citation coverage
    if citation_coverage < 0.3 && !response.is_empty() {
        warnings.push(format!(
            "Low citation coverage: {:.1}% (threshold: 30%)",
            citation_coverage * 100.0
        ));
    }

    // Check if response claims missing information (this is good, not a warning)
    let good_phrases = ["information is missing", "not found in the context", "cannot find"];
    let claims_missing = good_phrases.iter().any(|p| response_lower.contains(p));

    // Determine validity
    let is_valid = warnings.is_empty() || claims_missing;

    ValidationResult {
        is_valid,
        warnings,
        citation_coverage,
        citations,
    }
}

/// Trait for the CFS Intelligence Module (IM)
/// 
/// An IntelligenceEngine is a read-only consumer of the semantic substrate.
/// It synthesizes information from retrieved context into human-readable answers.
#[async_trait::async_trait]
pub trait IntelligenceEngine: Send + Sync {
    /// Generate a synthesized answer from the provided context and query.
    /// This is a read-only operation and cannot mutate the underlying graph.
    async fn generate(&self, context: &AssembledContext, query: &str) -> Result<String>;
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
    async fn generate(&self, context: &AssembledContext, query: &str) -> Result<String> {
        let formatted_context = ContextAssembler::format(context);
        
        // Engineered for consistency with mobile prompt
        let prompt = format!(
            "<|im_start|>system\nYou are a context reading machine. You do not have knowledge of the outside world.\n- Read the Context below carefully.\n- If the answer to the Query is in the Context, output it.\n- If the answer is NOT in the Context, say 'Information is missing from the substrate.' and nothing else.\n- Do not make up facts.\n<|im_end|>\n<|im_start|>user\nContext:\n{}\n\nQuery: {}<|im_end|>\n<|im_start|>assistant\n",
            formatted_context, query
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
    /// Query result cache
    cache: QueryCache,
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
            cache: QueryCache::default(),
        }
    }

    /// Create a new query engine with custom cache capacity
    pub fn with_cache_capacity(
        graph: Arc<Mutex<cfs_graph::GraphStore>>,
        embedder: Arc<cfs_embeddings::EmbeddingEngine>,
        cache_capacity: usize,
    ) -> Self {
        Self {
            graph,
            embedder,
            intelligence: None,
            cache: QueryCache::new(cache_capacity),
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
                doc_path: doc.path.to_string_lossy().to_string(),
            });
        }

        Ok(search_results)
    }

    /// Search with filters applied
    ///
    /// Per CFS-012: Supports filtering by document path, MIME type, and modification time.
    pub fn search_filtered(&self, query: &str, k: usize, filters: &[Filter]) -> Result<Vec<SearchResult>> {
        info!("Filtered search for: '{}' with {} filters", query, filters.len());

        // Get all matching documents based on filters
        let matching_doc_ids: std::collections::HashSet<Uuid> = {
            let graph = self.graph.lock().unwrap();
            let all_docs = graph.get_all_documents()?;

            all_docs
                .into_iter()
                .filter(|doc| filters.iter().all(|f| f.matches(doc)))
                .map(|doc| doc.id)
                .collect()
        };

        if matching_doc_ids.is_empty() {
            info!("No documents match filters");
            return Ok(Vec::new());
        }

        // Perform regular search
        let all_results = self.search(query, k * 3)?; // Get more results to filter

        // Filter results to only include matching documents
        let filtered_results: Vec<SearchResult> = all_results
            .into_iter()
            .filter(|r| matching_doc_ids.contains(&r.chunk.doc_id))
            .take(k)
            .collect();

        info!("Filtered search returned {} results", filtered_results.len());
        Ok(filtered_results)
    }

    /// Search with caching
    ///
    /// Per CFS-020: Uses query cache for faster repeated queries.
    pub fn search_cached(&self, query: &str, k: usize) -> Result<Vec<SearchResult>> {
        // Check cache validity
        let current_root = {
            let graph = self.graph.lock().unwrap();
            graph.compute_merkle_root()?
        };

        if !self.cache.is_valid(&current_root) {
            self.cache.invalidate(current_root);
        }

        // Check cache
        if let Some(chunk_ids) = self.cache.get(query) {
            info!("Cache hit for query: '{}'", query);

            let graph = self.graph.lock().unwrap();
            let mut results = Vec::new();

            for chunk_id in chunk_ids.iter().take(k) {
                if let Some(chunk) = graph.get_chunk(*chunk_id)? {
                    if let Some(doc) = graph.get_document(chunk.doc_id)? {
                        results.push(SearchResult {
                            chunk,
                            score: 1.0, // Score not preserved in cache
                            doc_path: doc.path.to_string_lossy().to_string(),
                        });
                    }
                }
            }

            return Ok(results);
        }

        // Cache miss - perform search
        let results = self.search(query, k)?;

        // Store in cache
        let chunk_ids: Vec<Uuid> = results.iter().map(|r| r.chunk.id).collect();
        self.cache.put(query, chunk_ids);

        Ok(results)
    }

    /// Invalidate the query cache
    pub fn invalidate_cache(&self) -> Result<()> {
        let root = {
            let graph = self.graph.lock().unwrap();
            graph.compute_merkle_root()?
        };
        self.cache.invalidate(root);
        Ok(())
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
            doc_path: doc.path.to_string_lossy().to_string(),
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
        let assembler = ContextAssembler::with_budget(2000); 
        let scored_chunks: Vec<ScoredChunk> = results
            .iter()
            .map(|r| ScoredChunk {
                chunk: r.chunk.clone(),
                score: r.score,
                document_path: r.doc_path.clone(),
            })
            .collect();

        let state_root = {
            let graph = self.graph.lock().unwrap();
            graph.compute_merkle_root()?
        };
        
        let assembled_context = assembler.assemble(scored_chunks, query, state_root);

        // 3. Generate answer using configured intelligence engine (no fallback)
        let answer = if let Some(ref engine) = self.intelligence {
            engine.generate(&assembled_context, query).await?
        } else {
            return Err(CfsError::NotFound("Intelligence engine not configured".into()));
        };

        Ok(GenerationResult {
            answer,
            context: ContextAssembler::format(&assembled_context),
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
            byte_offset: 0,
            byte_length: 11,
            sequence: 0,
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
            byte_offset: 0,
            byte_length: 43,
            sequence: 0,
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
                byte_offset: 0,
                byte_length: text.len() as u64,
                sequence: 0,
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

    #[test]
    fn test_filter_by_mime_type() {
        use cfs_core::Document;
        use std::path::PathBuf;

        let doc_md = Document::new(PathBuf::from("test.md"), b"content", 1000);
        let doc_pdf = Document::new(PathBuf::from("test.pdf"), b"pdf content", 1000);

        let filter = Filter::MimeType("text/markdown".to_string());

        assert!(filter.matches(&doc_md));
        assert!(!filter.matches(&doc_pdf));
    }

    #[test]
    fn test_filter_by_path_glob() {
        use cfs_core::Document;
        use std::path::PathBuf;

        let doc1 = Document::new(PathBuf::from("docs/readme.md"), b"content", 1000);
        let doc2 = Document::new(PathBuf::from("src/main.rs"), b"code", 1000);

        let filter = Filter::DocumentPath("docs/*.md".to_string());

        assert!(filter.matches(&doc1));
        assert!(!filter.matches(&doc2));
    }

    #[test]
    fn test_filter_by_modified_time() {
        use cfs_core::Document;
        use std::path::PathBuf;

        let old_doc = Document::new(PathBuf::from("old.md"), b"content", 1000);
        let new_doc = Document::new(PathBuf::from("new.md"), b"content", 2000);

        let filter_after = Filter::ModifiedAfter(1500);
        let filter_before = Filter::ModifiedBefore(1500);

        assert!(!filter_after.matches(&old_doc));
        assert!(filter_after.matches(&new_doc));

        assert!(filter_before.matches(&old_doc));
        assert!(!filter_before.matches(&new_doc));
    }

    #[test]
    fn test_citation_extraction() {
        use cfs_core::{ContextChunk, ContextMetadata};

        let context = AssembledContext {
            chunks: vec![ContextChunk {
                chunk_id: Uuid::new_v4(),
                document_path: "test.md".to_string(),
                text: "The quick brown fox jumps over the lazy dog".to_string(),
                score: 1.0,
                sequence: 0,
            }],
            total_tokens: 10,
            truncated: false,
            metadata: ContextMetadata {
                query_hash: [0u8; 32],
                state_root: [0u8; 32],
            },
        };

        // Response that contains text from context
        let response = "As mentioned, the quick brown fox jumps over the lazy dog in the story.";
        let citations = extract_citations(response, &context);

        assert!(!citations.is_empty(), "Should find citations for overlapping text");
        assert!(citations[0].confidence > 0.0);
    }

    #[test]
    fn test_hallucination_detection() {
        use cfs_core::{ContextChunk, ContextMetadata};

        let context = AssembledContext {
            chunks: vec![ContextChunk {
                chunk_id: Uuid::new_v4(),
                document_path: "test.md".to_string(),
                text: "The capital of France is Paris".to_string(),
                score: 1.0,
                sequence: 0,
            }],
            total_tokens: 10,
            truncated: false,
            metadata: ContextMetadata {
                query_hash: [0u8; 32],
                state_root: [0u8; 32],
            },
        };

        // Response with hallucination phrase
        let bad_response = "From my knowledge, I believe that Paris is a beautiful city.";
        let result = validate_response(bad_response, &context);

        assert!(!result.warnings.is_empty(), "Should detect hallucination phrases");
        assert!(result.warnings.iter().any(|w| w.contains("hallucination")));

        // Good response that admits missing info
        let good_response = "Information is missing from the substrate.";
        let result2 = validate_response(good_response, &context);

        assert!(result2.is_valid, "Should be valid when admitting missing info");
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
                byte_offset: 0,
                byte_length: content.len() as u64,
                sequence: 0,
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
            println!("Rank {}: [Score: {:.4}] {}", i+1, r.score, r.doc_path);
        }
        assert!(res1[0].doc_path.contains("lexical_gap.md"));

        // Test 2: Semantic (ZKP concept)
        let q2 = "cryptographic privacy statement validity"; // No "ZKP" or "SNARK" words
        let res2 = qe.search(q2, 3).unwrap();
        println!("\n--- QUERY: '{}' ---", q2);
        for (i, r) in res2.iter().enumerate() {
            println!("Rank {}: [Score: {:.4}] {}", i+1, r.score, r.doc_path);
        }
        // Should find zk.md or zksnark.md
        let found_zk = res2.iter().any(|r| r.doc_path.contains("zk"));
        assert!(found_zk);

        // Test 3: Hybrid (Privacy blockchain)
        let q3 = "Ethereum privacy zksnark";
        let res3 = qe.search(q3, 5).unwrap();
        println!("\n--- QUERY: '{}' ---", q3);
        for (i, r) in res3.iter().enumerate() {
             println!("Rank {}: [Score: {:.4}] {}", i+1, r.score, r.doc_path);
        }
    }
}
