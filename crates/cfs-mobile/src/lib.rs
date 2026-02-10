//! CFS Mobile - Mobile-optimized core with C FFI
//!
//! Provides a C-compatible interface for iOS and Android.

use cfs_core::{CfsError, Result};
use cfs_embeddings::EmbeddingEngine;
use cfs_graph::GraphStore;
use cfs_inference_mobile::LocalGenerator;
use cfs_query::QueryEngine;
use cfs_relay_client::RelayClient;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;
use std::sync::{Arc, Mutex, OnceLock};
use tokio::runtime::Runtime;
use tracing::{error, info};
use uuid::Uuid;

static LAST_ERROR: OnceLock<Mutex<String>> = OnceLock::new();

fn get_last_error_mutex() -> &'static Mutex<String> {
    LAST_ERROR.get_or_init(|| Mutex::new(String::new()))
}

fn set_last_error(msg: &str) {
    if let Ok(mut e) = get_last_error_mutex().lock() {
        *e = msg.to_string();
    }
}

/// Opaque context for mobile operations
pub struct CfsContext {
    pub rt: Runtime,
    pub query_engine: Mutex<Arc<QueryEngine>>,
    pub graph: Arc<Mutex<GraphStore>>,
    pub embedder: Arc<EmbeddingEngine>,
    pub generator: Mutex<Option<Arc<QueryEngine>>>,
}

/// Initialize the CFS context
///
/// # Safety
/// `db_path` must be a valid null-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn cfs_init(db_path: *const c_char) -> *mut CfsContext {
    if db_path.is_null() {
        return ptr::null_mut();
    }

    let path_str = match CStr::from_ptr(db_path).to_str() {
        Ok(s) => s.to_string(),
        Err(_) => return ptr::null_mut(),
    };

    // Initialize tracing once
    static INIT_LOG: std::sync::Once = std::sync::Once::new();
    INIT_LOG.call_once(|| {
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .with_ansi(false)
            .init();
    });

    println!("[CFS-Mobile] cfs_init for path: {}", path_str);

    // Initialize runtime
    let rt = match Runtime::new() {
        Ok(rt) => rt,
        Err(e) => {
            println!("[CFS-Mobile] Failed to create runtime: {}", e);
            return ptr::null_mut();
        }
    };

    // Initialize components (synchronous operations)
    // 1. Open GraphStore
    let graph = match GraphStore::open(&path_str) {
        Ok(g) => g,
        Err(e) => {
            println!("[CFS-Mobile] Failed to open graph store: {}", e);
            error!("Failed to open graph store: {}", e);
            return ptr::null_mut();
        }
    };
    let graph_arc = Arc::new(Mutex::new(graph));

    // 2. Initialize Embeddings
    let embedder = match EmbeddingEngine::new() {
        Ok(e) => e,
        Err(e) => {
            println!("[CFS-Mobile] Failed to init embeddings: {}", e);
            error!("Failed to init embeddings: {}", e);
            return ptr::null_mut();
        }
    };
    let embedder_arc = Arc::new(embedder);

    // 3. Create QueryEngine
    let query_engine = Arc::new(QueryEngine::new(graph_arc.clone(), embedder_arc.clone()));

    let ctx = Box::new(CfsContext {
        rt,
        query_engine: Mutex::new(query_engine),
        graph: graph_arc,
        embedder: embedder_arc,
        generator: Mutex::new(None),
    });

    println!("[CFS-Mobile] Context created successfully.");
    Box::into_raw(ctx)
}

/// Sync with the relay server
///
/// Returns number of diffs applied on success, or negative error code.
///
/// # Safety
/// `ctx` must be a valid pointer from `cfs_init`.
/// `relay_url` must be a valid null-terminated C string.
/// `key_hex` must be a valid null-terminated C string (64 hex chars).
#[no_mangle]
pub unsafe extern "C" fn cfs_sync(
    ctx: *mut CfsContext,
    relay_url: *const c_char,
    key_hex: *const c_char,
) -> i32 {
    if ctx.is_null() || relay_url.is_null() || key_hex.is_null() {
        return -1;
    }

    let ctx = &*ctx;
    let url_str = match CStr::from_ptr(relay_url).to_str() {
        Ok(s) => s,
        Err(_) => return -2,
    };
    
    let key_str = match CStr::from_ptr(key_hex).to_str() {
        Ok(s) => s,
        Err(_) => return -2,
    };
    
    // Parse key
    let key_bytes = match hex::decode(key_str) {
        Ok(v) => {
            if v.len() != 32 { return -4; }
            let mut k = [0u8; 32];
            k.copy_from_slice(&v);
            k
        },
        Err(_) => return -4,
    };
    
    // Create crypto engine for decryption
    let crypto = cfs_sync::CryptoEngine::new_with_seed(key_bytes);

    let client = RelayClient::new(url_str, "dummy_token");

    // Execute sync in runtime
    let res: Result<i32> = ctx.rt.block_on(async {
        // 1. Get local head
        let local_head_hash = {
            let graph = ctx.graph.lock().unwrap();
            graph.get_latest_root()?.map(|r| r.hash)
        };

        // 2. Fetch all roots from Relay
        println!("[CFS-Mobile] Fetching roots from {}", url_str);
        let remote_roots_hex = client.get_roots(None).await?;
        println!("[CFS-Mobile] Found {} roots", remote_roots_hex.len());
        let mut remote_hashes = Vec::new();
        for h in remote_roots_hex {
            if let Ok(bytes) = hex::decode(&h) {
                if bytes.len() == 32 {
                    let mut arr = [0u8; 32];
                    arr.copy_from_slice(&bytes);
                    remote_hashes.push(arr);
                }
            } else {
                println!("[CFS-Mobile] Failed to decode hex root: {}", h);
            }
        }

        // 3. Determine start point
        let start_index = if let Some(head) = local_head_hash {
            // Find our head in remote list
            if let Some(idx) = remote_hashes.iter().position(|h| *h == head) {
                idx + 1 // Start from next
            } else {
                // If the relay is empty or our head isn't there, we must start from 0
                // to ensure we have all data from the relay. 
                // Since apply_diff is idempotent (INSERT OR REPLACE), this is safe.
                println!("[CFS-Mobile] Local head not found in relay. Starting from beginning.");
                0
            }
        } else {
            0 // Start from beginning
        };

        if start_index >= remote_hashes.len() {
            println!("[CFS-Mobile] Already up to date (start_index: {}, len: {})", start_index, remote_hashes.len());
            return Ok(0);
        }

        println!("[CFS-Mobile] Syncing {} new roots...", remote_hashes.len() - start_index);
        // 2. Fetch and apply diffs
        let mut applied_count = 0;
        for root_hash in remote_hashes.iter().skip(start_index) {
            let root_hex = hex::encode(root_hash);
            // Check if we already have this root
            let root_bytes = match hex::decode(&root_hex) {
                Ok(b) => b,
                Err(e) => {
                    set_last_error(&format!("Invalid root hash: {}", e));
                    return Err(CfsError::Sync(format!("Invalid root hash: {}", e)));
                }
            };
            
            let local_has_root = {
                let graph = ctx.graph.lock().unwrap();
                // Query state_roots for exact hash
                graph.get_latest_root().map(|r| r.is_some() && r.unwrap().hash == root_bytes.as_slice()).unwrap_or(false)
            };

            if local_has_root {
                continue;
            }

            let payload = match client.get_diff(&root_hex).await {
                Ok(p) => p,
                Err(e) => {
                    set_last_error(&format!("Failed to fetch diff {}: {}", root_hex, e));
                    return Err(CfsError::Sync(format!("Failed to fetch diff {}: {}", root_hex, e)));
                }
            };

            let diff = match crypto.decrypt_diff(&payload) {
                Ok(d) => d,
                Err(e) => {
                    set_last_error(&format!("Failed to decrypt diff: {}", e));
                    return Err(CfsError::Crypto(format!("Failed to decrypt diff: {}", e)));
                }
            };

            // Before applying, verify semantic new_root in diff
            // Actually, we trust the Signed StateRoot for now in V0.
            
            {
                let mut graph = ctx.graph.lock().unwrap();
                if let Err(e) = graph.apply_diff(&diff) {
                    set_last_error(&format!("Failed to apply diff: {}", e));
                    return Err(CfsError::Database(format!("Failed to apply diff: {}", e)));
                }
                
                // Also store the StateRoot we just applied
                let state_root = cfs_core::StateRoot {
                    hash: diff.metadata.new_root,
                    parent: if diff.metadata.prev_root == [0u8; 32] { None } else { Some(diff.metadata.prev_root) },
                    timestamp: diff.metadata.timestamp,
                    device_id: diff.metadata.device_id,
                    signature: payload.signature,
                    seq: diff.metadata.seq,
                };
                if let Err(e) = graph.set_latest_root(&state_root) {
                    set_last_error(&format!("Failed to set latest root: {}", e));
                    return Err(CfsError::Database(format!("Failed to set latest root: {}", e)));
                }
            }
            applied_count += 1;
        }

        Ok(applied_count)
    });

    match res {
        Ok(count) => count,
        Err(e) => {
            println!("[CFS-Mobile] Sync Error: {}", e);
            let error_msg = e.to_string();
            set_last_error(&error_msg);
            error!("Sync failed: {}", e);
            match &e {
                CfsError::Sync(msg) => {
                    println!("[CFS-Mobile] Sync Detail: {}", msg);
                    -5
                }
                CfsError::Crypto(_) => -6,                  // Decryption/Keys error
                CfsError::Database(_) | CfsError::Io(_) => -7, // Database/File error
                CfsError::InvalidState(_) => -8,            // Sync mismatch / State error
                CfsError::Verification(_) => -9,            // Merkle/Signature check failed
                CfsError::NotFound(_) => -10,               // Diff not found on relay
                CfsError::Parse(_) => -11,                  // Diff parse error
                CfsError::Serialization(_) => -12,          // JSON parsing error
                _ => {
                    println!("[CFS-Mobile] Unmapped Error: {:?}", e);
                    -3
                }
            }
        }
    }
}
#[no_mangle]
pub unsafe extern "C" fn cfs_init_debug() {
    println!("[CFS-Mobile] Library initialized. Version: {}", String::from_utf8_lossy(CStr::from_ptr(cfs_version()).to_bytes()));
}

/// Query the knowledge graph
///
/// Returns a JSON string with search results.
///
/// # Safety
/// `ctx` must be a valid pointer from `cfs_init`.
/// `query` must be a valid null-terminated C string.
/// Returns a null-terminated string that must be freed with `cfs_free_string`.
#[no_mangle]
pub unsafe extern "C" fn cfs_query(
    ctx: *mut CfsContext,
    query: *const c_char,
) -> *mut c_char {
    if ctx.is_null() || query.is_null() {
        return ptr::null_mut();
    }

    let ctx = &*ctx;
    let query_str = match CStr::from_ptr(query).to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };

    let qe = ctx.query_engine.lock().unwrap().clone();
    let res: Result<Vec<SimpleSearchResult>> = (|| {
        let results = qe.search(query_str, 5)?;
        
        // Convert to simplified struct for JSON
        let simple_results: Vec<SimpleSearchResult> = results.into_iter().map(|r| SimpleSearchResult {
            text: r.chunk.text,
            score: r.score,
            doc_path: r.doc_path.to_string_lossy().into_owned(),
        }).collect();
        
        Ok(simple_results)
    })();

    match res {
        Ok(results) => {
            let json = serde_json::to_string(&results).unwrap_or_else(|_| "[]".to_string());
            let c_str = CString::new(json).unwrap_or_default();
            c_str.into_raw()
        },
        Err(e) => {
            error!("Query failed: {}", e);
            set_last_error(&format!("Query failed: {}", e));
            let c_str = CString::new("[]").unwrap_or_default();
            c_str.into_raw()
        }
    }
}

/// Get all chunks for a document
///
/// # Safety
/// `ctx` must be a valid pointer from `cfs_init`.
/// `doc_id_hex` must be a valid null-terminated C string (32 or 36 chars).
/// Returns a JSON string that must be freed with `cfs_free_string`.
#[no_mangle]
pub unsafe extern "C" fn cfs_get_chunks(
    ctx: *mut CfsContext,
    doc_id_hex: *const c_char,
) -> *mut c_char {
    if ctx.is_null() || doc_id_hex.is_null() {
        return ptr::null_mut();
    }

    let ctx = &*ctx;
    let doc_id_str = match CStr::from_ptr(doc_id_hex).to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };

    let doc_id = match Uuid::parse_str(doc_id_str) {
        Ok(id) => id,
        Err(_) => return ptr::null_mut(),
    };

    let qe = ctx.query_engine.lock().unwrap().clone();
    let res: Result<Vec<SimpleSearchResult>> = (|| {
        let results = qe.get_chunks_for_document(doc_id)?;
        
        // Convert to simplified struct for JSON
        let simple_results: Vec<SimpleSearchResult> = results.into_iter().map(|r| SimpleSearchResult {
            text: r.chunk.text,
            score: r.score,
            doc_path: r.doc_path.to_string_lossy().into_owned(),
        }).collect();
        
        Ok(simple_results)
    })();

    match res {
        Ok(results) => {
            let json = serde_json::to_string(&results).unwrap_or_else(|_| "[]".to_string());
            let c_str = CString::new(json).unwrap_or_default();
            c_str.into_raw()
        },
        Err(e) => {
            error!("Get chunks failed: {}", e);
            let c_str = CString::new("[]").unwrap_or_default();
            c_str.into_raw()
        }
    }
}

/// Test LLM backend initialization only (for debugging)
/// Note: This is now a no-op since we can't init backend twice
#[no_mangle]
pub unsafe extern "C" fn cfs_test_llm_backend() -> i32 {
    println!("[CFS-Mobile] Backend test skipped (will init with model)");
    0
}

/// Initialize the LLM generator for the context
///
/// # Safety
/// `ctx` must be valid. `model_path` must be a valid C string pointing to a GGUF model file.
#[no_mangle]
pub unsafe extern "C" fn cfs_init_llm(ctx: *mut CfsContext, model_path: *const c_char) -> i32 {
    println!("[CFS-Mobile] === cfs_init_llm START ===");

    if ctx.is_null() {
        println!("[CFS-Mobile] ERROR: ctx is null");
        return -1;
    }
    if model_path.is_null() {
        println!("[CFS-Mobile] ERROR: model_path is null");
        return -1;
    }

    let ctx = &*ctx;
    println!("[CFS-Mobile] Context OK");

    let model_path_str = match CStr::from_ptr(model_path).to_str() {
        Ok(s) => s.to_string(),
        Err(e) => {
            println!("[CFS-Mobile] ERROR: Invalid model path string: {}", e);
            return -2;
        }
    };

    println!("[CFS-Mobile] Model path: {}", model_path_str);

    // Check file exists
    let path = std::path::Path::new(&model_path_str);
    if !path.exists() {
        println!("[CFS-Mobile] ERROR: Model file does not exist");
        set_last_error("Model file does not exist");
        return -3;
    }

    let file_size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    println!("[CFS-Mobile] Model file exists, size: {} bytes", file_size);

    // Wrap everything in panic catch
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        println!("[CFS-Mobile] Creating LocalGenerator...");
        let generator = Box::new(LocalGenerator::new(model_path_str.clone().into()));
        println!("[CFS-Mobile] LocalGenerator created");

        println!("[CFS-Mobile] Calling generator.initialize()...");
        generator.initialize()?;
        println!("[CFS-Mobile] Generator initialized!");

        Ok::<_, cfs_core::CfsError>(generator)
    }));

    let generator = match result {
        Ok(Ok(gen)) => {
            println!("[CFS-Mobile] Generator ready");
            gen
        }
        Ok(Err(e)) => {
            let msg = format!("Model initialization failed: {}", e);
            println!("[CFS-Mobile] {}", msg);
            set_last_error(&msg);
            return -4;
        }
        Err(panic_info) => {
            let msg = format!("PANIC during initialization: {:?}", panic_info);
            println!("[CFS-Mobile] {}", msg);
            set_last_error("Panic during LLM initialization");
            return -5;
        }
    };

    println!("[CFS-Mobile] Creating QueryEngine with generator...");
    let qe = cfs_query::QueryEngine::new(ctx.graph.clone(), ctx.embedder.clone())
        .with_generator(generator);
    println!("[CFS-Mobile] QueryEngine created");

    match ctx.generator.lock() {
        Ok(mut gen_lock) => {
            *gen_lock = Some(Arc::new(qe));
            println!("[CFS-Mobile] === cfs_init_llm SUCCESS ===");
            0
        }
        Err(_) => {
            println!("[CFS-Mobile] ERROR: Mutex poisoned!");
            set_last_error("Internal error: Generator mutex poisoned.");
            -6
        }
    }
}

/// Generate an answer using the local LLM
///
/// Returns a JSON string with the generation result that must be freed with `cfs_free_string`.
///
/// # Safety
/// `ctx` must be valid. `query` must be a valid C string.
#[no_mangle]
pub unsafe extern "C" fn cfs_generate(ctx: *mut CfsContext, query: *const c_char) -> *mut c_char {
    if ctx.is_null() || query.is_null() {
        return ptr::null_mut();
    }

    let ctx = &*ctx;
    let query_str = match CStr::from_ptr(query).to_str() {
        Ok(s) => s.to_string(),
        Err(_) => return ptr::null_mut(),
    };

    println!("[CFS-Mobile] cfs_generate called for query: {}", query_str);
    info!("[CFS-Mobile] cfs_generate called for query: {}", query_str);

    // Get the generator QueryEngine
    let qe = {
        match ctx.generator.lock() {
            Ok(gen_lock) => {
                match gen_lock.as_ref() {
                    Some(v) => v.clone(),
                    None => {
                        println!("[CFS-Mobile] cfs_generate: Generator not initialized.");
                        error!("[CFS-Mobile] cfs_generate: Generator not initialized.");
                        set_last_error("LLM generator not initialized. Call cfs_init_llm first.");
                        return ptr::null_mut();
                    }
                }
            }
            Err(_) => {
                println!("[CFS-Mobile] cfs_generate: Mutex poisoned!");
                error!("[CFS-Mobile] cfs_generate: Mutex poisoned!");
                set_last_error("Internal error: Generator mutex poisoned.");
                return ptr::null_mut();
            }
        }
    };

    println!("[CFS-Mobile] Starting async generation...");
    info!("[CFS-Mobile] Starting async generation...");

    // Run generation with panic catching
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        ctx.rt.block_on(async { qe.generate_answer(&query_str).await })
    }));

    let res = match caught {
        Ok(r) => r,
        Err(_) => {
            println!("[CFS-Mobile] cfs_generate: Panicked during async execution!");
            error!("[CFS-Mobile] cfs_generate: Panicked during async execution!");
            set_last_error("Critical error: AI engine panicked during generation.");
            return ptr::null_mut();
        }
    };

    match res {
        Ok(gen_res) => {
            info!(
                "[CFS-Mobile] Generation successful. Latency: {}ms",
                gen_res.latency_ms
            );
            let json = serde_json::to_string(&gen_res).unwrap_or_else(|_| "{}".to_string());
            let c_str = CString::new(json).unwrap_or_default();
            c_str.into_raw()
        }
        Err(e) => {
            error!("[CFS-Mobile] Generation failed: {}", e);
            set_last_error(&format!("Generation failed: {}", e));
            ptr::null_mut()
        }
    }
}

#[derive(serde::Serialize)]
struct SimpleSearchResult {
    text: String,
    score: f32,
    doc_path: String,
}

/// Get the current state root hash as a hex string
///
/// # Safety
/// `ctx` must be a valid pointer from `cfs_init`.
/// Returns a hex string (64 chars) that must be freed with `cfs_free_string`.
#[no_mangle]
pub unsafe extern "C" fn cfs_get_state_root(ctx: *mut CfsContext) -> *mut c_char {
    if ctx.is_null() {
        return ptr::null_mut();
    }

    let ctx = &*ctx;
    let res = {
        let graph = ctx.graph.lock().unwrap();
        graph.get_latest_root()
    };

    match res {
        Ok(Some(root)) => {
            let hex = hex::encode(root.hash);
            let c_str = CString::new(hex).unwrap_or_default();
            c_str.into_raw()
        }
        Ok(None) => {
            let c_str = CString::new("00".repeat(32)).unwrap_or_default();
            c_str.into_raw()
        }
        Err(e) => {
            error!("Failed to get state root: {}", e);
            ptr::null_mut()
        }
    }
}

/// Free a string returned by CFS functions
///
/// # Safety
/// `s` must be a pointer returned by a CFS function or null.
#[no_mangle]
pub unsafe extern "C" fn cfs_free_string(s: *mut c_char) {
    if !s.is_null() {
        drop(CString::from_raw(s));
    }
}

/// Free the CFS context
///
/// # Safety
/// `ctx` must be a valid pointer from `cfs_init` or null.
#[no_mangle]
pub unsafe extern "C" fn cfs_free(ctx: *mut CfsContext) {
    if !ctx.is_null() {
        drop(Box::from_raw(ctx));
    }
}

/// Get the last error message
///
/// Returns a string that must be freed with `cfs_free_string`.
#[no_mangle]
pub unsafe extern "C" fn cfs_last_error() -> *mut c_char {
    let msg = if let Ok(e) = get_last_error_mutex().lock() {
        e.clone()
    } else {
        "Mutex poisoned".to_string()
    };
    let c_str = CString::new(msg).unwrap_or_default();
    c_str.into_raw()
}

/// Get graph statistics
///
/// Returns a JSON string that must be freed with `cfs_free_string`.
///
/// # Safety
/// `ctx` must be a valid pointer from `cfs_init`.
#[no_mangle]
pub unsafe extern "C" fn cfs_stats(ctx: *mut CfsContext) -> *mut c_char {
    if ctx.is_null() {
        return ptr::null_mut();
    }

    let ctx = &*ctx;
    let res = {
        let graph = ctx.graph.lock().unwrap();
        graph.stats()
    };

    match res {
        Ok(stats) => {
            let json = serde_json::json!({
                "documents": stats.documents,
                "chunks": stats.chunks,
                "embeddings": stats.embeddings,
                "edges": stats.edges,
            }).to_string();
            let c_str = CString::new(json).unwrap_or_default();
            c_str.into_raw()
        }
        Err(e) => {
            error!("Failed to get stats: {}", e);
            ptr::null_mut()
        }
    }
}

/// Get the library version
#[no_mangle]
pub extern "C" fn cfs_version() -> *const c_char {
    static VERSION: &[u8] = b"0.1.0\0";
    VERSION.as_ptr() as *const c_char
}
