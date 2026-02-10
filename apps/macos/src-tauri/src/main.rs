use tauri::State;
use std::sync::{Arc, Mutex};
use std::path::PathBuf;
use cfs_desktop::DesktopApp;
use cfs_query::QueryEngine;
use cfs_embeddings::EmbeddingEngine;
use cfs_core::Result;
use serde::Serialize;
use uuid::Uuid;

struct AppState {
    app: Mutex<Option<DesktopApp>>,
    data_dir: PathBuf,
}

#[derive(Serialize)]
struct Stats {
    state_root: String,
    doc_count: usize,
    chunk_count: usize,
    status: String,
}

#[derive(Serialize)]
struct UiSearchResult {
    text: String,
    score: f32,
    path: String,
}

#[derive(Serialize)]
struct UiDocument {
    id: String,
    path: String,
    hash: String,
}

#[tauri::command]
async fn add_watch_dir(state: State<'_, AppState>, path: String) -> Result<String> {
    let mut app_lock = state.app.lock().unwrap();
    if app_lock.is_none() {
        let app = DesktopApp::new(state.data_dir.clone())?;
        *app_lock = Some(app);
    }
    
    let app = app_lock.as_mut().unwrap();
    let path_buf = PathBuf::from(path);
    app.add_watch_dir(path_buf.clone())?;
    
    // Spawn ingestion loop in background
    let app_clone = app.clone();
    tokio::spawn(async move {
        if let Err(e) = app_clone.start().await {
            tracing::error!("Ingestion loop failed: {}", e);
        }
    });
    
    Ok(format!("Watching {:?}", path_buf))
}

#[tauri::command]
async fn get_stats(state: State<'_, AppState>) -> Result<Stats> {
    let app_lock = state.app.lock().unwrap();
    if let Some(app) = app_lock.as_ref() {
        let graph = app.graph();
        let lock = graph.lock().unwrap();
        let root = lock.get_latest_root()?.map(|r| hex::encode(r.hash)).unwrap_or_else(|| "00".repeat(32));
        let stats = lock.stats()?;
        
        Ok(Stats {
            state_root: root,
            doc_count: stats.documents,
            chunk_count: stats.chunks,
            status: app.status(),
        })
    } else {
        Ok(Stats {
            state_root: "Not Initialized".into(),
            doc_count: 0,
            chunk_count: 0,
            status: "Offline".into(),
        })
    }
}

#[tauri::command]
async fn perform_query(state: State<'_, AppState>, text: String) -> Result<Vec<UiSearchResult>> {
    let app_lock = state.app.lock().unwrap();
    if let Some(app) = app_lock.as_ref() {
        let graph = app.graph();
        let embedder = Arc::new(EmbeddingEngine::new()?);
        let qe = QueryEngine::new(graph, embedder);
        
        let results = qe.search(&text, 5)?;
        Ok(results.into_iter().map(|r| UiSearchResult {
            text: r.chunk.text,
            score: r.score,
            path: r.doc_path.to_string_lossy().into_owned(),
        }).collect())
    } else {
        Ok(vec![])
    }
}

#[tauri::command]
async fn generate_answer(state: State<'_, AppState>, text: String) -> Result<cfs_query::GenerationResult> {
    let qe = {
        let app_lock = state.app.lock().unwrap();
        if let Some(app) = app_lock.as_ref() {
            let graph = app.graph();
            let embedder = Arc::new(EmbeddingEngine::new()?);
            // Use Ollama as the generator for desktop
            let generator = Box::new(cfs_query::OllamaGenerator::new(
                "http://localhost:11434".into(),
                "mistral".into(),
            ));
            QueryEngine::new(graph, embedder).with_generator(generator)
        } else {
            return Err(cfs_core::CfsError::NotFound("App not initialized".into()));
        }
    }; // app_lock is dropped here

    qe.generate_answer(&text).await
}

#[tauri::command]
async fn trigger_sync(_state: State<'_, AppState>) -> Result<String> {
    // In V0, sync happens in background. Here we just confirm.
    Ok("Sync (Push) is active in background".into())
}

#[tauri::command]
async fn reset_data(state: State<'_, AppState>) -> Result<String> {
    let mut app_lock = state.app.lock().unwrap();
    *app_lock = None;
    
    // Delete data dir
    if state.data_dir.exists() {
        std::fs::remove_dir_all(&state.data_dir).map_err(|e| cfs_core::CfsError::Io(e))?;
    }
    
    Ok("Data directory deleted. App reset.".into())
}

#[tauri::command]
async fn list_documents(state: State<'_, AppState>) -> Result<Vec<UiDocument>> {
    let app_lock = state.app.lock().unwrap();
    if let Some(app) = app_lock.as_ref() {
        let graph = app.graph();
        let lock = graph.lock().unwrap();
        let docs = lock.get_all_documents()?;
        
        Ok(docs.into_iter().map(|d| UiDocument {
            id: d.id.to_string(),
            path: d.path.to_string_lossy().into_owned(),
            hash: hex::encode(d.hash),
        }).collect())
    } else {
        Ok(vec![])
    }
}

#[tauri::command]
async fn get_document_chunks(state: State<'_, AppState>, doc_id: String) -> Result<Vec<UiSearchResult>> {
    let app_lock = state.app.lock().unwrap();
    if let Some(app) = app_lock.as_ref() {
        let graph = app.graph();
        let embedder = Arc::new(EmbeddingEngine::new()?);
        let qe = QueryEngine::new(graph, embedder);
        
        let id = Uuid::parse_str(&doc_id).map_err(|e: uuid::Error| cfs_core::CfsError::Parse(e.to_string()))?;
        let results = qe.get_chunks_for_document(id)?;
        Ok(results.into_iter().map(|r| UiSearchResult {
            text: r.chunk.text,
            score: r.score,
            path: r.doc_path.to_string_lossy().into_owned(),
        }).collect())
    } else {
        Ok(vec![])
    }
}

fn main() {
    let data_dir = PathBuf::from("./.cfs");
    
    tauri::Builder::default()
        .manage(AppState {
            app: Mutex::new(None),
            data_dir,
        })
        .invoke_handler(tauri::generate_handler![
            add_watch_dir,
            get_stats,
            perform_query,
            generate_answer,
            trigger_sync,
            reset_data,
            list_documents,
            get_document_chunks
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

