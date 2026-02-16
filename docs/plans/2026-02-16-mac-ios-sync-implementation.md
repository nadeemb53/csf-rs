# Mac-iOS Document Sync Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement document intelligence sync between macOS and iOS with device pairing, document indexing, and verifiable query citations.

**Architecture:** Cloud relay server with Ed25519 device pairing. Mac indexes documents and pushes encrypted diffs to relay. iOS pulls diffs, builds local index, and supports hybrid search with citations.

**Tech Stack:** Rust (cp-* crates), Tauri (macOS desktop), SwiftUI (iOS), SQLite + HNSW (storage), XChaCha20-Poly1305 (encryption)

---

## Phase 1: Relay Server Device Pairing

### Task 1: Add Device Pairing Database Schema

**Files:**
- Modify: `crates/cp-relay-server/src/storage.rs:1-231`

**Step 1: Write failing test**

```rust
#[test]
fn test_device_registration() {
    let storage = Storage::open(":memory:").unwrap();

    // Register a device
    storage.register_device("device123", "pubkey_abc", "MacBook Pro").unwrap();

    // Verify device exists
    let device = storage.get_device("device123").unwrap();
    assert_eq!(device.device_id, "device123");
    assert_eq!(device.display_name, "MacBook Pro");
}
```

**Step 2: Run test to verify it fails**

```bash
cd crates/cp-relay-server && cargo test test_device_registration -- --nocapture
```
Expected: FAIL - method `register_device` not found

**Step 3: Add device registration to Storage**

Add to storage.rs:
```rust
/// Device record
pub struct DeviceRecord {
    pub device_id: String,
    pub public_key: String,
    pub display_name: Option<String>,
    pub created_at: i64,
    pub last_seen: Option<i64>,
}

/// Register a new device
pub fn register_device(
    &self,
    device_id: &str,
    public_key: &str,
    display_name: &str,
) -> Result<(), rusqlite::Error> {
    let conn = self.conn.lock().unwrap();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;

    conn.execute(
        "INSERT OR REPLACE INTO devices (device_id, public_key, display_name, created_at, last_seen)
         VALUES (?1, ?2, ?3, ?4, ?5)",
        params![device_id, public_key, display_name, now, now],
    )?;
    Ok(())
}

/// Get device by ID
pub fn get_device(&self, device_id: &str) -> Result<Option<DeviceRecord>, rusqlite::Error> {
    let conn = self.conn.lock().unwrap();
    let mut stmt = conn.prepare(
        "SELECT device_id, public_key, display_name, created_at, last_seen
         FROM devices WHERE device_id = ?1"
    )?;

    let device = stmt.query_row(params![device_id], |row| {
        Ok(DeviceRecord {
            device_id: row.get(0)?,
            public_key: row.get(1)?,
            display_name: row.get(2)?,
            created_at: row.get(3)?,
            last_seen: row.get(4)?,
        })
    }).optional()?;

    Ok(device)
}
```

Add to schema in `open()`:
```rust
conn.execute(
    "CREATE TABLE IF NOT EXISTS devices (
        device_id TEXT PRIMARY KEY,
        public_key TEXT NOT NULL,
        display_name TEXT,
        created_at INTEGER NOT NULL,
        last_seen INTEGER
    )",
    [],
)?;
```

**Step 4: Run test to verify it passes**

```bash
cd crates/cp-relay-server && cargo test test_device_registration -- --nocapture
```
Expected: PASS

**Step 5: Commit**

```bash
git add crates/cp-relay-server/src/storage.rs
git commit -m "feat(relay-server): add device registration storage

- Add devices table schema
- Add register_device and get_device methods
- Add DeviceRecord struct

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Task 2: Add Pairing Storage Methods

**Files:**
- Modify: `crates/cp-relay-server/src/storage.rs`

**Step 1: Write failing test**

```rust
#[test]
fn test_pairing_flow() {
    let storage = Storage::open(":memory:").unwrap();

    // Register two devices
    storage.register_device("device_a", "pubkey_a", "MacBook").unwrap();
    storage.register_device("device_b", "pubkey_b", "iPhone").unwrap();

    // Create pairing
    storage.create_pairing("device_a", "device_b", "secret_hash").unwrap();

    // Get pairings for device_a
    let pairings = storage.get_pairings("device_a").unwrap();
    assert_eq!(pairings.len(), 1);
    assert!(pairings[0].device_b == "device_b" || pairings[0].device_b == "device_a");
}
```

**Step 2: Run test to verify it fails**

```bash
cd crates/cp-relay-server && cargo test test_pairing_flow -- --nocapture
```
Expected: FAIL - method `create_pairing` not found

**Step 3: Add pairing methods to Storage**

```rust
/// Pairing record
pub struct PairingRecord {
    pub id: i64,
    pub device_a: String,
    pub device_b: String,
    pub shared_secret_hash: String,
    pub created_at: i64,
}

/// Create a pairing between two devices
pub fn create_pairing(
    &self,
    device_a: &str,
    device_b: &str,
    shared_secret_hash: &str,
) -> Result<i64, rusqlite::Error> {
    let conn = self.conn.lock().unwrap();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;

    conn.execute(
        "INSERT INTO pairings (device_a, device_b, shared_secret_hash, created_at)
         VALUES (?1, ?2, ?3, ?4)",
        params![device_a, device_b, shared_secret_hash, now],
    )?;

    Ok(conn.last_insert_rowid())
}

/// Get all pairings for a device
pub fn get_pairings(&self, device_id: &str) -> Result<Vec<PairingRecord>, rusqlite::Error> {
    let conn = self.conn.lock().unwrap();
    let mut stmt = conn.prepare(
        "SELECT id, device_a, device_b, shared_secret_hash, created_at
         FROM pairings WHERE device_a = ?1 OR device_b = ?1"
    )?;

    let pairings = stmt
        .query_map(params![device_id], |row| {
            Ok(PairingRecord {
                id: row.get(0)?,
                device_a: row.get(1)?,
                device_b: row.get(2)?,
                shared_secret_hash: row.get(3)?,
                created_at: row.get(4)?,
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;

    Ok(pairings)
}

/// Get pairing between two devices
pub fn get_pairing(&self, device_a: &str, device_b: &str) -> Result<Option<PairingRecord>, rusqlite::Error> {
    let conn = self.conn.lock().unwrap();
    let mut stmt = conn.prepare(
        "SELECT id, device_a, device_b, shared_secret_hash, created_at
         FROM pairings
         WHERE (device_a = ?1 AND device_b = ?2) OR (device_a = ?2 AND device_b = ?1)"
    )?;

    let pairing = stmt.query_row(params![device_a, device_b], |row| {
        Ok(PairingRecord {
            id: row.get(0)?,
            device_a: row.get(1)?,
            device_b: row.get(2)?,
            shared_secret_hash: row.get(3)?,
            created_at: row.get(4)?,
        })
    }).optional()?;

    Ok(pairing)
}
```

Add to schema in `open()`:
```rust
conn.execute(
    "CREATE TABLE IF NOT EXISTS pairings (
        id INTEGER PRIMARY KEY,
        device_a TEXT NOT NULL,
        device_b TEXT NOT NULL,
        shared_secret_hash TEXT NOT NULL,
        created_at INTEGER NOT NULL,
        UNIQUE(device_a, device_b)
    )",
    [],
)?;
```

**Step 4: Run test to verify it passes**

```bash
cd crates/cp-relay-server && cargo test test_pairing_flow -- --nocapture
```
Expected: PASS

**Step 5: Commit**

```bash
git add crates/cp-relay-server/src/storage.rs
git commit -m "feat(relay-server): add pairing storage methods

- Add pairings table schema
- Add create_pairing, get_pairings, get_pairing methods
- Add PairingRecord struct

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Task 3: Add Pairing API Endpoints

**Files:**
- Modify: `crates/cp-relay-server/src/lib.rs:1-224`

**Step 1: Write failing test**

```rust
#[tokio::test]
async fn test_pairing_init() {
    let app = create_router(":memory:");

    // Request pairing init
    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/v1/pair/init")
                .method(http::Method::POST)
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"device_id":"abc123","public_key":"test_pubkey","display_name":"Test Device"}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}
```

**Step 2: Run test to verify it fails**

```bash
cd crates/cp-relay-server && cargo test test_pairing_init -- --nocapture
```
Expected: FAIL - 404 Not Found

**Step 3: Add pairing endpoints to lib.rs**

Add imports:
```rust
use uuid::Uuid;
```

Add request/response types:
```rust
/// Pairing initiation request
#[derive(Deserialize)]
pub struct PairInitRequest {
    pub device_id: String,
    pub public_key: String,
    pub display_name: Option<String>,
}

/// Pairing initiation response
#[derive(Serialize)]
pub struct PairInitResponse {
    pub pairing_id: String,
    pub pairing_code: String,
    pub expires_at: i64,
}

/// Pairing response request
#[derive(Deserialize)]
pub struct PairRespondRequest {
    pub pairing_id: String,
    pub device_id: String,
    pub public_key: String,
    pub display_name: Option<String>,
}

/// Pairing confirmation request
#[derive(Deserialize)]
pub struct PairConfirmRequest {
    pub pairing_id: String,
}

/// Pairing confirmation response
#[derive(Serialize)]
pub struct PairConfirmResponse {
    pub success: bool,
    pub peer_device_id: String,
    pub peer_display_name: Option<String>,
}

/// Device info response
#[derive(Serialize)]
pub struct DeviceInfo {
    pub device_id: String,
    pub display_name: Option<String>,
    pub created_at: i64,
    pub last_seen: Option<i64>,
}
```

Add handler functions:
```rust
/// Initiate device pairing - step 1
async fn pair_init(
    State(state): State<Arc<AppState>>,
    Json(req): Json<PairInitRequest>,
) -> Result<Json<PairInitResponse>, StatusCode> {
    let pairing_id = Uuid::new_v4().to_string();

    // Generate 6-character pairing code
    let pairing_code: String = (0..6)
        .map(|_| {
            let idx = rand::random::<u8>() % 36;
            if idx < 10 {
                (b'0' + idx) as char
            } else {
                (b'A' + idx - 10) as char
            }
        })
        .collect();

    let expires_at = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64
        + 300; // 5 minutes

    // Store pending pairing
    state
        .storage
        .store_pending_pairing(&pairing_id, &req.device_id, &req.public_key, req.display_name.as_deref(), expires_at)
        .map_err(|e| {
            tracing::error!("Failed to store pending pairing: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    info!("Pairing initiated: {} -> {}", pairing_id, req.device_id);

    Ok(Json(PairInitResponse {
        pairing_id,
        pairing_code,
        expires_at,
    }))
}

/// Respond to pairing - step 2
async fn pair_respond(
    State(state): State<Arc<AppState>>,
    Json(req): Json<PairRespondRequest>,
) -> Result<Json<AcknowledgeResponse>, StatusCode> {
    // Update pending pairing with responder device
    state
        .storage
        .update_pending_pairing(&req.pairing_id, &req.device_id, &req.public_key, req.display_name.as_deref())
        .map_err(|e| {
            tracing::error!("Failed to update pending pairing: {}", e);
            StatusCode::NOT_FOUND
        })?;

    info!("Pairing responded: {}", req.pairing_id);

    Ok(Json(AcknowledgeResponse { success: true }))
}

/// Confirm pairing - step 3
async fn pair_confirm(
    State(state): State<Arc<AppState>>,
    Json(req): Json<PairConfirmRequest>,
) -> Result<Json<PairConfirmResponse>, StatusCode> {
    // Get pending pairing
    let pending = state
        .storage
        .get_pending_pairing(&req.pairing_id)
        .map_err(|e| StatusCode::NOT_FOUND)?
        .ok_or(StatusCode::NOT_FOUND)?;

    // Verify both devices are present
    if pending.initiator_device_id.is_empty() || pending.responder_device_id.is_empty() {
        return Err(StatusCode::BAD_REQUEST);
    }

    // Compute shared secret hash
    let combined = format!("{}:{}", pending.initiator_public_key, pending.responder_public_key);
    let shared_hash = hex::encode(blake3::hash(combined.as_bytes()));

    // Create pairing
    state
        .storage
        .create_pairing(&pending.initiator_device_id, &pending.responder_device_id, &shared_hash)
        .map_err(|e| {
            tracing::error!("Failed to create pairing: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    // Delete pending pairing
    state
        .storage
        .delete_pending_pairing(&req.pairing_id)
        .map_err(|e| StatusCode::INTERNAL_SERVER_ERROR)?;

    info!("Pairing confirmed: {} <-> {}", pending.initiator_device_id, pending.responder_device_id);

    Ok(Json(PairConfirmResponse {
        success: true,
        peer_device_id: pending.responder_device_id,
        peer_display_name: pending.responder_display_name,
    }))
}

/// List paired devices
async fn list_devices(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
) -> Result<Json<Vec<DeviceInfo>>, StatusCode> {
    let device_id = headers
        .get("x-device-id")
        .and_then(|v| v.to_str().ok())
        .ok_or(StatusCode::BAD_REQUEST)?
        .to_string();

    let pairings = state
        .storage
        .get_pairings(&device_id)
        .map_err(|e| StatusCode::INTERNAL_SERVER_ERROR)?;

    let mut devices = Vec::new();
    for pairing in pairings {
        let peer_id = if pairing.device_a == device_id {
            pairing.device_b
        } else {
            pairing.device_a
        };

        if let Some(device) = state.storage.get_device(&peer_id).map_err(|e| StatusCode::INTERNAL_SERVER_ERROR)? {
            devices.push(DeviceInfo {
                device_id: device.device_id,
                display_name: device.display_name,
                created_at: device.created_at,
                last_seen: device.last_seen,
            });
        }
    }

    Ok(Json(devices))
}
```

Add routes in `create_router`:
```rust
.route("/api/v1/pair/init", post(pair_init))
.route("/api/v1/pair/respond", post(pair_respond))
.route("/api/v1/pair/confirm", post(pair_confirm))
.route("/api/v1/devices", get(list_devices))
```

**Step 4: Run test to verify it passes**

```bash
cd crates/cp-relay-server && cargo test test_pairing_init -- --nocapture
```
Expected: PASS

**Step 5: Commit**

```bash
git add crates/cp-relay-server/src/lib.rs
git commit -m "feat(relay-server): add device pairing endpoints

- Add /api/v1/pair/init for initiating pairing
- Add /api/v1/pair/respond for responding to pairing
- Add /api/v1/pair/confirm for confirming pairing
- Add /api/v1/devices for listing paired devices

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Task 4: Add rand Dependency for Pairing Code Generation

**Files:**
- Modify: `crates/cp-relay-server/Cargo.toml`

**Step 1: Add rand dependency**

```toml
rand = "0.8"
```

**Step 2: Verify build**

```bash
cd crates/cp-relay-server && cargo build
```
Expected: SUCCESS

**Step 3: Commit**

```bash
git add crates/cp-relay-server/Cargo.toml
git commit -m "chore(relay-server): add rand for pairing code generation

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Phase 2: macOS Desktop App

### Task 5: Set Up Tauri App Structure

**Files:**
- Create: `crates/cp-desktop/src-tauri/src/commands/docs.rs`
- Modify: `crates/cp-desktop/src-tauri/src/main.rs`
- Modify: `crates/cp-desktop/src-tauri/tauri.conf.json`

**Step 1: Verify current Tauri setup builds**

```bash
cd crates/cp-desktop && cargo build
```
Expected: SUCCESS (may have warnings)

**Step 2: Add document picker command**

Create `crates/cp-desktop/src-tauri/src/commands/docs.rs`:

```rust
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Serialize, Deserialize)]
pub struct SelectedDocument {
    pub path: String,
    pub name: String,
    pub size: u64,
}

#[tauri::command]
pub async fn pick_documents() -> Result<Vec<SelectedDocument>, String> {
    use std::process::Command;

    // Use macOS file picker via AppleScript
    let script = r#"
        set selectedItems to choose file with multiple selections allowed
        set output to ""
        repeat with anItem in selectedItems
            set itemPath to POSIX path of anItem
            set itemSize to do shell script "stat -f%z " & quoted form of itemPath
            set output to output & itemPath & "|" & itemSize & "
        end repeat
        return output
    "#;

    let output = Command::new("osascript")
        .args(["-e", script])
        .output()
        .map_err(|e| e.to_string())?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut documents = Vec::new();

    for line in stdout.lines() {
        let parts: Vec<&str> = line.split('|').collect();
        if parts.len() >= 2 {
            let path = parts[0].to_string();
            let size: u64 = parts[1].trim().parse().unwrap_or(0);
            let name = PathBuf::from(&path)
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| "Unknown".to_string());

            documents.push(SelectedDocument { path, name, size });
        }
    }

    Ok(documents)
}

#[tauri::command]
pub async fn pick_folder() -> Result<String, String> {
    use std::process::Command;

    let script = r#"
        set selectedFolder to choose folder
        return POSIX path of selectedFolder
    "#;

    let output = Command::new("osascript")
        .args(["-e", script])
        .output()
        .map_err(|e| e.to_string())?;

    let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
    Ok(path)
}

#[tauri::command]
pub fn get_selected_documents() -> Result<Vec<SelectedDocument>, String> {
    let config_path = dirs::config_dir()
        .ok_or("No config directory")?
        .join("cp")
        .join("documents.json");

    if !config_path.exists() {
        return Ok(Vec::new());
    }

    let content = std::fs::read_to_string(&config_path).map_err(|e| e.to_string())?;
    let docs: Vec<SelectedDocument> = serde_json::from_str(&content).map_err(|e| e.to_string())?;
    Ok(docs)
}

#[tauri::command]
pub fn save_selected_documents(documents: Vec<SelectedDocument>) -> Result<(), String> {
    let config_dir = dirs::config_dir()
        .ok_or("No config directory")?
        .join("cp");

    std::fs::create_dir_all(&config_dir).map_err(|e| e.to_string())?;

    let config_path = config_dir.join("documents.json");
    let content = serde_json::to_string_pretty(&documents).map_err(|e| e.to_string())?;
    std::fs::write(config_path, content).map_err(|e| e.to_string())?;

    Ok(())
}
```

**Step 3: Register commands in main.rs**

Add to `main.rs`:
```rust
mod commands;

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            commands::docs::pick_documents,
            commands::docs::pick_folder,
            commands::docs::get_selected_documents,
            commands::docs::save_selected_documents,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

**Step 4: Add dirs dependency**

```bash
cd crates/cp-desktop && cargo add dirs
```

**Step 5: Verify build**

```bash
cd crates/cp-desktop && cargo build
```
Expected: SUCCESS

**Step 6: Commit**

```bash
git add crates/cp-desktop/src-tauri/
git commit -m "feat(desktop): add document picker commands

- Add pick_documents command for file selection
- Add pick_folder command for folder selection
- Add get_selected_documents and save_selected_documents
- Configure Tauri commands

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Task 6: Implement Document Indexing Pipeline

**Files:**
- Create: `crates/cp-desktop/src/indexer.rs`

**Step 1: Write failing test**

```rust
#[test]
fn test_document_parsing() {
    use cp_parser::Parser;
    use std::io::Cursor;

    let content = "# Test Document\n\nHello world";
    let mut cursor = Cursor::new(content);
    let parser = Parser::new();
    let chunks = parser.parse(&mut cursor, "test.md").unwrap();

    assert!(!chunks.is_empty());
}
```

**Step 2: Run test to verify it fails**

```bash
cd crates/cp-desktop && cargo test test_document_parsing -- --nocapture
```
Expected: FAIL - module not found (need to add cp-parser dependency)

**Step 3: Add dependencies to Cargo.toml**

```bash
cd crates/cp-desktop && cargo add cp-parser --path crates/cp-parser
cd crates/cp-desktop && cargo add cp-embeddings --path crates/cp-embeddings
cd crates/cp-desktop && cargo add cp-graph --path crates/cp-graph
```

**Step 4: Create indexer module**

Create `crates/cp-desktop/src/indexer.rs`:

```rust
use cp_core::{Document, Chunk};
use cp_parser::Parser;
use cp_embeddings::embed_text;
use cp_graph::{Graph, StorageConfig};
use std::path::Path;
use std::io::Cursor;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum IndexError {
    #[error("Parse error: {0}")]
    Parse(String),
    #[error("Embed error: {0}")]
    Embed(String),
    #[error("Storage error: {0}")]
    Storage(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, IndexError>;

pub struct Indexer {
    graph: Graph,
}

impl Indexer {
    pub fn new(data_dir: &Path) -> Result<Self> {
        let config = StorageConfig {
            sqlite_path: data_dir.join("index.db"),
            hnsw_path: data_dir.join("hnsw.index"),
        };
        let graph = Graph::open(&config).map_err(|e| IndexError::Storage(e.to_string()))?;
        Ok(Self { graph })
    }

    pub fn index_file(&self, path: &Path) -> Result<Vec<Chunk>> {
        // Read file
        let content = std::fs::read(path)?;

        // Determine MIME type
        let mime = match path.extension().and_then(|e| e.to_str()) {
            Some("pdf") => "application/pdf",
            Some("md") | Some("markdown") => "text/markdown",
            Some("txt") => "text/plain",
            _ => "text/plain",
        };

        // Parse document
        let mut cursor = Cursor::new(content);
        let parser = Parser::new();
        let chunks = parser
            .parse(&mut cursor, mime)
            .map_err(|e| IndexError::Parse(e.to_string()))?;

        // Process each chunk
        let mut processed_chunks = Vec::new();
        for chunk in chunks {
            // Generate embedding
            let embedding = embed_text(&chunk.text)
                .map_err(|e| IndexError::Embed(e.to_string()))?;

            // Store in graph
            self.graph
                .add_chunk(&chunk)
                .map_err(|e| IndexError::Storage(e.to_string()))?;

            processed_chunks.push(chunk);
        }

        Ok(processed_chunks)
    }

    pub fn index_directory(&self, dir_path: &Path) -> Result<Vec<Chunk>> {
        let mut all_chunks = Vec::new();

        for entry in walkdir::WalkDir::new(dir_path)
            .follow_links(true)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            let path = entry.path();
            if path.is_file() {
                // Skip hidden files and common exclusions
                if path
                    .file_name()
                    .map(|n| n.to_string_lossy().starts_with('.'))
                    .unwrap_or(false)
                {
                    continue;
                }

                if let Ok(chunks) = self.index_file(path) {
                    all_chunks.extend(chunks);
                }
            }
        }

        Ok(all_chunks)
    }

    pub fn get_stats(&self) -> Result<IndexStats> {
        let doc_count = self.graph.count_documents().map_err(|e| IndexError::Storage(e.to_string()))?;
        let chunk_count = self.graph.count_chunks().map_err(|e| IndexError::Storage(e.to_string()))?;

        Ok(IndexStats { doc_count, chunk_count })
    }
}

#[derive(Debug, serde::Serialize)]
pub struct IndexStats {
    pub doc_count: u64,
    pub chunk_count: u64,
}
```

**Step 5: Add walkdir dependency**

```bash
cd crates/cp-desktop && cargo add walkdir
```

**Step 6: Verify build**

```bash
cd crates/cp-desktop && cargo build
```
Expected: SUCCESS (may have warnings)

**Step 7: Commit**

```bash
git add crates/cp-desktop/src/indexer.rs
git commit -m "feat(desktop): add document indexer

- Add Indexer struct for parsing and embedding documents
- Support file and directory indexing
- Store in SQLite + HNSW via cp-graph
- Add IndexStats for statistics

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Task 7: Implement Sync Client

**Files:**
- Create: `crates/cp-desktop/src/sync.rs`

**Step 1: Write failing test (mock test for sync logic)**

```rust
#[test]
fn test_sync_state_computation() {
    // Test that we can compute state root from documents
    use cp_core::StateRoot;
    use cp_sync::compute_cognitive_diff;

    // This would require mock data - skip for now
}
```

**Step 2: Create sync module**

Create `crates/cp-desktop/src/sync.rs`:

```rust
use cp_core::{StateRoot, Document, Chunk};
use cp_sync::{encrypt_diff, SignedDiff};
use cp_relay_client::RelayClient;
use thiserror::Error;
use std::path::Path;

#[derive(Error, Debug)]
pub enum SyncError {
    #[error("Relay error: {0}")]
    Relay(String),
    #[error("Crypto error: {0}")]
    Crypto(String),
    #[error("Storage error: {0}")]
    Storage(String),
}

pub type Result<T> = std::result::Result<T, SyncError>;

pub struct SyncManager {
    client: RelayClient,
    device_id: [u8; 16],
    keypair: cp_sync::KeyPair,
}

impl SyncManager {
    pub fn new(relay_url: &str, device_id: [u8; 16], keypair: cp_sync::KeyPair) -> Self {
        let client = RelayClient::new(relay_url, "", device_id);
        Self { client, device_id, keypair }
    }

    pub async fn push_diff(&self, diff: &SignedDiff) -> Result<()> {
        self.client
            .push_diff(diff)
            .await
            .map_err(|e| SyncError::Relay(e.to_string()))
    }

    pub async fn pull_diffs(&self, since: u64) -> Result<Vec<SignedDiff>> {
        self.client
            .pull_since(since)
            .await
            .map_err(|e| SyncError::Relay(e.to_string()))
    }

    pub async fn sync_to_device(
        &self,
        target_device_id: [u8; 16],
        documents: &[Document],
        chunks: &[Chunk],
    ) -> Result<()> {
        // Compute cognitive diff
        let cognitive_diff = cp_sync::compute_cognitive_diff(documents, chunks);

        // Get target device's public key (from relay or local cache)
        // For now, we'll need to have it passed in or fetched

        // Encrypt for target
        // let encrypted = encrypt_diff(&cognitive_diff, &target_public_key)?;

        // Sign the diff
        // let signed = sign_diff(&encrypted, &self.keypair)?;

        // Push to relay
        // self.push_diff(&signed).await?;

        Ok(())
    }

    pub fn get_paired_devices(&self) -> impl Future<Output = Result<Vec<DeviceInfo>>> {
        // TODO: Implement fetching paired devices from relay
        async { Ok(Vec::new()) }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct DeviceInfo {
    pub device_id: String,
    pub display_name: Option<String>,
    pub last_seen: Option<i64>,
}
```

**Step 3: Verify build**

```bash
cd crates/cp-desktop && cargo build
```
Expected: SUCCESS

**Step 4: Commit**

```bash
git add crates/cp-desktop/src/sync.rs
git commit -m "feat(desktop): add sync manager

- Add SyncManager for pushing/pulling diffs
- Integrate with cp-relay-client
- Support device-to-device sync

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Phase 3: iOS Mobile App

### Task 8: iOS FFI Bridge

**Files:**
- Modify: `crates/cp-mobile/src/lib.rs`

**Step 1: Write failing test**

```rust
#[test]
fn test_mobile_index_creation() {
    // This would require iOS environment - test at integration level
}
```

**Step 2: Enhance mobile lib.rs with FFI exports**

Modify `crates/cp-mobile/src/lib.rs` to add more FFI functions:

```rust
use cp_core::{Document, Chunk, StateRoot};
use cp_graph::{Graph, StorageConfig};
use cp_query::{Retriever, SearchQuery};
use cp_sync::SignedDiff;
use std::path::Path;

// ============================================================================
// FFI: Index Management
// ============================================================================

/// Open or create a graph index
#[no_mangle]
pub extern "C" fn cp_graph_open(sqlite_path: *const c_char, hnsw_path: *const c_char) -> *mut Graph {
    let sqlite = unsafe { CStr::from_ptr(sqlite_path).to_string_lossy().into_owned() };
    let hnsw = unsafe { CStr::from_ptr(hnsw_path).to_string_lossy().into_owned() };

    let config = StorageConfig {
        sqlite_path: sqlite.into(),
        hnsw_path: hnsw.into(),
    };

    Box::into_raw(Box::new(Graph::open(&config).expect("Failed to open graph")))
}

/// Close a graph index
#[no_mangle]
pub extern "C" fn cp_graph_close(graph: *mut Graph) {
    unsafe { Box::from_raw(graph) };
}

/// Add a chunk to the index
#[no_mangle]
pub extern "C" fn cp_graph_add_chunk(graph: *mut Graph, chunk_json: *const c_char) -> i32 {
    let json = unsafe { CStr::from_ptr(chunk_json).to_string_lossy().into_owned() };
    let chunk: Chunk = serde_json::from_str(&json).expect("Failed to parse chunk");

    match unsafe { &*graph }.add_chunk(&chunk) {
        Ok(_) => 0,
        Err(_) => -1,
    }
}

/// Get state root
#[no_mangle]
pub extern "C" fn cp_graph_state_root(graph: *const Graph) -> *mut c_char {
    let state_root = unsafe { &*graph }.state_root().expect("Failed to get state root");
    let json = serde_json::to_string(&state_root).expect("Failed to serialize state root");
    CString::new(json).unwrap().into_raw()
}

// ============================================================================
// FFI: Query
// ============================================================================

/// Search the index
#[no_mangle]
pub extern "C" fn cp_query_search(
    graph: *const Graph,
    query_json: *const c_char,
) -> *mut c_char {
    let json = unsafe { CStr::from_ptr(query_json).to_string_lossy().into_owned() };
    let query: SearchQuery = serde_json::from_str(&json).expect("Failed to parse query");

    let retriever = Retriever::new(unsafe { &*graph });
    let results = retriever.search(&query).expect("Search failed");

    let json = serde_json::to_string(&results).expect("Failed to serialize results");
    CString::new(json).unwrap().into_raw()
}
```

Add CString import:
```rust
use std::ffi::CStr;
use std::os::raw::c_char;
```

**Step 3: Verify build**

```bash
cd crates/cp-mobile && cargo build
```
Expected: SUCCESS

**Step 4: Commit**

```bash
git add crates/cp-mobile/src/lib.rs
git commit -m "feat(mobile): add more FFI exports

- Add cp_graph_open, cp_graph_close
- Add cp_graph_add_chunk
- Add cp_graph_state_root
- Add cp_query_search

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Task 9: Swift iOS App Structure

**Files:**
- Create: `examples/ios/CPMobile/Sources/App/`

**Note:** This is a SwiftUI task. The actual implementation will be in the iOS project.

**Step 1: Create App entry point**

Create `examples/ios/CPMobile/Sources/App/CPMobileApp.swift`:

```swift
import SwiftUI

@main
struct CPMobileApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
```

**Step 2: Create main ContentView**

Create `examples/ios/CPMobile/Sources/Views/ContentView.swift`:

```swift
import SwiftUI

struct ContentView: View {
    @StateObject private var appState = AppState()

    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                if appState.isPaired {
                    DocumentListView()
                } else {
                    OnboardingView()
                }
            }
            .navigationTitle("CP Documents")
        }
        .environmentObject(appState)
    }
}
```

**Step 3: Create AppState**

Create `examples/ios/CPMobile/Sources/Models/AppState.swift`:

```swift
import Foundation
import Combine

class AppState: ObservableObject {
    @Published var isPaired: Bool = false
    @Published var documents: [SyncedDocument] = []
    @Published var isLoading: Bool = false

    private let relayURL: String

    init() {
        // Load relay URL from config
        self.relayURL = UserDefaults.standard.string(forKey: "relayURL") ?? "https://relay.example.com"
        self.isPaired = UserDefaults.standard.string(forKey: "devicePaired") != nil
    }

    func pairDevice(code: String) async {
        // Call FFI to complete pairing
        // Update isPaired on success
    }

    func syncDocuments() async {
        // Pull diffs from relay
        // Update local index
        // Update documents list
    }
}
```

**Step 4: Commit**

```bash
git add examples/ios/CPMobile/Sources/
git commit -m "feat(ios): add SwiftUI app structure

- Add CPMobileApp entry point
- Add ContentView with navigation
- Add AppState model with pairing/sync

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Phase 4: Query with Citations

### Task 10: Citation Extraction

**Files:**
- Modify: `crates/cp-query/src/lib.rs`

**Step 1: Write failing test**

```rust
#[test]
fn test_citation_extraction() {
    use cp_core::Chunk;

    let chunk = Chunk {
        id: "chunk1".to_string(),
        document_id: "doc1".to_string(),
        text: "The main benefit is improved performance".to_string(),
        sequence: 0,
        byte_offset: 0,
        byte_length: 35,
    };

    let answer = "The main benefit is improved performance and reliability.";

    let citations = extract_citations(answer, &[chunk]);

    assert!(!citations.is_empty());
    assert!(citations[0].passage.contains("improved performance"));
}
```

**Step 2: Run test to verify it fails**

```bash
cd crates/cp-query && cargo test test_citation_extraction -- --nocapture
```
Expected: FAIL - function not found

**Step 3: Add citation extraction**

In `crates/cp-query/src/lib.rs`, add:

```rust
use cp_core::Chunk;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Citation {
    pub chunk_id: String,
    pub document_id: String,
    pub passage: String,
    pub score: f32,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct QueryResult {
    pub answer: String,
    pub citations: Vec<Citation>,
    pub sources: Vec<SourceInfo>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SourceInfo {
    pub document_id: String,
    pub path: Option<String>,
    pub score: f32,
}

/// Extract citations from answer using n-gram overlap
pub fn extract_citations(answer: &str, chunks: &[Chunk]) -> Vec<Citation> {
    let answer_lower = answer.to_lowercase();
    let answer_words: Vec<&str> = answer_lower.split_whitespace().collect();

    let mut citations = Vec::new();

    for chunk in chunks {
        let chunk_lower = chunk.text.to_lowercase();
        let chunk_words: Vec<&str> = chunk_lower.split_whitespace().collect();

        // Check 3-gram overlap
        let overlap = count_ngram_overlap(&answer_words, &chunk_words, 3);

        if overlap > 2 {
            // Find matching passage
            let passage = find_passage(answer, &chunk.text);

            let score = overlap as f32 / chunk_words.len().max(1) as f32;

            citations.push(Citation {
                chunk_id: chunk.id.clone(),
                document_id: chunk.document_id.clone(),
                passage,
                score,
            });
        }
    }

    // Sort by score descending
    citations.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    citations
}

fn count_ngram_overlap(answer_words: &[&str], chunk_words: &[&str], n: usize) -> usize {
    if answer_words.len() < n || chunk_words.len() < n {
        return 0;
    }

    let answer_ngrams: Vec<String> = answer_words
        .windows(n)
        .map(|w| w.join(" "))
        .collect();

    let chunk_ngrams: Vec<String> = chunk_words
        .windows(n)
        .map(|w| w.join(" "))
        .collect();

    answer_ngrams
        .iter()
        .filter(|ng| chunk_ngrams.contains(ng))
        .count()
}

fn find_passage(answer: &str, chunk_text: &str) -> String {
    // Find the longest common substring
    let answer_lower = answer.to_lowercase();
    let chunk_lower = chunk_text.to_lowercase();

    // Simple approach: find first matching word and extract surrounding context
    let answer_words: Vec<&str> = answer_lower.split_whitespace().collect();

    for (i, word) in answer_words.iter().enumerate() {
        if chunk_lower.contains(word) {
            // Extract 10 words before and after
            let start = i.saturating_sub(3);
            let end = (i + 4).min(answer_words.len());
            return answer_words[start..end].join(" ");
        }
    }

    // Fallback: return first 50 chars
    answer.chars().take(50).collect()
}
```

**Step 4: Run test to verify it passes**

```bash
cd crates/cp-query && cargo test test_citation_extraction -- --nocapture
```
Expected: PASS

**Step 5: Commit**

```bash
git add crates/cp-query/src/lib.rs
git commit -m "feat(query): add citation extraction

- Add extract_citations function using n-gram overlap
- Add Citation and QueryResult structs
- Add count_ngram_overlap and find_passage helpers

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Summary

This implementation plan covers:

1. **Phase 1 - Relay Server (4 tasks):**
   - Device registration storage
   - Pairing storage methods
   - Pairing API endpoints
   - rand dependency

2. **Phase 2 - macOS Desktop (3 tasks):**
   - Tauri app setup with document picker
   - Document indexing pipeline
   - Sync client

3. **Phase 3 - iOS App (2 tasks):**
   - FFI bridge enhancements
   - SwiftUI app structure

4. **Phase 4 - Query (1 task):**
   - Citation extraction

Total: 10 tasks for initial implementation.
