use cfs_core::{CognitiveDiff, CfsError, Hlc, Result, StateRoot};
use cfs_graph::GraphStore;
use cfs_relay_client::RelayClient;
use cfs_sync::CryptoEngine;
use std::sync::{Arc, Mutex};
use tracing::info;
use uuid::Uuid;

/// Manages synchronization with the relay server
pub struct SyncManager {
    relay_client: RelayClient,
    graph: Arc<Mutex<GraphStore>>,
    crypto: CryptoEngine,
    pending_diff: Mutex<CognitiveDiff>,
}

impl SyncManager {
    /// Create a new sync manager
    pub fn new(
        relay_url: &str,
        relay_token: &str,
        graph: Arc<Mutex<GraphStore>>,
        key_path: &std::path::Path,
    ) -> Result<Self> {
        // Load or generate keys
        let crypto = if key_path.exists() {
            let seed = std::fs::read(key_path).map_err(|e| CfsError::Io(e))?;
            if seed.len() != 32 {
                return Err(CfsError::Crypto("Invalid key file length".into()));
            }
            let mut arr = [0u8; 32];
            arr.copy_from_slice(&seed);
            CryptoEngine::new_with_seed(arr)
        } else {
            // V0 Dev Default Key: 32 bytes of 0x01
            let seed = [1u8; 32];
            std::fs::write(key_path, seed).map_err(|e| CfsError::Io(e))?;
            CryptoEngine::new_with_seed(seed)
        };

        // Initialize pending diff (empty for now, essentially a "session" diff)
        // In reality, we should load pending diffs from disk or DB to persist across restarts.
        // For MVP, if we crash, we lose un-pushed changes (or regenerate them from DB scan? No).
        // Best effort: we assume DesktopApp runs long enough to push.
        let device_id = Uuid::new_v4();
        let node_id = *device_id.as_bytes();
        let hlc = Hlc::new(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            node_id,
        );
        let pending = CognitiveDiff::empty([0u8; 32], device_id, 0, hlc);

        Ok(Self {
            relay_client: RelayClient::new_legacy(relay_url, relay_token),
            graph,
            crypto,
            pending_diff: Mutex::new(pending),
        })
    }

    /// Record a document addition (deduplicates by ID)
    pub fn record_add_doc(&self, doc: cfs_core::Document) {
        let mut diff = self.pending_diff.lock().unwrap();
        // Deduplicate: remove any existing entry with same ID, then add new one
        diff.added_docs.retain(|d| d.id != doc.id);
        diff.added_docs.push(doc);
    }

    /// Record a chunk addition (deduplicates by ID)
    pub fn record_add_chunk(&self, chunk: cfs_core::Chunk) {
        let mut diff = self.pending_diff.lock().unwrap();
        diff.added_chunks.retain(|c| c.id != chunk.id);
        diff.added_chunks.push(chunk);
    }

    /// Record an embedding addition (deduplicates by ID)
    pub fn record_add_embedding(&self, emb: cfs_core::Embedding) {
        let mut diff = self.pending_diff.lock().unwrap();
        diff.added_embeddings.retain(|e| e.id != emb.id);
        diff.added_embeddings.push(emb);
    }

    /// Record a document removal
    pub fn record_remove_doc(&self, doc_id: Uuid, chunk_ids: Vec<Uuid>, embedding_ids: Vec<Uuid>) {
        let mut diff = self.pending_diff.lock().unwrap();
        diff.removed_doc_ids.push(doc_id);
        diff.removed_chunk_ids.extend(chunk_ids);
        diff.removed_embedding_ids.extend(embedding_ids);
    }

    /// Push pending changes to the relay
    pub async fn push(&self) -> Result<()> {
        // CRITICAL: Take both locks together to prevent race condition
        // The merkle root must be computed with the same state as the diff
        let (mut diff, new_semantic_root, prev_root, prev_seq) = {
            let graph = self.graph.lock().unwrap();
            let mut pending = self.pending_diff.lock().unwrap();

            // Get current state
            let current_root_hash = graph.get_latest_root()?.map(|r| r.hash).unwrap_or([0u8; 32]);
            let current_seq = graph.get_latest_root()?.map(|r| r.seq).unwrap_or(0);

            // Compute merkle root while holding both locks
            let new_root = graph.compute_merkle_root()?;

            if pending.is_empty() && new_root == current_root_hash {
                info!("State is identical and no pending changes. Skipping sync.");
                return Ok(());
            }

            // Swap with new empty diff while still holding locks
            let old = pending.clone();
            let device_id = old.metadata.device_id;
            let node_id = *device_id.as_bytes();
            let hlc = Hlc::new(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_millis() as u64,
                node_id,
            );
            *pending = CognitiveDiff::empty([0u8; 32], device_id, old.metadata.seq + 1, hlc);

            (old, new_root, current_root_hash, current_seq)
        };

        info!("Syncing {} semantic changes...", diff.change_count());

        // 2. Set diff metadata (prev_root and prev_seq already extracted above)
        diff.metadata.prev_root = prev_root;
        diff.metadata.seq = prev_seq + 1;
        let device_id = Uuid::new_v4();
        let node_id = *device_id.as_bytes();
        let hlc = Hlc::new(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            node_id,
        );
        diff.metadata.hlc = hlc.clone();
        diff.metadata.device_id = device_id;
        diff.metadata.new_root = new_semantic_root;

        // 3. Create StateRoot object and Sign it
        let signature = self.crypto.sign(&new_semantic_root).to_bytes();

        let state_root = StateRoot {
            hash: new_semantic_root,
            parent: if prev_root == [0u8; 32] { None } else { Some(prev_root) },
            hlc,
            device_id: diff.metadata.device_id,
            signature,
            seq: diff.metadata.seq,
        };

        // 4. Store new root in DB
        {
            let mut graph = self.graph.lock().unwrap();
            graph.set_latest_root(&state_root)?;
        }

        // 5. Encrypt Diff
        let payload = self.crypto.encrypt_diff(&diff)?; // Encrypts and Signs

        // 6. Upload
        self.relay_client.upload_diff(payload, &state_root).await?;

        info!("Sync complete. New root: {}", state_root.hash_hex());

        Ok(())
    }

    /// Pull changes from the relay and apply to local graph
    pub async fn pull(&self) -> Result<usize> {
        let remote_roots = self.relay_client.get_roots(None).await?;
        let mut applied_count = 0;
        
        let local_head = {
            let graph = self.graph.lock().unwrap();
            graph.get_latest_root()?.map(|r| r.hash)
        };
        
        for root_hex in remote_roots {
            let root_bytes = hex::decode(&root_hex).map_err(|e| CfsError::Parse(e.to_string()))?;
            if Some(root_bytes.as_slice().try_into().unwrap()) == local_head {
                continue; 
            }
            
            let payload = self.relay_client.get_diff(&root_hex).await?;
            let diff = self.crypto.decrypt_diff(&payload)?;
            
            {
                let mut graph = self.graph.lock().unwrap();
                graph.apply_diff(&diff)?;
            }
            applied_count += 1;
        }

        if applied_count > 0 {
            info!("Pulled {} new diffs from relay", applied_count);
        }

        Ok(applied_count)
    }
}
