//! Persistent HNSW Index using usearch
//!
//! Per CFS-002: Provides mmap-backed HNSW for fast vector search
//! with checkpoint validation against state root.

use cfs_core::{CfsError, Result};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use tracing::{info, warn};
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};
use uuid::Uuid;

/// Configuration for the persistent HNSW index
#[derive(Debug, Clone)]
pub struct IndexConfig {
    /// Number of dimensions
    pub dimensions: usize,
    /// HNSW M parameter (max connections per node)
    pub connectivity: usize,
    /// HNSW ef_construction parameter
    pub ef_construction: usize,
    /// HNSW ef_search parameter
    pub ef_search: usize,
    /// Maximum capacity (number of vectors)
    pub capacity: usize,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            dimensions: 384,
            connectivity: 16,
            ef_construction: 200,
            ef_search: 64,
            capacity: 100_000,
        }
    }
}

/// Persistent HNSW index with mmap-backed storage
pub struct PersistentHnswIndex {
    /// The usearch index
    index: Index,
    /// Path to the index file (None for in-memory)
    index_path: Option<PathBuf>,
    /// State root hash when index was last checkpointed
    checkpoint_root: Option<[u8; 32]>,
    /// Mapping from UUID to internal index key
    uuid_to_key: HashMap<Uuid, u64>,
    /// Mapping from internal key to UUID
    key_to_uuid: HashMap<u64, Uuid>,
    /// Next available key
    next_key: u64,
    /// Whether the index needs rebuild
    needs_rebuild: bool,
    /// Configuration
    config: IndexConfig,
}

impl PersistentHnswIndex {
    /// Create a new in-memory index
    pub fn new(config: IndexConfig) -> Result<Self> {
        let options = IndexOptions {
            dimensions: config.dimensions,
            metric: MetricKind::Cos,
            quantization: ScalarKind::F32,
            connectivity: config.connectivity,
            expansion_add: config.ef_construction,
            expansion_search: config.ef_search,
            multi: false,
        };

        let index = Index::new(&options)
            .map_err(|e| CfsError::Database(format!("Failed to create index: {}", e)))?;

        index
            .reserve(config.capacity)
            .map_err(|e| CfsError::Database(format!("Failed to reserve capacity: {}", e)))?;

        Ok(Self {
            index,
            index_path: None,
            checkpoint_root: None,
            uuid_to_key: HashMap::new(),
            key_to_uuid: HashMap::new(),
            next_key: 0,
            needs_rebuild: false,
            config,
        })
    }

    /// Open or create a persistent index at the given path
    pub fn open(path: PathBuf, config: IndexConfig) -> Result<Self> {
        let options = IndexOptions {
            dimensions: config.dimensions,
            metric: MetricKind::Cos,
            quantization: ScalarKind::F32,
            connectivity: config.connectivity,
            expansion_add: config.ef_construction,
            expansion_search: config.ef_search,
            multi: false,
        };

        let index = Index::new(&options)
            .map_err(|e| CfsError::Database(format!("Failed to create index: {}", e)))?;

        let mut inst = Self {
            index,
            index_path: Some(path.clone()),
            checkpoint_root: None,
            uuid_to_key: HashMap::new(),
            key_to_uuid: HashMap::new(),
            next_key: 0,
            needs_rebuild: false,
            config,
        };

        // Try to load existing index
        if path.exists() {
            match inst.load() {
                Ok(_) => info!("Loaded existing index from {:?}", path),
                Err(e) => {
                    warn!("Failed to load index: {}, will rebuild", e);
                    inst.needs_rebuild = true;
                }
            }
        }

        Ok(inst)
    }

    /// Load index from disk
    fn load(&mut self) -> Result<()> {
        if let Some(path) = self.index_path.clone() {
            self.index
                .load(path.to_str().unwrap())
                .map_err(|e| CfsError::Database(format!("Failed to load index: {}", e)))?;

            // Load mapping file
            let map_path = path.with_extension("map");
            if map_path.exists() {
                let data = std::fs::read(&map_path)
                    .map_err(|e| CfsError::Database(format!("Failed to read map: {}", e)))?;
                self.load_mapping(&data)?;
            }

            // Load checkpoint file
            let checkpoint_path = path.with_extension("checkpoint");
            if checkpoint_path.exists() {
                let data = std::fs::read(&checkpoint_path)
                    .map_err(|e| CfsError::Database(format!("Failed to read checkpoint: {}", e)))?;
                if data.len() == 32 {
                    let mut root = [0u8; 32];
                    root.copy_from_slice(&data);
                    self.checkpoint_root = Some(root);
                }
            }
        }
        Ok(())
    }

    /// Save index to disk
    pub fn save(&self) -> Result<()> {
        if let Some(path) = &self.index_path {
            self.index
                .save(path.to_str().unwrap())
                .map_err(|e| CfsError::Database(format!("Failed to save index: {}", e)))?;

            // Save mapping
            let map_path = path.with_extension("map");
            let map_data = self.serialize_mapping();
            std::fs::write(&map_path, map_data)
                .map_err(|e| CfsError::Database(format!("Failed to write map: {}", e)))?;

            // Save checkpoint
            if let Some(root) = &self.checkpoint_root {
                let checkpoint_path = path.with_extension("checkpoint");
                std::fs::write(&checkpoint_path, root)
                    .map_err(|e| CfsError::Database(format!("Failed to write checkpoint: {}", e)))?;
            }

            info!("Saved index to {:?}", path);
        }
        Ok(())
    }

    /// Serialize UUID mappings
    fn serialize_mapping(&self) -> Vec<u8> {
        let mut data = Vec::new();
        // Write next_key
        data.extend_from_slice(&self.next_key.to_le_bytes());
        // Write count
        let count = self.uuid_to_key.len() as u64;
        data.extend_from_slice(&count.to_le_bytes());
        // Write mappings
        for (uuid, key) in &self.uuid_to_key {
            data.extend_from_slice(uuid.as_bytes());
            data.extend_from_slice(&key.to_le_bytes());
        }
        data
    }

    /// Load UUID mappings
    fn load_mapping(&mut self, data: &[u8]) -> Result<()> {
        if data.len() < 16 {
            return Err(CfsError::Database("Invalid mapping data".into()));
        }

        let next_key = u64::from_le_bytes(data[0..8].try_into().unwrap());
        let count = u64::from_le_bytes(data[8..16].try_into().unwrap()) as usize;

        self.next_key = next_key;
        self.uuid_to_key.clear();
        self.key_to_uuid.clear();

        let mut offset = 16;
        for _ in 0..count {
            if offset + 24 > data.len() {
                return Err(CfsError::Database("Truncated mapping data".into()));
            }
            let uuid = Uuid::from_slice(&data[offset..offset + 16])
                .map_err(|_| CfsError::Database("Invalid UUID in mapping".into()))?;
            let key = u64::from_le_bytes(data[offset + 16..offset + 24].try_into().unwrap());
            self.uuid_to_key.insert(uuid, key);
            self.key_to_uuid.insert(key, uuid);
            offset += 24;
        }

        Ok(())
    }

    /// Insert a vector
    pub fn insert(&mut self, emb_id: Uuid, vector: Vec<f32>) -> Result<()> {
        let key = self.next_key;
        self.next_key += 1;

        self.index
            .add(key, &vector)
            .map_err(|e| CfsError::Database(format!("Failed to add vector: {}", e)))?;

        self.uuid_to_key.insert(emb_id, key);
        self.key_to_uuid.insert(key, emb_id);

        Ok(())
    }

    /// Search for similar vectors
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(Uuid, f32)> {
        match self.index.search(query, k) {
            Ok(results) => {
                results
                    .keys
                    .iter()
                    .zip(results.distances.iter())
                    .filter_map(|(&key, &dist)| {
                        self.key_to_uuid.get(&key).map(|id| {
                            // Convert distance to similarity (1 - cosine_distance)
                            (*id, 1.0 - dist)
                        })
                    })
                    .collect()
            }
            Err(e) => {
                warn!("Search failed: {}", e);
                Vec::new()
            }
        }
    }

    /// Check if index is valid against current state root
    pub fn is_valid(&self, current_root: &[u8; 32]) -> bool {
        match &self.checkpoint_root {
            Some(root) => root == current_root,
            None => false,
        }
    }

    /// Update checkpoint to current state root
    pub fn checkpoint(&mut self, state_root: [u8; 32]) -> Result<()> {
        self.checkpoint_root = Some(state_root);
        self.needs_rebuild = false;
        self.save()
    }

    /// Mark index as needing rebuild
    pub fn invalidate(&mut self) {
        self.needs_rebuild = true;
    }

    /// Check if index needs rebuild
    pub fn needs_rebuild(&self) -> bool {
        self.needs_rebuild
    }

    /// Clear the index
    pub fn clear(&mut self) -> Result<()> {
        // Recreate the index since usearch Index doesn't have a clear() method
        let options = IndexOptions {
            dimensions: self.config.dimensions,
            metric: MetricKind::Cos,
            quantization: ScalarKind::F32,
            connectivity: self.config.connectivity,
            expansion_add: self.config.ef_construction,
            expansion_search: self.config.ef_search,
            multi: false,
        };

        self.index = Index::new(&options)
            .map_err(|e| CfsError::Database(format!("Failed to recreate index: {}", e)))?;

        self.index
            .reserve(self.config.capacity)
            .map_err(|e| CfsError::Database(format!("Failed to reserve capacity: {}", e)))?;

        self.uuid_to_key.clear();
        self.key_to_uuid.clear();
        self.next_key = 0;
        self.checkpoint_root = None;
        self.needs_rebuild = false;
        Ok(())
    }

    /// Get the number of vectors in the index
    pub fn len(&self) -> usize {
        self.index.size()
    }

    /// Check if index is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Thread-safe wrapper for the persistent index
pub struct SharedPersistentIndex {
    inner: Arc<RwLock<PersistentHnswIndex>>,
}

impl SharedPersistentIndex {
    pub fn new(config: IndexConfig) -> Result<Self> {
        Ok(Self {
            inner: Arc::new(RwLock::new(PersistentHnswIndex::new(config)?)),
        })
    }

    pub fn open(path: PathBuf, config: IndexConfig) -> Result<Self> {
        Ok(Self {
            inner: Arc::new(RwLock::new(PersistentHnswIndex::open(path, config)?)),
        })
    }

    pub fn insert(&self, emb_id: Uuid, vector: Vec<f32>) -> Result<()> {
        let mut index = self.inner.write().unwrap();
        index.insert(emb_id, vector)
    }

    pub fn search(&self, query: &[f32], k: usize) -> Vec<(Uuid, f32)> {
        let index = self.inner.read().unwrap();
        index.search(query, k)
    }

    pub fn save(&self) -> Result<()> {
        let index = self.inner.read().unwrap();
        index.save()
    }

    pub fn checkpoint(&self, state_root: [u8; 32]) -> Result<()> {
        let mut index = self.inner.write().unwrap();
        index.checkpoint(state_root)
    }

    pub fn is_valid(&self, current_root: &[u8; 32]) -> bool {
        let index = self.inner.read().unwrap();
        index.is_valid(current_root)
    }

    pub fn invalidate(&self) {
        let mut index = self.inner.write().unwrap();
        index.invalidate();
    }

    pub fn needs_rebuild(&self) -> bool {
        let index = self.inner.read().unwrap();
        index.needs_rebuild()
    }

    pub fn clear(&self) -> Result<()> {
        let mut index = self.inner.write().unwrap();
        index.clear()
    }

    pub fn len(&self) -> usize {
        let index = self.inner.read().unwrap();
        index.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Clone for SharedPersistentIndex {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_in_memory_index() {
        let mut index = PersistentHnswIndex::new(IndexConfig::default()).unwrap();

        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();

        // Create normalized vectors
        let v1: Vec<f32> = (0..384).map(|i| if i == 0 { 1.0 } else { 0.0 }).collect();
        let v2: Vec<f32> = (0..384).map(|i| if i == 1 { 1.0 } else { 0.0 }).collect();

        index.insert(id1, v1.clone()).unwrap();
        index.insert(id2, v2).unwrap();

        let results = index.search(&v1, 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, id1); // Most similar to itself
    }

    #[test]
    fn test_persistent_index() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("test.usearch");

        let id1 = Uuid::new_v4();
        let v1: Vec<f32> = (0..384).map(|i| if i == 0 { 1.0 } else { 0.0 }).collect();

        // Create and save
        {
            let mut index = PersistentHnswIndex::open(path.clone(), IndexConfig::default()).unwrap();
            index.insert(id1, v1.clone()).unwrap();
            index.checkpoint([1u8; 32]).unwrap();
        }

        // Reload and verify
        {
            let index = PersistentHnswIndex::open(path, IndexConfig::default()).unwrap();
            assert!(index.is_valid(&[1u8; 32]));
            assert!(!index.is_valid(&[2u8; 32]));

            let results = index.search(&v1, 1);
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].0, id1);
        }
    }

    #[test]
    fn test_index_invalidation() {
        let mut index = PersistentHnswIndex::new(IndexConfig::default()).unwrap();

        assert!(!index.needs_rebuild());

        index.invalidate();
        assert!(index.needs_rebuild());

        index.checkpoint([1u8; 32]).unwrap();
        assert!(!index.needs_rebuild());
    }
}
