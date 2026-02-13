use cfs_core::{CognitiveDiff, DiffMetadata, Document, Hlc};
use cfs_graph::GraphStore;
use uuid::Uuid;
use std::path::PathBuf;

#[test]
fn test_sync_convergence_simple() {
    let mut graph_a = GraphStore::in_memory().unwrap();
    let mut graph_b = GraphStore::in_memory().unwrap();

    let device_id = Uuid::new_v4();
    let doc = Document::new(PathBuf::from("test.md"), b"Content", 123);

    // Create HLC timestamp using system time
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;
    let node_id: [u8; 16] = *Uuid::new_v4().as_bytes();
    let hlc = Hlc::new(now_ms, node_id);

    // 1. Create Diff on A
    let mut diff = CognitiveDiff::empty([0u8; 32], device_id, 1, hlc);
    diff.added_docs.push(doc.clone());

    let new_root_hash = [1u8; 32];
    let hlc2 = Hlc::new(now_ms + 1, node_id);
    diff.metadata = DiffMetadata {
        prev_root: [0u8; 32],
        new_root: new_root_hash,
        hlc: hlc2,
        device_id,
        seq: 1,
    };

    // 2. Apply to A and B
    graph_a.apply_diff(&diff).unwrap();
    graph_b.apply_diff(&diff).unwrap();

    // 3. Assert convergence
    let root_a = graph_a.compute_merkle_root().unwrap();
    let root_b = graph_b.compute_merkle_root().unwrap();

    assert_eq!(root_a, root_b, "Merkle roots must converge after applying same diff");
}
