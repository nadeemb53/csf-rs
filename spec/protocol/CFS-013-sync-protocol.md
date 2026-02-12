# CFS-013: Synchronization Protocol

> **Spec Version**: 1.0.0-draft
> **Status**: Draft
> **Category**: Protocol
> **Requires**: CFS-001, CFS-002, CFS-003

## Synopsis

This specification defines the synchronization protocol for CFS, enabling encrypted state replication between devices via a blind relay server.

## Motivation

Multi-device synchronization is essential for:

1. **Data Availability**: Access knowledge from any device
2. **Offline Resilience**: Devices can sync when connectivity returns
3. **Backup**: Replicated data protects against device loss
4. **Collaboration**: Share knowledge bases (future extension)

CFS synchronization provides:

- **End-to-End Encryption**: Server cannot read content
- **Cryptographic Verification**: State integrity is provable
- **Minimal Bandwidth**: Only deltas are transmitted

## Technical Specification

### 1. Synchronization Architecture

```
┌─────────────────┐                              ┌─────────────────┐
│     Desktop     │                              │     Mobile      │
│    (Writer)     │                              │    (Reader)     │
├─────────────────┤                              ├─────────────────┤
│                 │                              │                 │
│  ┌───────────┐  │    ┌─────────────────┐      │  ┌───────────┐  │
│  │ GraphStore│  │    │   Relay Server  │      │  │ GraphStore│  │
│  └─────┬─────┘  │    │    (Blind)      │      │  └─────┬─────┘  │
│        │        │    │                 │      │        │        │
│        v        │    │  ┌───────────┐  │      │        ^        │
│  ┌───────────┐  │    │  │ Encrypted │  │      │  ┌───────────┐  │
│  │ Generate  │──────>│  │   Diffs   │  │<──────│  │  Apply    │  │
│  │   Diff    │  │    │  │ Storage   │  │      │  │   Diff    │  │
│  └───────────┘  │    │  └───────────┘  │      │  └───────────┘  │
│        │        │    │                 │      │        │        │
│        v        │    └─────────────────┘      │        v        │
│  ┌───────────┐  │                              │  ┌───────────┐  │
│  │  Encrypt  │  │                              │  │  Decrypt  │  │
│  │  + Sign   │  │                              │  │  + Verify │  │
│  └───────────┘  │                              │  └───────────┘  │
│                 │                              │                 │
└─────────────────┘                              └─────────────────┘
```

### 2. Cryptographic Primitives

CFS synchronization uses the following cryptographic algorithms:

| Purpose | Algorithm | Key Size |
|---------|-----------|----------|
| Symmetric Encryption | XChaCha20-Poly1305 | 256-bit |
| Key Derivation | HKDF-SHA256 | Variable |
| Signing | Ed25519 | 256-bit |
| Hashing | BLAKE3 | 256-bit |

#### Key Hierarchy

```
┌─────────────────────┐
│   Master Secret     │  (Generated once per device pair)
│   (32 bytes)        │
└──────────┬──────────┘
           │
           ├──────────────────────────┐
           │ HKDF(info="encryption")  │ HKDF(info="signing")
           v                          v
┌─────────────────────┐    ┌─────────────────────┐
│   Encryption Key    │    │   Signing Key       │
│   (32 bytes)        │    │   (Ed25519 seed)    │
└─────────────────────┘    └─────────────────────┘
```

### 3. Device Identity

Each device has a unique identity:

```
struct DeviceIdentity {
    device_id: UUID,              // UUIDv5 from public key
    public_key: [u8; 32],         // Ed25519 public key
    private_key: [u8; 64],        // Ed25519 private key (local only)
}

function generate_device_identity() -> DeviceIdentity:
    // 1. Generate Ed25519 keypair
    (public_key, private_key) = ed25519_keygen()

    // 2. Derive device ID from public key
    device_id = UUIDv5(DEVICE_NAMESPACE, public_key)

    return DeviceIdentity { device_id, public_key, private_key }
```

### 4. Diff Generation

When the substrate changes, a diff is generated:

```
function generate_diff(
    old_state: StateRoot,
    new_state: StateRoot,
    graph: GraphStore
) -> CognitiveDiff:
    // 1. Get all entities at each state
    old_docs = graph.documents_at(old_state)
    new_docs = graph.documents_at(new_state)

    old_chunks = graph.chunks_at(old_state)
    new_chunks = graph.chunks_at(new_state)

    old_embs = graph.embeddings_at(old_state)
    new_embs = graph.embeddings_at(new_state)

    old_edges = graph.edges_at(old_state)
    new_edges = graph.edges_at(new_state)

    // 2. Compute additions/updates
    added_docs = new_docs.difference(old_docs)
    added_chunks = new_chunks.difference(old_chunks)
    added_embs = new_embs.difference(old_embs)
    added_edges = new_edges.difference(old_edges)

    // 3. Compute removals
    removed_docs = old_docs.keys().difference(new_docs.keys())
    removed_chunks = old_chunks.keys().difference(new_chunks.keys())
    removed_embs = old_embs.keys().difference(new_embs.keys())
    removed_edges = old_edges.keys().difference(new_edges.keys())

    // 4. Build diff
    return CognitiveDiff {
        documents: added_docs,
        chunks: added_chunks,
        embeddings: added_embs,
        edges: added_edges,

        removed_documents: removed_docs,
        removed_chunks: removed_chunks,
        removed_embeddings: removed_embs,
        removed_edges: removed_edges,

        prev_root: old_state.hash,
        new_root: new_state.hash,
        timestamp: now(),
        device_id: device.id,
        sequence: old_state.sequence + 1,
    }
```

### 5. Diff Serialization

Diffs are serialized using CBOR + zstd compression:

```
function serialize_diff(diff: CognitiveDiff) -> bytes:
    // 1. Encode to CBOR
    cbor_bytes = cbor::encode(diff)

    // 2. Compress with zstd
    compressed = zstd::compress(cbor_bytes, level=3)

    return compressed

function deserialize_diff(data: bytes) -> CognitiveDiff:
    // 1. Decompress
    decompressed = zstd::decompress(data)

    // 2. Decode CBOR
    return cbor::decode(decompressed)
```

#### Compression Ratios

| Content Type | Typical Ratio |
|--------------|---------------|
| Text chunks | 3:1 - 5:1 |
| Embeddings (i16) | 1.2:1 - 1.5:1 |
| Overall diff | 2:1 - 3:1 |

### 6. Encryption

Diffs are encrypted before transmission:

```
struct EncryptedDiff {
    nonce: [u8; 24],           // XChaCha20-Poly1305 nonce
    ciphertext: bytes,         // Encrypted diff
    tag: [u8; 16],             // Poly1305 authentication tag
    sender_id: UUID,           // Device that created diff
    sequence: u64,             // Diff sequence number
}

function encrypt_diff(diff: CognitiveDiff, key: [u8; 32]) -> EncryptedDiff:
    // 1. Serialize diff
    plaintext = serialize_diff(diff)

    // 2. Generate random nonce
    nonce = random_bytes(24)

    // 3. Encrypt with XChaCha20-Poly1305
    (ciphertext, tag) = xchacha20_poly1305_encrypt(plaintext, key, nonce)

    return EncryptedDiff {
        nonce,
        ciphertext,
        tag,
        sender_id: diff.device_id,
        sequence: diff.sequence,
    }

function decrypt_diff(enc: EncryptedDiff, key: [u8; 32]) -> CognitiveDiff:
    // 1. Decrypt
    plaintext = xchacha20_poly1305_decrypt(enc.ciphertext, key, enc.nonce, enc.tag)

    // 2. Deserialize
    return deserialize_diff(plaintext)
```

### 7. Signing

Each diff is signed by the originating device:

```
struct SignedDiff {
    encrypted_diff: EncryptedDiff,
    signature: [u8; 64],         // Ed25519 signature
    public_key: [u8; 32],        // Sender's public key
}

function sign_diff(enc: EncryptedDiff, private_key: [u8; 64]) -> SignedDiff:
    // Sign the encrypted payload
    message = enc.nonce || enc.ciphertext || enc.tag || enc.sender_id || enc.sequence.to_bytes()
    signature = ed25519_sign(message, private_key)

    return SignedDiff { encrypted_diff: enc, signature, public_key }

function verify_signature(signed: SignedDiff) -> bool:
    message = signed.encrypted_diff.nonce || ...  // Same as above
    return ed25519_verify(message, signed.signature, signed.public_key)
```

### 8. Relay Server Protocol

The relay server stores encrypted diffs without ability to decrypt.

#### API Endpoints

```
POST /push
    Header: X-Device-ID: <sender_device_id>
    Header: X-Recipient-ID: <recipient_device_id>
    Body: SignedDiff (binary)
    Response: { success: bool, sequence: u64 }

GET /pull
    Header: X-Device-ID: <recipient_device_id>
    Query: since=<sequence>
    Response: [SignedDiff, ...]

DELETE /acknowledge
    Header: X-Device-ID: <recipient_device_id>
    Body: { sequence: u64 }
    Response: { success: bool }
```

#### Server State

The relay server maintains minimal state:

```
struct DiffStore {
    // Encrypted diffs by (sender, recipient, sequence)
    diffs: Map<(UUID, UUID, u64), SignedDiff>,

    // Latest sequence per recipient
    latest_sequence: Map<UUID, u64>,

    // Retention policy: delete diffs older than 30 days
    retention_days: 30,
}
```

#### Blind Relay Properties

The relay server:

- **CANNOT** decrypt diff contents (no access to keys)
- **CANNOT** verify diff validity (cannot read plaintext)
- **CAN** see sender/recipient device IDs
- **CAN** see message sizes and timestamps
- **CAN** enforce rate limits and storage quotas

### 9. Push Flow (Desktop → Relay)

```
function push_to_relay(diff: CognitiveDiff, device: DeviceIdentity, relay_url: String):
    // 1. Derive shared key (established during pairing)
    key = load_shared_key(device.id)

    // 2. Encrypt diff
    encrypted = encrypt_diff(diff, key)

    // 3. Sign diff
    signed = sign_diff(encrypted, device.private_key)

    // 4. Push to relay
    response = http::post(
        relay_url + "/push",
        headers = {
            "X-Device-ID": device.device_id,
            "X-Recipient-ID": recipient_device_id,
        },
        body = signed.to_bytes()
    )

    if !response.success:
        raise SyncError("Push failed")

    // 5. Record sync state
    record_pushed(diff.sequence)
```

### 10. Pull Flow (Relay → Mobile)

```
function pull_from_relay(device: DeviceIdentity, relay_url: String, graph: GraphStore):
    // 1. Get last synced sequence
    last_sequence = get_last_synced_sequence(device.id)

    // 2. Pull new diffs
    response = http::get(
        relay_url + "/pull",
        headers = {
            "X-Device-ID": device.device_id,
        },
        query = { "since": last_sequence }
    )

    // 3. Process each diff
    for signed_diff in response.diffs:
        // 3a. Verify signature
        if !verify_signature(signed_diff):
            raise SyncError("Invalid signature")

        // 3b. Decrypt
        key = load_shared_key(signed_diff.encrypted_diff.sender_id)
        diff = decrypt_diff(signed_diff.encrypted_diff, key)

        // 3c. Verify state transition
        current_root = graph.get_latest_state_root()
        if diff.prev_root != current_root.hash:
            raise SyncError("State root mismatch")

        // 3d. Apply diff
        apply_diff(diff, graph)

        // 3e. Verify new state root
        computed_root = graph.compute_state_root()
        if computed_root != diff.new_root:
            raise SyncError("State verification failed")

        // 3f. Acknowledge receipt
        http::delete(
            relay_url + "/acknowledge",
            headers = { "X-Device-ID": device.device_id },
            body = { "sequence": diff.sequence }
        )

        // 3g. Record progress
        record_synced(diff.sequence)
```

### 11. Diff Application

```
function apply_diff(diff: CognitiveDiff, graph: GraphStore):
    tx = graph.begin_transaction()

    try:
        // 1. Remove deleted entities (order matters for foreign keys)
        for edge_key in diff.removed_edges:
            graph.delete_edge(edge_key)

        for emb_id in diff.removed_embeddings:
            graph.delete_embedding(emb_id)

        for chunk_id in diff.removed_chunks:
            graph.delete_chunk(chunk_id)

        for doc_id in diff.removed_documents:
            graph.delete_document(doc_id)

        // 2. Add/update entities (order matters for foreign keys)
        for doc in diff.documents:
            graph.upsert_document(doc)

        for chunk in diff.chunks:
            graph.upsert_chunk(chunk)

        for emb in diff.embeddings:
            graph.upsert_embedding(emb)

        for edge in diff.edges:
            graph.upsert_edge(edge)

        // 3. Rebuild Runtime Index (Transient)
        // This is a background task to update the local HNSW cache
        graph.rebuild_hnsw_index()

        // 4. Record state root
        new_root = StateRoot {
            hash: diff.new_root,
            parent_hash: Some(diff.prev_root),
            timestamp: diff.timestamp,
            device_id: diff.device_id,
            signature: [0; 64],  // Computed by verifier
            sequence: diff.sequence,
        }
        graph.insert_state_root(new_root)

        // 5. Commit
        graph.commit_transaction(tx)

    except error:
        graph.rollback_transaction(tx)
        raise error
```

### 12. Device Pairing

Before syncing, devices must establish a shared secret:

```
function pair_devices(device_a: DeviceIdentity, device_b: DeviceIdentity) -> SharedSecret:
    // 1. Exchange public keys (out-of-band: QR code, NFC, manual entry)
    // Device A displays: device_a.public_key (base64)
    // Device B scans and sends: device_b.public_key (base64)

    // 2. Derive shared secret using X25519 key agreement
    // Convert Ed25519 keys to X25519 for key agreement
    x25519_private_a = ed25519_to_x25519_private(device_a.private_key)
    x25519_public_b = ed25519_to_x25519_public(device_b.public_key)

    // 3. Compute shared secret
    shared = x25519_key_agreement(x25519_private_a, x25519_public_b)

    // 4. Derive encryption key
    encryption_key = hkdf_sha256(
        ikm = shared,
        info = b"cfs-sync-encryption-v1",
        salt = device_a.device_id || device_b.device_id,
        length = 32
    )

    // 5. Store paired device info
    store_paired_device(device_b.device_id, device_b.public_key, encryption_key)

    return SharedSecret { key: encryption_key }
```

### 13. Conflict Resolution

CFS V1 uses a simple conflict resolution strategy:

#### Write-Once Semantics

- Desktop is the sole writer
- Mobile is read-only
- No conflicts possible

#### Future: Multi-Writer Conflict Resolution

```
enum ConflictResolution {
    // Last-write-wins based on timestamp
    LastWriteWins,

    // Keep both versions, create conflict marker
    KeepBoth,

    // Use vector clocks for causality tracking
    VectorClocks,

    // CRDT-based automatic merge
    CRDT,
}
```

## Desired Properties

### 1. Confidentiality

**Property**: Relay server CANNOT read diff contents.

**Mechanism**: XChaCha20-Poly1305 authenticated encryption.

### 2. Integrity

**Property**: Tampering with diffs is detectable.

**Mechanism**: Poly1305 authentication tag + Ed25519 signature.

### 3. Authenticity

**Property**: Diffs can only be created by authorized devices.

**Mechanism**: Ed25519 signatures with device identity.

### 4. Forward Secrecy

**Property**: Compromising future keys doesn't compromise past messages.

**Mechanism**: Per-diff nonces prevent key reuse attacks.

### 5. Idempotency

**Property**: Applying the same diff twice has no additional effect.

**Mechanism**: Upsert semantics for all entities.

### 6. Ordering

**Property**: Diffs are applied in sequence order.

**Mechanism**: Sequence numbers + prev_root verification.

## Security Considerations

### Key Storage

Private keys MUST be stored securely:

- **iOS**: Keychain Services
- **Android**: Android Keystore
- **Desktop**: OS credential storage (Keychain, DPAPI, Secret Service)

### Nonce Uniqueness

XChaCha20 requires unique nonces:

```
// 24-byte nonce provides sufficient space for random generation
// Collision probability < 2^-64 after 2^48 messages
nonce = random_bytes(24)
```

### Relay Trust

The relay server is semi-trusted:

- **Trusted for**: Availability, delivery, rate limiting
- **Not trusted for**: Confidentiality, integrity, authenticity

### Metadata Privacy

The relay sees:

- Device IDs (pseudonymous)
- Message sizes
- Sync frequency
- IP addresses

Mitigations:

- Use Tor or VPN for network-level privacy
- Rotate device IDs periodically (future)

## Test Vectors

### Encryption Test

```
Key:       0x000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f
Nonce:     0x000102030405060708090a0b0c0d0e0f1011121314151617
Plaintext: b"Hello, CFS!"
Ciphertext: 0x7c3f5e... (implementation specific)
Tag:       0x8a2b... (implementation specific)
```

### Signature Test

```
Private Key: 0x9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60
Public Key:  0xd75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a
Message:     b"CFS diff"
Signature:   0xe5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b
```

## References

- [XChaCha20-Poly1305](https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-xchacha)
- [Ed25519 Signatures](https://ed25519.cr.yp.to/)
- [X25519 Key Agreement](https://www.rfc-editor.org/rfc/rfc7748)
- [CBOR Specification](https://www.rfc-editor.org/rfc/rfc8949)
- [zstd Compression](https://facebook.github.io/zstd/)
