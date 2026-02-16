//! Storage backend for the relay server
//!
//! Per CP-013: Stores encrypted diffs by (sender_device_id, target_device_id, sequence)

use rusqlite::{params, Connection, OptionalExtension};
use std::sync::Mutex;

use crate::SignedDiffJson;

/// Device record
pub struct DeviceRecord {
    pub device_id: String,
    pub public_key: String,
    pub display_name: Option<String>,
    pub created_at: i64,
    pub last_seen: Option<i64>,
}

/// Pairing record
pub struct PairingRecord {
    pub id: i64,
    pub device_a: String,
    pub device_b: String,
    pub shared_secret_hash: String,
    pub created_at: i64,
}

/// Pending pairing record
pub struct PendingPairing {
    pub pairing_id: String,
    pub initiator_device_id: String,
    pub initiator_public_key: String,
    pub initiator_display_name: Option<String>,
    pub responder_device_id: String,
    pub responder_public_key: String,
    pub responder_display_name: Option<String>,
    pub expires_at: i64,
}

/// SQLite-based storage for encrypted diffs
/// Uses a Mutex to ensure thread safety with Axum's async handlers
pub struct Storage {
    conn: Mutex<Connection>,
}

// Safety: We use Mutex to ensure only one thread accesses Connection at a time
unsafe impl Send for Storage {}
unsafe impl Sync for Storage {}

impl Storage {
    /// Open or create storage at the given path
    pub fn open(path: &str) -> Result<Self, rusqlite::Error> {
        let conn = if path == ":memory:" {
            Connection::open_in_memory()?
        } else {
            Connection::open(path)?
        };

        // Per CP-013: DiffStore schema
        conn.execute(
            "CREATE TABLE IF NOT EXISTS diffs (
                sender_device_id TEXT NOT NULL,
                target_device_id TEXT NOT NULL,
                sequence INTEGER NOT NULL,
                ciphertext TEXT NOT NULL,
                nonce TEXT NOT NULL,
                signature TEXT NOT NULL,
                sender_public_key TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                acknowledged INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (sender_device_id, target_device_id, sequence)
            )",
            [],
        )?;

        // Index for efficient pull queries
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_diffs_target_sequence
             ON diffs(target_device_id, sequence)",
            [],
        )?;

        // Devices table
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

        // Pairings table
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

        // Pending pairings table
        conn.execute(
            "CREATE TABLE IF NOT EXISTS pending_pairings (
                pairing_id TEXT PRIMARY KEY,
                initiator_device_id TEXT NOT NULL,
                initiator_public_key TEXT NOT NULL,
                initiator_display_name TEXT,
                responder_device_id TEXT NOT NULL DEFAULT '',
                responder_public_key TEXT NOT NULL DEFAULT '',
                responder_display_name TEXT,
                expires_at INTEGER NOT NULL
            )",
            [],
        )?;

        // Devices table for device pairing
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

        // Pairings table for device-to-device trust
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

        // Pending pairings for pairing protocol
        conn.execute(
            "CREATE TABLE IF NOT EXISTS pending_pairings (
                pairing_id TEXT PRIMARY KEY,
                initiator_device_id TEXT NOT NULL,
                initiator_public_key TEXT NOT NULL,
                initiator_display_name TEXT,
                responder_device_id TEXT NOT NULL DEFAULT '',
                responder_public_key TEXT NOT NULL DEFAULT '',
                responder_display_name TEXT,
                expires_at INTEGER NOT NULL
            )",
            [],
        )?;

        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    /// Store a diff
    ///
    /// Returns (sequence, timestamp) on success
    pub fn store_diff(
        &self,
        sender_device_id: &str,
        target_device_id: &str,
        body: &[u8],
    ) -> Result<(u64, i64), rusqlite::Error> {
        // Parse the body as JSON
        let diff_json: SignedDiffJson = serde_json::from_slice(body)
            .map_err(|e| rusqlite::Error::InvalidParameterName(format!("JSON parse error: {}", e)))?;

        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        let sequence = diff_json.sequence;

        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT OR REPLACE INTO diffs
             (sender_device_id, target_device_id, sequence, ciphertext, nonce, signature, sender_public_key, timestamp)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                sender_device_id,
                target_device_id,
                sequence,
                diff_json.ciphertext,
                diff_json.nonce,
                diff_json.signature,
                diff_json.sender_public_key,
                timestamp
            ],
        )?;

        Ok((sequence, timestamp))
    }

    /// Get diffs for a recipient since a sequence number
    pub fn get_diffs_since(
        &self,
        target_device_id: &str,
        since: u64,
    ) -> Result<Vec<SignedDiffJson>, rusqlite::Error> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT sender_device_id, target_device_id, sequence, ciphertext, nonce, signature, sender_public_key, timestamp
             FROM diffs
             WHERE target_device_id = ?1 AND sequence > ?2 AND acknowledged = 0
             ORDER BY sequence ASC"
        )?;

        let diffs = stmt
            .query_map(params![target_device_id, since], |row| {
                Ok(SignedDiffJson {
                    sender_device_id: row.get(0)?,
                    target_device_id: row.get(1)?,
                    sequence: row.get(2)?,
                    ciphertext: row.get(3)?,
                    nonce: row.get(4)?,
                    signature: row.get(5)?,
                    sender_public_key: row.get(6)?,
                    timestamp: row.get(7)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(diffs)
    }

    /// Acknowledge diffs up to a sequence number
    pub fn acknowledge_diffs(
        &self,
        target_device_id: &str,
        sequence: u64,
    ) -> Result<(), rusqlite::Error> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "UPDATE diffs SET acknowledged = 1
             WHERE target_device_id = ?1 AND sequence <= ?2",
            params![target_device_id, sequence],
        )?;
        Ok(())
    }

    /// Delete acknowledged diffs older than retention period
    /// Returns number of deleted rows
    #[allow(dead_code)]
    pub fn cleanup_old_diffs(&self, retention_days: i64) -> Result<usize, rusqlite::Error> {
        let cutoff = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64
            - (retention_days * 24 * 60 * 60);

        let conn = self.conn.lock().unwrap();
        let deleted = conn.execute(
            "DELETE FROM diffs WHERE acknowledged = 1 AND timestamp < ?1",
            params![cutoff],
        )?;

        Ok(deleted)
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

    /// Store a pending pairing
    pub fn store_pending_pairing(
        &self,
        pairing_id: &str,
        initiator_device_id: &str,
        initiator_public_key: &str,
        initiator_display_name: Option<&str>,
        expires_at: i64,
    ) -> Result<(), rusqlite::Error> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO pending_pairings
             (pairing_id, initiator_device_id, initiator_public_key, initiator_display_name, responder_device_id, responder_public_key, expires_at)
             VALUES (?1, ?2, ?3, ?4, '', '', ?5)",
            params![pairing_id, initiator_device_id, initiator_public_key, initiator_display_name, expires_at],
        )?;
        Ok(())
    }

    /// Update pending pairing with responder
    pub fn update_pending_pairing(
        &self,
        pairing_id: &str,
        responder_device_id: &str,
        responder_public_key: &str,
        responder_display_name: Option<&str>,
    ) -> Result<(), rusqlite::Error> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "UPDATE pending_pairings
             SET responder_device_id = ?1, responder_public_key = ?2, responder_display_name = ?3
             WHERE pairing_id = ?4",
            params![responder_device_id, responder_public_key, responder_display_name, pairing_id],
        )?;
        Ok(())
    }

    /// Get pending pairing
    pub fn get_pending_pairing(&self, pairing_id: &str) -> Result<Option<PendingPairing>, rusqlite::Error> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT pairing_id, initiator_device_id, initiator_public_key, initiator_display_name,
                    responder_device_id, responder_public_key, responder_display_name, expires_at
             FROM pending_pairings WHERE pairing_id = ?1"
        )?;

        let pairing = stmt.query_row(params![pairing_id], |row| {
            Ok(PendingPairing {
                pairing_id: row.get(0)?,
                initiator_device_id: row.get(1)?,
                initiator_public_key: row.get(2)?,
                initiator_display_name: row.get(3)?,
                responder_device_id: row.get(4)?,
                responder_public_key: row.get(5)?,
                responder_display_name: row.get(6)?,
                expires_at: row.get(7)?,
            })
        }).optional()?;

        Ok(pairing)
    }

    /// Delete pending pairing
    pub fn delete_pending_pairing(&self, pairing_id: &str) -> Result<(), rusqlite::Error> {
        let conn = self.conn.lock().unwrap();
        conn.execute("DELETE FROM pending_pairings WHERE pairing_id = ?1", params![pairing_id])?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_store_and_pull_diff() {
        let storage = Storage::open(":memory:").unwrap();

        // Create a test diff
        let diff_json = SignedDiffJson {
            sequence: 1,
            sender_device_id: "sender123".to_string(),
            target_device_id: "recipient456".to_string(),
            ciphertext: "encrypted_data".to_string(),
            nonce: "nonce123".to_string(),
            signature: "sig123".to_string(),
            sender_public_key: "pubkey123".to_string(),
            timestamp: 1000,
        };
        let body = serde_json::to_vec(&diff_json).unwrap();

        // Store the diff
        let (seq, ts) = storage.store_diff("sender123", "recipient456", &body).unwrap();
        assert_eq!(seq, 1);
        assert!(ts > 0);

        // Pull diffs since sequence 0
        let diffs = storage.get_diffs_since("recipient456", 0).unwrap();
        assert_eq!(diffs.len(), 1);
        assert_eq!(diffs[0].sequence, 1);
        assert_eq!(diffs[0].ciphertext, "encrypted_data");

        // Pull diffs since sequence 1 (should be empty)
        let diffs = storage.get_diffs_since("recipient456", 1).unwrap();
        assert_eq!(diffs.len(), 0);
    }

    #[test]
    fn test_acknowledge_diffs() {
        let storage = Storage::open(":memory:").unwrap();

        // Store multiple diffs
        for seq in 1..=3 {
            let diff_json = SignedDiffJson {
                sequence: seq,
                sender_device_id: "sender123".to_string(),
                target_device_id: "recipient456".to_string(),
                ciphertext: format!("data_{}", seq),
                nonce: "nonce".to_string(),
                signature: "sig".to_string(),
                sender_public_key: "pubkey".to_string(),
                timestamp: 1000 + seq as i64,
            };
            let body = serde_json::to_vec(&diff_json).unwrap();
            storage.store_diff("sender123", "recipient456", &body).unwrap();
        }

        // Acknowledge up to sequence 2
        storage.acknowledge_diffs("recipient456", 2).unwrap();

        // Should still get sequence 3 (not acknowledged)
        let diffs = storage.get_diffs_since("recipient456", 0).unwrap();
        assert_eq!(diffs.len(), 1);
        assert_eq!(diffs[0].sequence, 3);
    }

    #[test]
    fn test_device_registration() {
        let storage = Storage::open(":memory:").unwrap();

        // Register a device
        storage.register_device("device123", "pubkey_abc", "MacBook Pro").unwrap();

        // Verify device exists
        let device = storage.get_device("device123").unwrap();
        assert!(device.is_some());
        let device = device.unwrap();
        assert_eq!(device.device_id, "device123");
        assert_eq!(device.display_name, Some("MacBook Pro".to_string()));
    }

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
        assert!(pairings[0].device_b == "device_b" || pairings[0].device_a == "device_b");
    }
}
