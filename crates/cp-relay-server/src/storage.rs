//! Storage backend for the relay server
//!
//! Per CP-013: Stores encrypted diffs by (sender_device_id, target_device_id, sequence)
//! Also stores device registrations for the pairing protocol

use rusqlite::{params, Connection, OptionalExtension};
use std::sync::Mutex;

use crate::SignedDiffJson;

/// Device record for paired devices
pub struct DeviceRecord {
    pub device_id: String,
    pub public_key: String,
    pub display_name: Option<String>,
    pub created_at: i64,
    pub last_seen: Option<i64>,
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

        // Device registration table for pairing protocol
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

    /// Register a new device for the pairing protocol
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

        let device = stmt
            .query_row(params![device_id], |row| {
                Ok(DeviceRecord {
                    device_id: row.get(0)?,
                    public_key: row.get(1)?,
                    display_name: row.get(2)?,
                    created_at: row.get(3)?,
                    last_seen: row.get(4)?,
                })
            })
            .optional()?;

        Ok(device)
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
}
