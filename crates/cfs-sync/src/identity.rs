//! Device identity management for CFS
//!
//! Per CFS-013: Provides device identity generation and pairing via X25519 key agreement.

use cfs_core::{CfsError, Result};
use ed25519_dalek::{SigningKey, VerifyingKey, Signer, Signature};
use hkdf::Hkdf;
use rand::RngCore;
use sha2::Sha256;
use x25519_dalek::{PublicKey as X25519PublicKey, StaticSecret};
use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;

/// Device identity containing signing keys and derived key agreement keys
#[derive(Clone)]
pub struct DeviceIdentity {
    /// Device ID (BLAKE3-16 of Ed25519 public key)
    pub device_id: [u8; 16],

    /// Ed25519 public key for signatures
    pub public_key: [u8; 32],

    /// Ed25519 signing key (private)
    signing_key: SigningKey,

    /// X25519 static secret for key agreement (derived from Ed25519 key)
    x25519_secret: StaticSecret,

    /// X25519 public key
    x25519_public: X25519PublicKey,
}

impl DeviceIdentity {
    /// Generate a new random device identity
    pub fn generate() -> Self {
        let mut rng = rand::thread_rng();
        let mut seed = [0u8; 32];
        rng.fill_bytes(&mut seed);
        Self::from_seed(seed)
    }

    /// Create device identity from a seed (deterministic)
    pub fn from_seed(seed: [u8; 32]) -> Self {
        // Ed25519 signing key from seed
        let signing_key = SigningKey::from_bytes(&seed);
        let verifying_key = signing_key.verifying_key();
        let public_key = verifying_key.to_bytes();

        // Device ID: BLAKE3-16 of public key
        let mut device_id = [0u8; 16];
        device_id.copy_from_slice(&blake3::hash(&public_key).as_bytes()[0..16]);

        // Derive X25519 key from Ed25519 seed
        // Using HKDF to derive a separate key for X25519
        let hk = Hkdf::<Sha256>::new(None, &seed);
        let mut x25519_seed = [0u8; 32];
        hk.expand(b"cfs-x25519-key", &mut x25519_seed)
            .expect("HKDF expand failed");

        let x25519_secret = StaticSecret::from(x25519_seed);
        let x25519_public = X25519PublicKey::from(&x25519_secret);

        Self {
            device_id,
            public_key,
            signing_key,
            x25519_secret,
            x25519_public,
        }
    }

    /// Sign data with this device's Ed25519 key
    pub fn sign(&self, data: &[u8]) -> [u8; 64] {
        let signature = self.signing_key.sign(data);
        signature.to_bytes()
    }

    /// Get the X25519 public key for key agreement
    pub fn x25519_public_key(&self) -> [u8; 32] {
        self.x25519_public.to_bytes()
    }

    /// Perform key agreement with a remote public key to derive a shared secret
    pub fn agree(&self, remote_x25519_public: &[u8; 32]) -> [u8; 32] {
        let remote_key = X25519PublicKey::from(*remote_x25519_public);
        let shared_secret = self.x25519_secret.diffie_hellman(&remote_key);
        *shared_secret.as_bytes()
    }

    /// Pair with a remote device to create a PairedDevice
    pub fn pair_with(&self, remote_public_key: &[u8; 32], remote_x25519_public: &[u8; 32]) -> Result<PairedDevice> {
        // Compute remote device ID
        let mut remote_device_id = [0u8; 16];
        remote_device_id.copy_from_slice(&blake3::hash(remote_public_key).as_bytes()[0..16]);

        // Derive shared encryption key via HKDF
        let shared_secret = self.agree(remote_x25519_public);

        // Sort device IDs to ensure both sides derive the same key
        let (id_a, id_b) = if self.device_id < remote_device_id {
            (self.device_id, remote_device_id)
        } else {
            (remote_device_id, self.device_id)
        };

        let mut info = Vec::with_capacity(32);
        info.extend_from_slice(&id_a);
        info.extend_from_slice(&id_b);

        let hk = Hkdf::<Sha256>::new(None, &shared_secret);
        let mut encryption_key = [0u8; 32];
        hk.expand(&info, &mut encryption_key)
            .map_err(|_| CfsError::Crypto("HKDF expand failed".into()))?;

        Ok(PairedDevice {
            device_id: remote_device_id,
            public_key: *remote_public_key,
            x25519_public_key: *remote_x25519_public,
            encryption_key,
            last_synced_seq: 0,
        })
    }

    /// Export the identity seed (for backup/recovery)
    pub fn export_seed(&self) -> [u8; 32] {
        self.signing_key.to_bytes()
    }
}

impl std::fmt::Debug for DeviceIdentity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceIdentity")
            .field("device_id", &hex::encode(self.device_id))
            .field("public_key", &hex::encode(self.public_key))
            .finish()
    }
}

/// A paired remote device with derived encryption key
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairedDevice {
    /// Remote device ID
    pub device_id: [u8; 16],

    /// Remote Ed25519 public key
    pub public_key: [u8; 32],

    /// Remote X25519 public key
    pub x25519_public_key: [u8; 32],

    /// Derived encryption key for this pair
    pub encryption_key: [u8; 32],

    /// Last synced sequence number
    pub last_synced_seq: u64,
}

impl PairedDevice {
    /// Update the last synced sequence number
    pub fn update_last_synced(&mut self, seq: u64) {
        self.last_synced_seq = seq;
    }

    /// Get device ID as hex string
    pub fn device_id_hex(&self) -> String {
        hex::encode(self.device_id)
    }
}

/// Pairing request containing public keys for key agreement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairingRequest {
    /// Sender's device ID
    pub device_id: [u8; 16],

    /// Sender's Ed25519 public key
    pub public_key: [u8; 32],

    /// Sender's X25519 public key for key agreement
    pub x25519_public_key: [u8; 32],

    /// Optional human-readable device name
    pub device_name: Option<String>,
}

impl PairingRequest {
    /// Create a pairing request from a device identity
    pub fn from_identity(identity: &DeviceIdentity, device_name: Option<String>) -> Self {
        Self {
            device_id: identity.device_id,
            public_key: identity.public_key,
            x25519_public_key: identity.x25519_public_key(),
            device_name,
        }
    }
}

/// Pairing confirmation containing mutual verification data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairingConfirmation {
    /// The pairing request being confirmed
    pub request: PairingRequest,

    /// Signature over the request by the confirming device
    #[serde(with = "BigArray")]
    pub signature: [u8; 64],

    /// Confirming device's public key
    pub confirmer_public_key: [u8; 32],
}

impl PairingConfirmation {
    /// Create a pairing confirmation
    pub fn create(identity: &DeviceIdentity, request: &PairingRequest) -> Self {
        // Sign the request data
        let mut data = Vec::new();
        data.extend_from_slice(&request.device_id);
        data.extend_from_slice(&request.public_key);
        data.extend_from_slice(&request.x25519_public_key);

        let signature = identity.sign(&data);

        Self {
            request: request.clone(),
            signature,
            confirmer_public_key: identity.public_key,
        }
    }

    /// Verify the confirmation signature
    pub fn verify(&self) -> Result<()> {
        let verifying_key = VerifyingKey::from_bytes(&self.confirmer_public_key)
            .map_err(|e| CfsError::Crypto(format!("Invalid public key: {}", e)))?;

        let mut data = Vec::new();
        data.extend_from_slice(&self.request.device_id);
        data.extend_from_slice(&self.request.public_key);
        data.extend_from_slice(&self.request.x25519_public_key);

        let signature = Signature::from_bytes(&self.signature);

        verifying_key
            .verify_strict(&data, &signature)
            .map_err(|_| CfsError::Verification("Invalid pairing confirmation signature".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_identity_generation() {
        let id1 = DeviceIdentity::generate();
        let id2 = DeviceIdentity::generate();

        // Different devices should have different IDs
        assert_ne!(id1.device_id, id2.device_id);
        assert_ne!(id1.public_key, id2.public_key);
    }

    #[test]
    fn test_device_identity_from_seed() {
        let seed = [42u8; 32];
        let id1 = DeviceIdentity::from_seed(seed);
        let id2 = DeviceIdentity::from_seed(seed);

        // Same seed should produce same identity
        assert_eq!(id1.device_id, id2.device_id);
        assert_eq!(id1.public_key, id2.public_key);
    }

    #[test]
    fn test_device_pairing_symmetric() {
        let alice = DeviceIdentity::generate();
        let bob = DeviceIdentity::generate();

        // Alice pairs with Bob
        let alice_view_of_bob = alice
            .pair_with(&bob.public_key, &bob.x25519_public_key())
            .unwrap();

        // Bob pairs with Alice
        let bob_view_of_alice = bob
            .pair_with(&alice.public_key, &alice.x25519_public_key())
            .unwrap();

        // Both should derive the same encryption key
        assert_eq!(alice_view_of_bob.encryption_key, bob_view_of_alice.encryption_key);
    }

    #[test]
    fn test_signing_and_verification() {
        let identity = DeviceIdentity::generate();
        let data = b"test message";

        let signature = identity.sign(data);

        // Verify using ed25519-dalek
        let verifying_key = VerifyingKey::from_bytes(&identity.public_key).unwrap();
        let sig = Signature::from_bytes(&signature);
        assert!(verifying_key.verify_strict(data, &sig).is_ok());
    }

    #[test]
    fn test_pairing_request_and_confirmation() {
        let alice = DeviceIdentity::generate();
        let bob = DeviceIdentity::generate();

        // Alice creates a pairing request
        let request = PairingRequest::from_identity(&alice, Some("Alice's Phone".into()));

        // Bob confirms the request
        let confirmation = PairingConfirmation::create(&bob, &request);

        // Verification should succeed
        assert!(confirmation.verify().is_ok());
    }
}
