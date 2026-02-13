//! Cryptographic operations for CFS

use cfs_core::{CfsError, CognitiveDiff, Result};
use chacha20poly1305::{
    aead::{Aead, KeyInit},
    XChaCha20Poly1305, XNonce,
};
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use rand::RngCore;

use crate::{serialize_diff, deserialize_diff, EncryptedPayload};

/// Cryptographic engine for encrypting and signing diffs
pub struct CryptoEngine {
    /// Symmetric encryption key
    symmetric_key: [u8; 32],
    /// Ed25519 signing key
    signing_key: SigningKey,
}

impl CryptoEngine {
    /// Create a new crypto engine with random keys
    pub fn new() -> Self {
        let mut rng = rand::thread_rng();
        
        let mut symmetric_key = [0u8; 32];
        rng.fill_bytes(&mut symmetric_key);
        
        let mut signing_seed = [0u8; 32];
        rng.fill_bytes(&mut signing_seed);
        let signing_key = SigningKey::from_bytes(&signing_seed);
        
        Self {
            symmetric_key,
            signing_key,
        }
    }

    /// Create from existing keys
    pub fn from_keys(symmetric_key: [u8; 32], signing_key_bytes: [u8; 32]) -> Self {
        Self {
            symmetric_key,
            signing_key: SigningKey::from_bytes(&signing_key_bytes),
        }
    }

    /// Create from seed (derives keys deterministically)
    pub fn new_with_seed(seed: [u8; 32]) -> Self {
        // Simple derivation for MVP:
        // symmetric = hash(seed || "enc")
        // signing = hash(seed || "sign")
        
        let mut hasher = blake3::Hasher::new();
        hasher.update(&seed);
        hasher.update(b"enc");
        let symmetric_key = *hasher.finalize().as_bytes();
        
        let mut hasher = blake3::Hasher::new();
        hasher.update(&seed);
        hasher.update(b"sign");
        let signing_seed = *hasher.finalize().as_bytes();
        
        Self {
            symmetric_key,
            signing_key: SigningKey::from_bytes(&signing_seed),
        }
    }

    /// Get the public key for verification
    pub fn public_key(&self) -> [u8; 32] {
        self.signing_key.verifying_key().to_bytes()
    }

    /// Encrypt and sign a cognitive diff
    pub fn encrypt_diff(&self, diff: &CognitiveDiff) -> Result<EncryptedPayload> {
        // Serialize and compress
        let plaintext = serialize_diff(diff)?;
        
        // Generate random nonce
        let mut nonce_bytes = [0u8; 24];
        rand::thread_rng().fill_bytes(&mut nonce_bytes);
        let nonce = XNonce::from_slice(&nonce_bytes);
        
        // Encrypt
        let cipher = XChaCha20Poly1305::new_from_slice(&self.symmetric_key)
            .map_err(|e| CfsError::Crypto(e.to_string()))?;
        
        let ciphertext = cipher
            .encrypt(nonce, plaintext.as_ref())
            .map_err(|e| CfsError::Crypto(e.to_string()))?;
        
        // Sign the ciphertext
        let signature = self.signing_key.sign(&ciphertext);
        
        Ok(EncryptedPayload {
            ciphertext,
            nonce: nonce_bytes,
            signature: signature.to_bytes(),
            public_key: self.public_key(),
        })
    }

    /// Sign arbitrary data
    pub fn sign(&self, data: &[u8]) -> Signature {
        self.signing_key.sign(data)
    }

    /// Decrypt and verify a payload
    pub fn decrypt_diff(&self, payload: &EncryptedPayload) -> Result<CognitiveDiff> {
        // Verify signature
        let verifying_key = VerifyingKey::from_bytes(&payload.public_key)
            .map_err(|e| CfsError::Crypto(e.to_string()))?;
        
        let signature = Signature::from_bytes(&payload.signature);
        
        verifying_key
            .verify(&payload.ciphertext, &signature)
            .map_err(|_| CfsError::Verification("Invalid signature".into()))?;
        
        // Decrypt
        let nonce = XNonce::from_slice(&payload.nonce);
        let cipher = XChaCha20Poly1305::new_from_slice(&self.symmetric_key)
            .map_err(|e| CfsError::Crypto(e.to_string()))?;
        
        let plaintext = cipher
            .decrypt(nonce, payload.ciphertext.as_ref())
            .map_err(|_| CfsError::Crypto("Decryption failed".into()))?;
        
        // Deserialize
        deserialize_diff(&plaintext)
    }
}

impl Default for CryptoEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cfs_core::Hlc;
    use uuid::Uuid;

    #[test]
    fn test_encrypt_decrypt_roundtrip() {
        let engine = CryptoEngine::new();
        let diff = CognitiveDiff::empty([0u8; 32], Uuid::new_v4(), 0, Hlc::new(1000, [0u8; 16]));
        
        let encrypted = engine.encrypt_diff(&diff).unwrap();
        let decrypted = engine.decrypt_diff(&encrypted).unwrap();
        
        assert_eq!(diff.metadata.device_id, decrypted.metadata.device_id);
    }

    #[test]
    fn test_signature_verification() {
        let engine = CryptoEngine::new();
        let diff = CognitiveDiff::empty([0u8; 32], Uuid::new_v4(), 0, Hlc::new(1000, [0u8; 16]));
        
        let mut encrypted = engine.encrypt_diff(&diff).unwrap();
        
        // Tamper with signature
        encrypted.signature[0] ^= 0xFF;
        
        assert!(engine.decrypt_diff(&encrypted).is_err());
    }
}
