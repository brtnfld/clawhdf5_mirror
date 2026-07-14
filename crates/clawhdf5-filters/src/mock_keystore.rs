//! Mock key management for P2.2 prototype.
//!
//! This module provides a `MockKeyStore` that reads a static Data Encryption Key (DEK)
//! from the environment variable `CLAW_DUMMY_DEK`. This is a prototype implementation
//! for testing the ChaCha20-Poly1305 AEAD filter; production key management (HKDF, KEK
//! wrapping, multi-party distribution) is deferred to Phase 3.
//!
//! # Security Warning
//!
//! **DO NOT USE IN PRODUCTION.** This mock implementation:
//! - Uses a single static key for all encryption
//! - Has no key derivation or rotation
//! - Reads the key from an environment variable (insecure)
//!
//! # Usage
//!
//! Set the environment variable before running:
//!
//! ```bash
//! export CLAW_DUMMY_DEK="0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
//! ```
//!
//! Or configure in `.cargo/config.toml`:
//!
//! ```toml
//! [env]
//! CLAW_DUMMY_DEK = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
//! ```
//!
//! # Spec Reference
//!
//! S2-D2-Yr2 P2.2 step 1: "Implement a MockKeyStore that derives a static 32-byte DEK
//! from the environment variable CLAW_DUMMY_DEK (hex-encoded)."

use core::fmt;

/// Size of the Data Encryption Key (DEK) in bytes.
/// ChaCha20-Poly1305 requires a 256-bit (32-byte) key.
pub const DEK_SIZE: usize = 32;

/// A 256-bit Data Encryption Key for ChaCha20-Poly1305.
pub type Dek = [u8; DEK_SIZE];

/// Error type for key store operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KeyStoreError {
    /// The `CLAW_DUMMY_DEK` environment variable is not set.
    EnvVarNotSet,
    /// The hex string could not be decoded.
    InvalidHex(String),
    /// The decoded key has the wrong length.
    InvalidKeyLength { expected: usize, actual: usize },
}

impl fmt::Display for KeyStoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KeyStoreError::EnvVarNotSet => {
                write!(f, "CLAW_DUMMY_DEK environment variable not set")
            }
            KeyStoreError::InvalidHex(msg) => {
                write!(f, "invalid hex encoding: {msg}")
            }
            KeyStoreError::InvalidKeyLength { expected, actual } => {
                write!(
                    f,
                    "invalid key length: expected {expected} bytes, got {actual}"
                )
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for KeyStoreError {}

/// Mock key store for prototype encryption testing.
///
/// Reads the DEK from the `CLAW_DUMMY_DEK` environment variable.
/// This is a placeholder until production key management is implemented in Phase 3.
#[derive(Debug, Clone)]
pub struct MockKeyStore {
    dek: Dek,
}

impl MockKeyStore {
    /// Environment variable name for the mock DEK.
    pub const ENV_VAR: &'static str = "CLAW_DUMMY_DEK";

    /// Create a new MockKeyStore by reading the DEK from the environment.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `CLAW_DUMMY_DEK` is not set
    /// - The value is not valid hexadecimal
    /// - The decoded key is not exactly 32 bytes
    ///
    /// # Example
    ///
    /// ```ignore
    /// std::env::set_var("CLAW_DUMMY_DEK", "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff");
    /// let keystore = MockKeyStore::from_env()?;
    /// let dek = keystore.dek();
    /// ```
    pub fn from_env() -> Result<Self, KeyStoreError> {
        let hex_str = std::env::var(Self::ENV_VAR).map_err(|_| KeyStoreError::EnvVarNotSet)?;

        let dek = Self::decode_hex_key(&hex_str)?;
        Ok(Self { dek })
    }

    /// Create a MockKeyStore from a hex-encoded key string.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The value is not valid hexadecimal
    /// - The decoded key is not exactly 32 bytes
    pub fn from_hex(hex_str: &str) -> Result<Self, KeyStoreError> {
        let dek = Self::decode_hex_key(hex_str)?;
        Ok(Self { dek })
    }

    /// Create a MockKeyStore from raw key bytes.
    ///
    /// # Panics
    ///
    /// Panics if the key is not exactly 32 bytes.
    #[must_use]
    pub fn from_bytes(key: &[u8]) -> Self {
        let dek: Dek = key.try_into().expect("key must be exactly 32 bytes");
        Self { dek }
    }

    /// Get the Data Encryption Key.
    #[must_use]
    pub fn dek(&self) -> &Dek {
        &self.dek
    }

    /// Decode a hex string to a 32-byte key.
    fn decode_hex_key(hex_str: &str) -> Result<Dek, KeyStoreError> {
        // Remove any whitespace or 0x prefix
        let hex_str = hex_str.trim().trim_start_matches("0x");

        // Decode hex to bytes
        let bytes = Self::hex_decode(hex_str)?;

        // Verify length
        if bytes.len() != DEK_SIZE {
            return Err(KeyStoreError::InvalidKeyLength {
                expected: DEK_SIZE,
                actual: bytes.len(),
            });
        }

        let mut dek = [0u8; DEK_SIZE];
        dek.copy_from_slice(&bytes);
        Ok(dek)
    }

    /// Decode a hex string to bytes (no external dependency).
    fn hex_decode(hex_str: &str) -> Result<Vec<u8>, KeyStoreError> {
        if !hex_str.len().is_multiple_of(2) {
            return Err(KeyStoreError::InvalidHex(
                "hex string must have even length".to_string(),
            ));
        }

        let mut bytes = Vec::with_capacity(hex_str.len() / 2);

        for i in (0..hex_str.len()).step_by(2) {
            let byte_str = &hex_str[i..i + 2];
            let byte = u8::from_str_radix(byte_str, 16).map_err(|e| {
                KeyStoreError::InvalidHex(format!("invalid hex byte '{}': {}", byte_str, e))
            })?;
            bytes.push(byte);
        }

        Ok(bytes)
    }
}

/// Get the mock DEK from the environment.
///
/// This is the simplified API matching the S2-D2-Yr2 spec signature.
///
/// # Panics
///
/// Panics if `CLAW_DUMMY_DEK` is not set or is invalid.
/// For production code, use `MockKeyStore::from_env()` which returns a `Result`.
///
/// # Example
///
/// ```ignore
/// let dek: [u8; 32] = mock_dek();
/// ```
#[must_use]
pub fn mock_dek() -> Dek {
    MockKeyStore::from_env()
        .expect("set CLAW_DUMMY_DEK for prototype")
        .dek
}

/// Get the mock DEK from the environment, returning a Result.
///
/// This is the non-panicking version of `mock_dek()`.
pub fn try_mock_dek() -> Result<Dek, KeyStoreError> {
    MockKeyStore::from_env().map(|ks| ks.dek)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_valid_hex() {
        let hex = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        let keystore = MockKeyStore::from_hex(hex).unwrap();
        let dek = keystore.dek();

        assert_eq!(dek.len(), 32);
        assert_eq!(dek[0], 0x01);
        assert_eq!(dek[1], 0x23);
        assert_eq!(dek[15], 0xef);
    }

    #[test]
    fn decode_uppercase_hex() {
        let hex = "0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF";
        let keystore = MockKeyStore::from_hex(hex).unwrap();
        assert_eq!(keystore.dek()[15], 0xef);
    }

    #[test]
    fn decode_with_0x_prefix() {
        let hex = "0x0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        let keystore = MockKeyStore::from_hex(hex).unwrap();
        assert_eq!(keystore.dek().len(), 32);
    }

    #[test]
    fn decode_with_whitespace() {
        let hex = "  0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef  ";
        let keystore = MockKeyStore::from_hex(hex).unwrap();
        assert_eq!(keystore.dek().len(), 32);
    }

    #[test]
    fn reject_wrong_length() {
        let hex = "0123456789abcdef"; // Only 8 bytes
        let result = MockKeyStore::from_hex(hex);
        assert!(matches!(
            result,
            Err(KeyStoreError::InvalidKeyLength {
                expected: 32,
                actual: 8
            })
        ));
    }

    #[test]
    fn reject_invalid_hex_char() {
        let hex = "0123456789abcdefGGGG456789abcdef0123456789abcdef0123456789abcdef";
        let result = MockKeyStore::from_hex(hex);
        assert!(matches!(result, Err(KeyStoreError::InvalidHex(_))));
    }

    #[test]
    fn reject_odd_length_hex() {
        let hex = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcde"; // 63 chars
        let result = MockKeyStore::from_hex(hex);
        assert!(matches!(result, Err(KeyStoreError::InvalidHex(_))));
    }

    #[test]
    fn from_bytes_works() {
        let key = [0x42u8; 32];
        let keystore = MockKeyStore::from_bytes(&key);
        assert_eq!(keystore.dek(), &key);
    }

    #[test]
    fn from_env_reads_variable() {
        // Set the environment variable for this test
        // SAFETY: This test runs single-threaded and we clean up after
        unsafe {
            std::env::set_var(
                MockKeyStore::ENV_VAR,
                "deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
            );
        }

        let keystore = MockKeyStore::from_env().unwrap();
        assert_eq!(keystore.dek()[0], 0xde);
        assert_eq!(keystore.dek()[1], 0xad);
        assert_eq!(keystore.dek()[2], 0xbe);
        assert_eq!(keystore.dek()[3], 0xef);

        // Clean up
        // SAFETY: This test runs single-threaded
        unsafe {
            std::env::remove_var(MockKeyStore::ENV_VAR);
        }
    }

    #[test]
    fn error_display() {
        let err = KeyStoreError::EnvVarNotSet;
        assert_eq!(
            err.to_string(),
            "CLAW_DUMMY_DEK environment variable not set"
        );

        let err = KeyStoreError::InvalidKeyLength {
            expected: 32,
            actual: 16,
        };
        assert_eq!(
            err.to_string(),
            "invalid key length: expected 32 bytes, got 16"
        );
    }
}
