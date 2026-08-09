use crate::config::{AppConfig, CredentialBackend};

use std::path::PathBuf;

use super::error::CredentialError;
use super::file::FileCredentialStore;
#[cfg(feature = "keychain")]
use super::keychain::KeychainCredentialStore;
use super::store::CredentialStore;

/// Create the appropriate credential store based on the app configuration.
pub fn create_credential_store(
    config: &AppConfig,
) -> Result<Box<dyn CredentialStore>, CredentialError> {
    match config.credentials.backend {
        #[cfg(feature = "keychain")]
        CredentialBackend::Keychain => Ok(Box::new(KeychainCredentialStore::new())),
        CredentialBackend::File => Ok(Box::new(FileCredentialStore::new(
            &config.credentials.file_path,
        )?)),
        CredentialBackend::Auto => {
            #[cfg(feature = "keychain")]
            {
                Ok(Box::new(KeychainThenEncryptedFile::new(
                    config.credentials.file_path.clone(),
                )))
            }
            #[cfg(not(feature = "keychain"))]
            {
                Ok(Box::new(FileCredentialStore::new(
                    &config.credentials.file_path,
                )?))
            }
        }
        #[cfg(not(feature = "keychain"))]
        CredentialBackend::Keychain => Err(CredentialError::BackendError(
            "keychain backend is not compiled into this build".to_string(),
        )),
    }
}

/// Uses the OS keychain whenever it works and lazily opens the encrypted file
/// only when the keychain is unavailable. The file cannot silently downgrade to
/// plaintext because opening it requires `AGILEPLUS_CREDENTIAL_KEY`.
#[cfg(feature = "keychain")]
struct KeychainThenEncryptedFile {
    keychain: KeychainCredentialStore,
    file_path: PathBuf,
}

#[cfg(feature = "keychain")]
impl KeychainThenEncryptedFile {
    fn new(file_path: PathBuf) -> Self {
        Self {
            keychain: KeychainCredentialStore::new(),
            file_path,
        }
    }

    fn with_fallback<T>(
        &self,
        operation: impl FnOnce(&FileCredentialStore) -> Result<T, CredentialError>,
    ) -> Result<T, CredentialError> {
        // Construct on demand so a healthy keychain never requires an
        // encryption key. The file store reloads its encrypted state on each
        // fallback operation, keeping this adapter stateless and fail-closed.
        operation(&FileCredentialStore::new(&self.file_path)?)
    }
}

#[cfg(feature = "keychain")]
impl CredentialStore for KeychainThenEncryptedFile {
    fn get(&self, service: &str, key: &str) -> Result<String, CredentialError> {
        self.keychain
            .get(service, key)
            .or_else(|error| match error {
                CredentialError::BackendError(_) => {
                    self.with_fallback(|file| file.get(service, key))
                }
                other => Err(other),
            })
    }

    fn set(&self, service: &str, key: &str, value: &str) -> Result<(), CredentialError> {
        self.keychain
            .set(service, key, value)
            .or_else(|error| match error {
                CredentialError::BackendError(_) => {
                    self.with_fallback(|file| file.set(service, key, value))
                }
                other => Err(other),
            })
    }

    fn delete(&self, service: &str, key: &str) -> Result<(), CredentialError> {
        self.keychain
            .delete(service, key)
            .or_else(|error| match error {
                CredentialError::BackendError(_) => {
                    self.with_fallback(|file| file.delete(service, key))
                }
                other => Err(other),
            })
    }

    fn list_keys(&self, service: &str) -> Result<Vec<String>, CredentialError> {
        self.keychain
            .list_keys(service)
            .or_else(|error| match error {
                CredentialError::BackendError(_) => {
                    self.with_fallback(|file| file.list_keys(service))
                }
                other => Err(other),
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::credentials::keys;

    #[test]
    fn configured_file_backend_uses_exact_path_and_survives_reload() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("configured.enc");
        let home = dir.path().join("home");
        std::fs::create_dir_all(home.join(".agileplus")).unwrap();
        std::fs::write(
            home.join(".agileplus/config.toml"),
            format!(
                "[credentials]\nbackend = \"file\"\nfile_path = \"{}\"\n",
                path.display()
            ),
        )
        .unwrap();
        let previous_home = std::env::var_os("HOME");
        unsafe { std::env::set_var("HOME", &home) };
        unsafe { std::env::set_var("AGILEPLUS_CREDENTIAL_KEY", "test-key") };
        let config = AppConfig::load().unwrap();
        let store = create_credential_store(&config).unwrap();
        store
            .set("agileplus", keys::API_KEYS, "sha256:test")
            .unwrap();
        assert!(path.is_file());
        assert_eq!(
            store.get("agileplus", keys::API_KEYS).unwrap(),
            "sha256:test"
        );
        unsafe { std::env::remove_var("AGILEPLUS_CREDENTIAL_KEY") };
        match previous_home {
            Some(value) => unsafe { std::env::set_var("HOME", value) },
            None => unsafe { std::env::remove_var("HOME") },
        }
    }

    #[test]
    fn malformed_app_config_fails_closed() {
        let dir = tempfile::tempdir().unwrap();
        let home = dir.path().join("home");
        std::fs::create_dir_all(home.join(".agileplus")).unwrap();
        std::fs::write(
            home.join(".agileplus/config.toml"),
            "[credentials\nbackend = \"file\"",
        )
        .unwrap();
        let previous_home = std::env::var_os("HOME");
        unsafe { std::env::set_var("HOME", &home) };
        assert!(AppConfig::load().is_err());
        match previous_home {
            Some(value) => unsafe { std::env::set_var("HOME", value) },
            None => unsafe { std::env::remove_var("HOME") },
        }
    }
}
