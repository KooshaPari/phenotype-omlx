//! Verified loading policy for precompiled Metal libraries.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

/// Runtime policy. Production never accepts shader source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeMode {
    Production,
    Reference,
}

/// Immutable artifact hashes approved for production loading.
#[derive(Debug, Clone, Default)]
pub struct ArtifactAllowlist {
    hashes: HashMap<String, [u8; 32]>,
}

impl ArtifactAllowlist {
    pub fn new(entries: impl IntoIterator<Item = (String, [u8; 32])>) -> Self {
        Self {
            hashes: entries.into_iter().collect(),
        }
    }

    /// Parse the canonical artifact manifest emitted by the build tooling.
    pub fn from_manifest_json(bytes: &[u8]) -> Result<Self, ArtifactError> {
        let manifest: ArtifactManifest = serde_json::from_slice(bytes)
            .map_err(|source| ArtifactError::ManifestParse { source })?;
        let mut entries = Vec::with_capacity(manifest.artifacts.len());
        for entry in manifest.artifacts {
            validate_name(&entry.name)?;
            let digest = decode_digest(&entry.sha256)?;
            entries.push((entry.name, digest));
        }
        Ok(Self::new(entries))
    }
}

/// Stable on-disk representation of approved `.metallib` artifacts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactManifest {
    pub artifacts: Vec<ArtifactManifestEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactManifestEntry {
    pub name: String,
    pub sha256: String,
}

/// A verified precompiled Metal library.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetallibArtifact {
    name: String,
    sha256: [u8; 32],
    bytes: Vec<u8>,
}

impl MetallibArtifact {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn sha256(&self) -> &[u8; 32] {
        &self.sha256
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

#[derive(Debug, Error)]
pub enum ArtifactError {
    #[error("invalid metallib artifact name '{0}'")]
    InvalidName(String),
    #[error("metallib artifact '{0}' is not allowlisted")]
    NotAllowlisted(String),
    #[error("metallib artifact '{name}' hash mismatch")]
    HashMismatch { name: String },
    #[error("failed to read metallib artifact '{name}': {source}")]
    Read {
        name: String,
        #[source]
        source: std::io::Error,
    },
    #[error("invalid artifact manifest: {source}")]
    ManifestParse { source: serde_json::Error },
    #[error("invalid sha256 digest '{0}' in artifact manifest")]
    InvalidDigest(String),
}

/// Loader rooted at a trusted artifact directory.
#[derive(Debug, Clone)]
pub struct MetallibLoader {
    root: PathBuf,
    allowlist: ArtifactAllowlist,
}

impl MetallibLoader {
    pub fn new(root: impl Into<PathBuf>, allowlist: ArtifactAllowlist) -> Self {
        Self {
            root: root.into(),
            allowlist,
        }
    }

    /// Construct a loader directly from the canonical build manifest.
    pub fn from_manifest_json(
        root: impl Into<PathBuf>,
        manifest: &[u8],
    ) -> Result<Self, ArtifactError> {
        Ok(Self::new(
            root,
            ArtifactAllowlist::from_manifest_json(manifest)?,
        ))
    }

    pub fn load(&self, name: &str) -> Result<MetallibArtifact, ArtifactError> {
        validate_name(name)?;
        let expected = self
            .allowlist
            .hashes
            .get(name)
            .ok_or_else(|| ArtifactError::NotAllowlisted(name.to_owned()))?;
        let bytes = std::fs::read(self.root.join(name)).map_err(|source| ArtifactError::Read {
            name: name.to_owned(),
            source,
        })?;
        let actual: [u8; 32] = Sha256::digest(&bytes).into();
        if &actual != expected {
            return Err(ArtifactError::HashMismatch {
                name: name.to_owned(),
            });
        }
        Ok(MetallibArtifact {
            name: name.to_owned(),
            sha256: actual,
            bytes,
        })
    }
}

fn validate_name(name: &str) -> Result<(), ArtifactError> {
    let path = Path::new(name);
    let is_basename = path.file_name().and_then(|value| value.to_str()) == Some(name);
    if !is_basename || path.extension().and_then(|value| value.to_str()) != Some("metallib") {
        return Err(ArtifactError::InvalidName(name.to_owned()));
    }
    Ok(())
}

fn decode_digest(value: &str) -> Result<[u8; 32], ArtifactError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(ArtifactError::InvalidDigest(value.to_owned()));
    }
    let mut digest = [0u8; 32];
    for (index, slot) in digest.iter_mut().enumerate() {
        *slot = u8::from_str_radix(&value[index * 2..index * 2 + 2], 16)
            .map_err(|_| ArtifactError::InvalidDigest(value.to_owned()))?;
    }
    Ok(digest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    /// Monotonic per-process counter so concurrent test threads do not
    /// collide on the same temp-dir name even when their nanos coalesce.
    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn fixture(bytes: &[u8]) -> (PathBuf, ArtifactAllowlist) {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "metal-runtime-artifact-{}-{}-{seq}",
            std::process::id(),
            nanos
        ));
        // `create_dir_all` is idempotent — defence in depth against
        // any future name collision.
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("model.metallib"), bytes).unwrap();
        let digest = Sha256::digest(bytes).into();
        (
            root,
            ArtifactAllowlist::new([("model.metallib".into(), digest)]),
        )
    }

    #[test]
    fn loads_allowlisted_metallib_when_hash_matches() {
        let (root, allowlist) = fixture(b"precompiled-metal-library");
        let artifact = MetallibLoader::new(&root, allowlist)
            .load("model.metallib")
            .unwrap();
        assert_eq!(artifact.bytes(), b"precompiled-metal-library");
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rejects_modified_allowlisted_metallib() {
        let (root, allowlist) = fixture(b"approved");
        std::fs::write(root.join("model.metallib"), b"modified").unwrap();
        let error = MetallibLoader::new(&root, allowlist)
            .load("model.metallib")
            .unwrap_err();
        assert!(matches!(error, ArtifactError::HashMismatch { .. }));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rejects_unlisted_and_traversal_artifacts() {
        let (root, allowlist) = fixture(b"approved");
        let loader = MetallibLoader::new(&root, allowlist);
        assert!(matches!(
            loader.load("other.metallib"),
            Err(ArtifactError::NotAllowlisted(_))
        ));
        assert!(matches!(
            loader.load("../model.metallib"),
            Err(ArtifactError::InvalidName(_))
        ));
        assert!(matches!(
            loader.load("model.metal"),
            Err(ArtifactError::InvalidName(_))
        ));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn manifest_parser_accepts_canonical_digest_and_preserves_loader_contract() {
        let bytes = b"manifest-artifact";
        let digest: [u8; 32] = Sha256::digest(bytes).into();
        let hex = digest
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        let manifest = serde_json::json!({
            "artifacts": [{"name": "model.metallib", "sha256": hex}]
        });
        let allowlist =
            ArtifactAllowlist::from_manifest_json(&serde_json::to_vec(&manifest).unwrap()).unwrap();
        let (root, _) = fixture(bytes);
        let artifact = MetallibLoader::new(root.clone(), allowlist)
            .load("model.metallib")
            .unwrap();
        assert_eq!(artifact.sha256(), &digest);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn manifest_parser_rejects_non_sha256_digest() {
        let manifest = br#"{"artifacts":[{"name":"model.metallib","sha256":"00"}]}"#;
        assert!(matches!(
            ArtifactAllowlist::from_manifest_json(manifest),
            Err(ArtifactError::InvalidDigest(_))
        ));
    }
}
