//! Verified loading policy for precompiled Metal libraries.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

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
}

/// A verified precompiled Metal library.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetallibArtifact {
    pub name: String,
    pub sha256: [u8; 32],
    pub bytes: Vec<u8>,
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn fixture(bytes: &[u8]) -> (PathBuf, ArtifactAllowlist) {
        let root = std::env::temp_dir().join(format!(
            "metal-runtime-artifact-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&root).unwrap();
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
        assert_eq!(artifact.bytes, b"precompiled-metal-library");
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
}
