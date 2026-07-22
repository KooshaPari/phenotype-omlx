//! Dataset provenance: explicit source, revision, split, and content-hash metadata
//! for every loaded evaluation dataset. Provenance is required so that scored
//! results are reproducible and attributable to the exact dataset bytes that
//! were evaluated.
//!
//! The values here are intentionally explicit: `source` records where the
//! dataset was obtained from, `source_revision` pins the upstream version,
//! `split` records which evaluation split was loaded (e.g. `test`, `dev`,
//! `diamond`), and `sha256` is the deterministic content hash of the loaded
//! bytes. `task_count` is computed at load time so reports can cross-check the
//! declared count against the actual loader output.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::Path;

/// SHA-256 content hash of the loaded dataset bytes, rendered as lowercase hex.
pub type Sha256 = String;

/// Provenance metadata attached to a [`crate::Dataset`].
///
/// All fields are required: there is no implicit provenance. Callers that
/// construct provenance by hand must populate every field; loaders compute
/// `sha256` from the bytes they read and `task_count` after parsing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DatasetProvenance {
    /// Where the dataset was obtained (URL, path, or filesystem location).
    pub source: String,
    /// Upstream revision identifier (commit SHA, tag, or version string).
    pub source_revision: String,
    /// Dataset split name (e.g. `test`, `dev`, `diamond`).
    pub split: String,
    /// Hex-encoded SHA-256 of the loaded bytes.
    pub sha256: Sha256,
    /// Number of tasks produced by the loader. Captured at load time so
    /// downstream reports can detect drift between declared and actual size.
    pub task_count: usize,
}

impl DatasetProvenance {
    /// Build a new provenance record with the given source, revision, split,
    /// content bytes, and task count. The SHA-256 is computed from `bytes`
    /// deterministically.
    pub fn new(
        source: impl Into<String>,
        source_revision: impl Into<String>,
        split: impl Into<String>,
        bytes: &[u8],
        task_count: usize,
    ) -> Self {
        Self {
            source: source.into(),
            source_revision: source_revision.into(),
            split: split.into(),
            sha256: sha256_hex(bytes),
            task_count,
        }
    }

    /// Build provenance from a filesystem path. The path is recorded as the
    /// source and the file's bytes are hashed for `sha256`. The caller
    /// supplies the upstream revision and split because the file itself does
    /// not encode them.
    pub fn from_path(
        path: impl AsRef<Path>,
        source_revision: impl Into<String>,
        split: impl Into<String>,
        bytes: &[u8],
        task_count: usize,
    ) -> Self {
        let source = path.as_ref().display().to_string();
        Self::new(source, source_revision, split, bytes, task_count)
    }
}

impl fmt::Display for DatasetProvenance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "source={} rev={} split={} sha256={} tasks={}",
            self.source, self.source_revision, self.split, self.sha256, self.task_count
        )
    }
}

/// Compute the lowercase hex SHA-256 digest of `bytes`. Uses a tiny pure-Rust
/// implementation so the crate does not pull in a crypto dependency just to
/// fingerprint dataset files.
pub fn sha256_hex(bytes: &[u8]) -> Sha256 {
    let mut core = Sha256Core::new();
    core.update(bytes);
    core.finalize()
        .iter()
        .map(|b| format!("{:02x}", b))
        .collect()
}

/// Minimal pure-Rust SHA-256 implementation (FIPS 180-4). Sufficient for
/// dataset fingerprinting; not constant-time hardened for secrets.
struct Sha256Core {
    state: [u32; 8],
    buffer: [u8; 64],
    buffer_len: usize,
    bit_count: u64,
}

impl Sha256Core {
    const INITIAL_STATE: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];

    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];

    fn new() -> Self {
        Self {
            state: Self::INITIAL_STATE,
            buffer: [0u8; 64],
            buffer_len: 0,
            bit_count: 0,
        }
    }

    fn update(&mut self, mut input: &[u8]) {
        self.bit_count = self.bit_count.wrapping_add((input.len() as u64) * 8);
        while !input.is_empty() {
            let take = (64 - self.buffer_len).min(input.len());
            self.buffer[self.buffer_len..self.buffer_len + take].copy_from_slice(&input[..take]);
            self.buffer_len += take;
            input = &input[take..];
            if self.buffer_len == 64 {
                let block = self.buffer;
                self.process_block(&block);
                self.buffer_len = 0;
            }
        }
    }

    fn finalize(mut self) -> [u8; 32] {
        let bit_count = self.bit_count;
        self.buffer[self.buffer_len] = 0x80;
        self.buffer_len += 1;
        if self.buffer_len > 56 {
            for b in &mut self.buffer[self.buffer_len..] {
                *b = 0;
            }
            let block = self.buffer;
            self.process_block(&block);
            self.buffer_len = 0;
        }
        for b in &mut self.buffer[self.buffer_len..56] {
            *b = 0;
        }
        self.buffer[56..64].copy_from_slice(&bit_count.to_be_bytes());
        let block = self.buffer;
        self.process_block(&block);

        let mut out = [0u8; 32];
        for (i, word) in self.state.iter().enumerate() {
            out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
        }
        out
    }

    fn process_block(&mut self, block: &[u8; 64]) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                block[i * 4],
                block[i * 4 + 1],
                block[i * 4 + 2],
                block[i * 4 + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let mut a = self.state[0];
        let mut b = self.state[1];
        let mut c = self.state[2];
        let mut d = self.state[3];
        let mut e = self.state[4];
        let mut f = self.state[5];
        let mut g = self.state[6];
        let mut h = self.state[7];

        for (i, &w_i) in w.iter().enumerate() {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(Self::K[i])
                .wrapping_add(w_i);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let mj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(mj);

            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }

        self.state[0] = self.state[0].wrapping_add(a);
        self.state[1] = self.state[1].wrapping_add(b);
        self.state[2] = self.state[2].wrapping_add(c);
        self.state[3] = self.state[3].wrapping_add(d);
        self.state[4] = self.state[4].wrapping_add(e);
        self.state[5] = self.state[5].wrapping_add(f);
        self.state[6] = self.state[6].wrapping_add(g);
        self.state[7] = self.state[7].wrapping_add(h);
    }
}

/// Compute the SHA-256 digest of `bytes` using [`Sha256Core`].
pub fn sha256(bytes: &[u8]) -> [u8; 32] {
    let mut core = Sha256Core::new();
    core.update(bytes);
    core.finalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_of_abc_matches_known_vector() {
        // Known SHA-256 of "abc" (FIPS 180-4 example).
        let digest = sha256_hex(b"abc");
        assert_eq!(
            digest,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn sha256_of_empty_string_matches_known_vector() {
        let digest = sha256_hex(b"");
        assert_eq!(
            digest,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn sha256_handles_long_inputs_across_blocks() {
        // Force the implementation to buffer and pad across multiple blocks.
        let bytes = vec![0xa5u8; 1024];
        let digest = sha256_hex(&bytes);
        // Cross-checked independently with the openssl CLI.
        assert_eq!(
            digest,
            "e75809e0d15667ce44e6aa5c64689a4917b245eb0920094ff0b017dc0612a17a"
        );
    }

    #[test]
    fn provenance_records_all_required_fields() {
        let prov = DatasetProvenance::new(
            "https://example.test/mmlu.csv",
            "v1.0",
            "test",
            b"subject,question,A,answer\n",
            1,
        );
        assert_eq!(prov.source, "https://example.test/mmlu.csv");
        assert_eq!(prov.source_revision, "v1.0");
        assert_eq!(prov.split, "test");
        assert_eq!(prov.task_count, 1);
        assert_eq!(prov.sha256.len(), 64);
        // Display includes every field.
        let display = prov.to_string();
        assert!(display.contains("source="));
        assert!(display.contains("rev="));
        assert!(display.contains("split="));
        assert!(display.contains("sha256="));
        assert!(display.contains("tasks="));
    }
}
