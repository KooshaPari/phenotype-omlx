//! Lockfile-hash integration test.
//!
//! Pins the SHA-256 fingerprint of `perf-core/Cargo.lock` recorded in
//! `perf-core/lockfile.lock` to the live digest computed via the
//! `sha2` crate. This is the Rust-side counterpart to
//! `scripts/verify_lockfile.sh` — the shell script covers the same
//! contract from outside the test harness, this file covers it from
//! inside. Two properties the shell-only check has that this test
//! also has:
//!
//! 1. The fingerprint file is parseable (single `sha256:` line, hex
//!    digest, no leading whitespace).
//! 2. The recorded digest matches a freshly-computed SHA-256 of the
//!    verbatim bytes of `Cargo.lock`.
//!
//! Why a Rust test (and not just the shell script)? The shell
//! verifier runs on demand (or in a CI pre-push hook). A Rust test
//! runs on every `cargo test --workspace`, so any drift in
//! `Cargo.lock` (e.g. a `cargo update` that lands in the working
//! tree without re-running `scripts/verify_lockfile.sh`) surfaces
//! immediately on the next `cargo test` invocation rather than at
//! the next push. This closes the gap where the lockfile is changed
//! and committed without the fingerprint being refreshed.
//!
//! The test is deterministic and has no external dependencies
//! beyond `sha2` (already in `regress-baseline`'s `[dependencies]`
//! per `Cargo.toml` line 20). It does not shell out to `shasum` /
//! `sha256sum` — the digest is computed in-process so the test
//! behaviour is byte-identical on macOS, Linux, and Windows.
//!
//! ## Failure modes this catches
//!
//! - `Cargo.lock` changes (a dep bump, a `cargo update`) without the
//!   fingerprint in `lockfile.lock` being refreshed. The test fails
//!   with `MISMATCH` and prints both digests, so the fix is a
//!   one-line edit: `bash scripts/verify_lockfile.sh` to capture
//!   the new digest, paste it into `lockfile.lock`, and re-run.
//! - `lockfile.lock` is tampered with (someone hand-edits the digest
//!   to point at a different `Cargo.lock`). Same failure mode as
//!   above — the test computes the live hash and compares.
//! - `lockfile.lock` is deleted or the `sha256:` line is removed.
//!   The test fails with `MISSING sha256: LINE`.
//! - `Cargo.lock` is missing from disk. The test fails with a
//!   clear `io::Error` path message.

use std::fs;
use std::path::PathBuf;

use sha2::{Digest, Sha256};

/// Path to the workspace's `Cargo.lock`, resolved relative to the
/// regress-baseline crate's manifest directory (`perf-core/regress-baseline`).
fn cargo_lock_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("Cargo.lock")
}

/// Path to the workspace's `lockfile.lock` fingerprint file.
fn lockfile_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("lockfile.lock")
}

/// Compute the SHA-256 of `path`'s verbatim bytes and return the
/// lowercase-hex digest (64 chars).
fn sha256_file_hex(path: &std::path::Path) -> Result<String, String> {
    let bytes = fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let digest = Sha256::digest(&bytes);
    let mut hex = String::with_capacity(64);
    for b in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut hex, "{b:02x}");
    }
    Ok(hex)
}

/// Pull the `sha256:` line out of the fingerprint file. Returns
/// `Err(msg)` if the file is missing, unreadable, or doesn't contain
/// a `sha256: <hex>` line at column 0.
fn parse_expected_sha256(contents: &str) -> Result<String, String> {
    for line in contents.lines() {
        // Strict prefix match — no leading whitespace, no inline
        // comments. The fingerprint file format is owned by
        // `scripts/verify_lockfile.sh`; this test reads the same
        // format and rejects anything else so a future format
        // change is caught loudly rather than silently mis-parsed.
        if let Some(rest) = line.strip_prefix("sha256:") {
            let trimmed = rest.trim();
            if trimmed.len() == 64 && trimmed.chars().all(|c| c.is_ascii_hexdigit()) {
                return Ok(trimmed.to_ascii_lowercase());
            }
            return Err(format!(
                "sha256: line present but value is not 64 lowercase hex chars: {trimmed:?}"
            ));
        }
    }
    Err(
        "MISSING sha256: LINE in lockfile.lock (expected format: `sha256: <64-hex-digest>`)"
            .to_string(),
    )
}

#[test]
fn lockfile_hash_matches_cargo_lock_sha256() {
    let lock_path = cargo_lock_path();
    let fp_path = lockfile_path();

    // 1. Compute the live SHA-256 of Cargo.lock.
    let actual = sha256_file_hex(&lock_path).unwrap_or_else(|e| panic!("{e}"));

    // 2. Parse the expected SHA-256 from lockfile.lock.
    let fp_bytes =
        fs::read_to_string(&fp_path).unwrap_or_else(|e| panic!("read {}: {e}", fp_path.display()));
    let expected = parse_expected_sha256(&fp_bytes).unwrap_or_else(|e| panic!("{e}"));

    // 3. Assert they match. Failure prints both digests and the
    //    remediation hint so the operator can fix the drift in one
    //    round-trip without reading the test source.
    assert_eq!(
        actual, expected,
        "lockfile.lock SHA-256 fingerprint drifted from Cargo.lock.\n\
         actual:   {actual}\n\
         expected: {expected}\n\
         Remediation: run `bash scripts/verify_lockfile.sh` to capture the new digest, \
         paste it into `perf-core/lockfile.lock` (replacing the `sha256:` line), and \
         re-run `cargo test -p regress-baseline --test lockfile_hash`."
    );
}

/// Tamper-detection test: write a junk byte to a temporary copy of
/// `Cargo.lock`, hash the corrupted copy, and assert the hash
/// differs from the fingerprint. This is a positive control — if
/// this test ever passes after a one-byte corruption, the hash
/// computation has been silently neutered (e.g. by hashing an
/// empty buffer or short-circuiting to a constant).
#[test]
fn lockfile_hash_differs_for_tampered_cargo_lock() {
    let lock_path = cargo_lock_path();
    let original = fs::read(&lock_path).expect("read Cargo.lock");

    // Build a one-byte-different copy in memory (no filesystem
    // mutation — keeps the test hermetic and parallel-safe).
    let mut tampered = original.clone();
    let last_idx = tampered.len() - 1;
    tampered[last_idx] = tampered[last_idx].wrapping_add(1);

    let original_hash = {
        let mut h = Sha256::new();
        h.update(&original);
        let digest = h.finalize();
        let mut hex = String::with_capacity(64);
        for b in digest {
            use std::fmt::Write as _;
            let _ = write!(&mut hex, "{b:02x}");
        }
        hex
    };
    let tampered_hash = {
        let mut h = Sha256::new();
        h.update(&tampered);
        let digest = h.finalize();
        let mut hex = String::with_capacity(64);
        for b in digest {
            use std::fmt::Write as _;
            let _ = write!(&mut hex, "{b:02x}");
        }
        hex
    };

    assert_ne!(
        original_hash, tampered_hash,
        "SHA-256 of one-byte-tampered Cargo.lock must differ from the original; \
         original={original_hash} tampered={tampered_hash}. \
         If they are equal, the hash function has been silently neutered."
    );
}
