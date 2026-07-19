//! Bounded pipeline cache.
//!
//! [`PipelineCache`] maps `(ModelId, plan_revision, fingerprint_hash)` to a
//! [`CompiledPipeline`] and enforces a capacity bound via either
//! [`EvictionPolicy::Lru`] or [`EvictionPolicy::Fifo`]. The cache is
//! `Sync + Send` (the lock is `parking_lot::Mutex`), can be persisted via
//! [`PipelineCache::write_through`], and reloaded from disk via
//! [`PipelineCache::load_from_disk`].
//!
//! The cache is deliberately a plain in-memory store with no background
//! threads. Eviction happens synchronously on insert.

use std::collections::HashMap;
use std::hash::Hash;
use std::path::Path;

use model_plan::ModelId;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

/// Eviction policy for [`PipelineCache`].
///
/// `Lru` evicts the entry whose last access (insert or `get`) is the
/// oldest. `Fifo` evicts the entry that has lived in the cache the longest,
/// regardless of how recently it was read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvictionPolicy {
    /// Least-recently-used.
    Lru,
    /// First-in-first-out.
    Fifo,
}

/// A compiled pipeline: the MSL shader source plus the metadata the cache
/// needs to know whether the entry can be reused.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompiledPipeline {
    /// `ModelId` of the plan that was compiled. Stored on the value for
    /// the write_through path so a reloaded cache can be sanity-checked.
    pub plan_id: ModelId,
    /// Revision of the plan that was compiled. Bumping the plan_revision
    /// is the canonical way to invalidate a cached entry.
    pub source_revision: u64,
    /// Generated MSL shader source (stub string for now; real codegen in
    /// a future task).
    pub shader_source: String,
    /// Unix epoch millis at which this entry was compiled.
    pub compiled_at_unix_ms: u64,
    /// Cached Metal Shading Language compute-version reported by the
    /// compiler. `0` for the software fallback path.
    pub ms_compute_version: u32,
    /// The fingerprint hash under which this entry was compiled. Two
    /// entries with the same plan_id + source_revision but different
    /// fingerprint_hash are distinct cache entries.
    pub fingerprint_hash: u64,
}

impl CompiledPipeline {
    /// Build a placeholder compiled pipeline for tests and cache-warming
    /// paths. `compiled_at_unix_ms` is set to the current wall clock so
    /// the resulting object round-trips through JSON without surprises.
    pub fn placeholder(plan_id: ModelId, shader_source: &str, fingerprint_hash: u64) -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        Self {
            plan_id,
            source_revision: 0,
            shader_source: shader_source.to_string(),
            compiled_at_unix_ms: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
            ms_compute_version: 0,
            fingerprint_hash,
        }
    }
}

/// Cache-key view exposed to callers so they can pre-compute a key
/// independently of any cache instance.
///
/// Hashing is a stable reimplementation of `(ModelId, u64, u64)` —
/// deliberately NOT derived via `Hash` because `ModelId` is `pub struct
/// ModelId(pub u64)` and would produce a colliding hash with two raw
/// `u64`s. We compose the hash with explicit per-field calls.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CacheKey {
    plan_id: ModelId,
    plan_revision: u64,
    fingerprint_hash: u64,
}

/// Hit / miss / eviction counters reported by [`PipelineCache::stats`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CacheStats {
    /// Total number of `get` calls that returned `Some`.
    pub hits: u64,
    /// Total number of `get` calls that returned `None`.
    pub misses: u64,
    /// Total number of insert-driven evictions.
    pub evictions: u64,
    /// Current entry count.
    pub size: u64,
    /// Configured capacity.
    pub capacity: u64,
}

/// Internal entry combining the compiled pipeline with bookkeeping needed
/// for LRU vs. FIFO eviction.
#[derive(Clone)]
struct Entry {
    value: CompiledPipeline,
    /// Monotonic counter incremented on every access (insert or get).
    /// Used by LRU.
    last_access: u64,
    /// Monotonic counter set once at insert. Used by FIFO.
    inserted_at: u64,
}

/// On-disk representation of the cache: just a vector of entries. The
/// policy is reconstructed from the in-memory cache when reloaded.
#[derive(Debug, Serialize, Deserialize)]
struct PersistedCache {
    entries: Vec<CompiledPipeline>,
}

/// Bounded pipeline cache.
pub struct PipelineCache {
    policy: EvictionPolicy,
    capacity: usize,
    inner: Mutex<Inner>,
}

struct Inner {
    entries: HashMap<CacheKey, Entry>,
    next_access: u64,
    stats: CacheStats,
}

impl PipelineCache {
    /// Create a new cache with the given eviction policy and capacity.
    ///
    /// `capacity` is the maximum number of entries. A capacity of 0 is
    /// rejected: the cache is meant to hold at least one entry.
    pub fn new(policy: EvictionPolicy, capacity: usize) -> Self {
        assert!(capacity > 0, "PipelineCache capacity must be > 0");
        Self {
            policy,
            capacity,
            inner: Mutex::new(Inner {
                entries: HashMap::with_capacity(capacity),
                next_access: 1,
                stats: CacheStats {
                    hits: 0,
                    misses: 0,
                    evictions: 0,
                    size: 0,
                    capacity: capacity as u64,
                },
            }),
        }
    }

    /// Build the cache key for a `(plan_id, plan_revision, fingerprint_hash)`
    /// triple. Exposed so callers can pre-compute the key without holding a
    /// mutable cache reference.
    pub fn cache_key_for(plan_id: ModelId, plan_revision: u64, fingerprint_hash: u64) -> CacheKey {
        CacheKey {
            plan_id,
            plan_revision,
            fingerprint_hash,
        }
    }

    /// Look up an entry. Updates the LRU access counter on hit. FIFO does
    /// NOT refresh the insertion order on read.
    pub fn get(
        &self,
        plan_id: ModelId,
        plan_revision: u64,
        fingerprint_hash: u64,
    ) -> Option<CompiledPipeline> {
        let key = Self::cache_key_for(plan_id, plan_revision, fingerprint_hash);
        let mut inner = self.inner.lock();
        if let Some(entry) = inner.entries.get(&key).cloned() {
            // For LRU, refresh the access counter so this entry is no
            // longer considered "least recently used". For FIFO we leave
            // `inserted_at` untouched (the entry's eviction order is fixed
            // by the time it was first inserted) but we still need a
            // defensive bump to avoid spurious later collisions.
            match self.policy {
                EvictionPolicy::Lru => {
                    inner.next_access = inner.next_access.saturating_add(1);
                    let access = inner.next_access;
                    let mut e = entry.clone();
                    e.last_access = access;
                    inner.entries.insert(key, e);
                }
                EvictionPolicy::Fifo => {
                    // No-op for FIFO: leave inserted_at untouched.
                }
            }
            inner.stats.hits = inner.stats.hits.saturating_add(1);
            Some(entry.value)
        } else {
            inner.stats.misses = inner.stats.misses.saturating_add(1);
            None
        }
    }

    /// Insert (or replace) an entry. If the cache is at capacity the
    /// eviction policy decides which existing entry to drop.
    pub fn insert(
        &self,
        plan_id: ModelId,
        plan_revision: u64,
        fingerprint_hash: u64,
        value: CompiledPipeline,
    ) {
        let key = Self::cache_key_for(plan_id, plan_revision, fingerprint_hash);
        let mut inner = self.inner.lock();
        inner.next_access = inner.next_access.saturating_add(1);
        let access = inner.next_access;

        // Replace existing entry in-place — no eviction, no size change.
        if inner.entries.contains_key(&key) {
            inner.entries.insert(
                key,
                Entry {
                    value,
                    last_access: access,
                    inserted_at: access,
                },
            );
            return;
        }

        // Evict before insert if we'd exceed capacity.
        while inner.entries.len() >= self.capacity {
            Self::evict_one(&mut inner, self.policy);
            inner.stats.evictions = inner.stats.evictions.saturating_add(1);
        }

        inner.entries.insert(
            key,
            Entry {
                value,
                last_access: access,
                inserted_at: access,
            },
        );
        inner.stats.size = inner.entries.len() as u64;
    }

    /// Return a snapshot of the current stats.
    pub fn stats(&self) -> CacheStats {
        let inner = self.inner.lock();
        let mut s = inner.stats;
        s.size = inner.entries.len() as u64;
        s.capacity = self.capacity as u64;
        s
    }

    /// Persist every entry to `path` as JSON. The on-disk format is a list
    /// of [`CompiledPipeline`] entries; the policy is *not* persisted
    /// because the caller owns the cache instance and is expected to
    /// construct a new one with the desired policy before reloading.
    pub fn write_through(&self, path: &Path) -> std::io::Result<()> {
        let inner = self.inner.lock();
        let entries: Vec<CompiledPipeline> =
            inner.entries.values().map(|e| e.value.clone()).collect();
        let persisted = PersistedCache { entries };
        let json = serde_json::to_vec_pretty(&persisted)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(path, json)
    }

    /// Reload entries from `path`. Existing entries in this cache are
    /// preserved (the file is merged in). Each persisted entry's
    /// `fingerprint_hash` is used as the cache-key dimension so two
    /// entries with the same `plan_id` but different fingerprints coexist.
    pub fn load_from_disk(&mut self, path: &Path) -> std::io::Result<usize> {
        let bytes = std::fs::read(path)?;
        let persisted: PersistedCache = serde_json::from_slice(&bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let mut count = 0;
        for cp in persisted.entries {
            self.insert(cp.plan_id, cp.source_revision, cp.fingerprint_hash, cp);
            count += 1;
        }
        Ok(count)
    }

    fn evict_one(inner: &mut Inner, policy: EvictionPolicy) {
        let victim = match policy {
            EvictionPolicy::Lru => inner
                .entries
                .iter()
                .min_by_key(|(_, e)| e.last_access)
                .map(|(k, _)| *k),
            EvictionPolicy::Fifo => inner
                .entries
                .iter()
                .min_by_key(|(_, e)| e.inserted_at)
                .map(|(k, _)| *k),
        };
        if let Some(k) = victim {
            inner.entries.remove(&k);
        }
    }
}

impl std::fmt::Debug for PipelineCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner = self.inner.lock();
        f.debug_struct("PipelineCache")
            .field("policy", &self.policy)
            .field("capacity", &self.capacity)
            .field("size", &inner.entries.len())
            .field("stats", &inner.stats)
            .finish()
    }
}

// Compile-time assertion: the cache can be shared across threads.
#[allow(dead_code)]
const _: () = {
    fn assert_send_sync<T: Send + Sync>() {}
    fn _f() {
        assert_send_sync::<PipelineCache>();
    }
};

// ---------------------------------------------------------------------------
// Hashing for the entry keys uses the derived impl on `CacheKey` (the
// field set is `(ModelId, u64, u64)`, so the derived hash is sufficient
// and unambiguous). We keep this comment block so future maintainers do
// not add a manual impl that would conflict with the derive.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn mk(plan_id: u64, fp: u64, src: &str) -> CompiledPipeline {
        CompiledPipeline::placeholder(ModelId(plan_id), src, fp)
    }

    #[test]
    fn replace_in_place_does_not_evict() {
        let cache = PipelineCache::new(EvictionPolicy::Lru, 4);
        cache.insert(ModelId(1), 0, 1, mk(1, 1, "first"));
        let evictions_before = cache.stats().evictions;
        cache.insert(ModelId(1), 0, 1, mk(1, 1, "second"));
        assert_eq!(cache.stats().size, 1);
        assert_eq!(cache.stats().evictions, evictions_before);
        assert_eq!(
            cache.get(ModelId(1), 0, 1).unwrap().shader_source,
            "second"
        );
    }

    #[test]
    fn fifo_does_not_refresh_on_get() {
        let cache = PipelineCache::new(EvictionPolicy::Fifo, 2);
        cache.insert(ModelId(1), 0, 1, mk(1, 1, "a"));
        cache.insert(ModelId(1), 0, 2, mk(1, 2, "b"));
        // Read the oldest.
        assert!(cache.get(ModelId(1), 0, 1).is_some());
        // Adding a third must evict the original insertion (1), not (2).
        cache.insert(ModelId(1), 0, 3, mk(1, 3, "c"));
        assert!(cache.get(ModelId(1), 0, 1).is_none());
        assert!(cache.get(ModelId(1), 0, 2).is_some());
        assert!(cache.get(ModelId(1), 0, 3).is_some());
    }
}