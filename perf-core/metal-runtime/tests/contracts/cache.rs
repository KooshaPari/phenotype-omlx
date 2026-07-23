//! §2 — Bounded LRU/FIFO cache contracts.
//!
//! Covers insert/get, missing-key returns None, LRU eviction, FIFO eviction,
//! hits/misses/evictions counters, distinctness by fingerprint hash, and
//! `write_through` + `load_from_disk` persistence.

use metal_runtime::{CompiledPipeline, EvictionPolicy, GpuFamily, PipelineCache};
use model_plan::ModelId;

use super::common::identity_fp;

#[test]
fn cache_insert_and_get_returns_some_value() {
    let mut cache = PipelineCache::new(EvictionPolicy::Lru, 16);
    let fp = identity_fp(GpuFamily::Software);
    let compiled = CompiledPipeline::placeholder(ModelId(1), "src", fp.fingerprint_hash());
    cache.insert(ModelId(1), 0, fp.fingerprint_hash(), compiled.clone());
    let got = cache.get(ModelId(1), 0, fp.fingerprint_hash());
    assert!(got.is_some());
    assert_eq!(got.unwrap().shader_source, "src");
}

#[test]
fn cache_get_on_missing_key_returns_none() {
    let mut cache = PipelineCache::new(EvictionPolicy::Lru, 16);
    let fp = identity_fp(GpuFamily::Software);
    let missing = cache.get(ModelId(999), 0, fp.fingerprint_hash());
    assert!(missing.is_none());
}

#[test]
fn cache_lru_evicts_least_recently_used_when_over_capacity() {
    let mut cache = PipelineCache::new(EvictionPolicy::Lru, 3);
    cache.insert(
        ModelId(1),
        0,
        1,
        CompiledPipeline::placeholder(ModelId(1), "a", 1),
    );
    cache.insert(
        ModelId(1),
        0,
        2,
        CompiledPipeline::placeholder(ModelId(1), "b", 2),
    );
    cache.insert(
        ModelId(1),
        0,
        3,
        CompiledPipeline::placeholder(ModelId(1), "c", 3),
    );
    // Touch key 1 so it is no longer least-recently-used.
    assert!(cache.get(ModelId(1), 0, 1).is_some());
    // Insert a 4th entry — key 2 should be evicted (LRU after the touch).
    cache.insert(
        ModelId(1),
        0,
        4,
        CompiledPipeline::placeholder(ModelId(1), "d", 4),
    );
    assert!(cache.get(ModelId(1), 0, 1).is_some(), "key 1 was touched");
    assert!(
        cache.get(ModelId(1), 0, 2).is_none(),
        "key 2 should be LRU-evicted"
    );
    assert!(cache.get(ModelId(1), 0, 3).is_some());
    assert!(cache.get(ModelId(1), 0, 4).is_some());
}

#[test]
fn cache_fifo_evicts_in_insertion_order_regardless_of_access() {
    let mut cache = PipelineCache::new(EvictionPolicy::Fifo, 3);
    cache.insert(
        ModelId(1),
        0,
        1,
        CompiledPipeline::placeholder(ModelId(1), "a", 1),
    );
    cache.insert(
        ModelId(1),
        0,
        2,
        CompiledPipeline::placeholder(ModelId(1), "b", 2),
    );
    cache.insert(
        ModelId(1),
        0,
        3,
        CompiledPipeline::placeholder(ModelId(1), "c", 3),
    );
    // Touch key 1 — under LRU this would refresh it, but FIFO must still
    // evict the oldest (key 1).
    assert!(cache.get(ModelId(1), 0, 1).is_some());
    cache.insert(
        ModelId(1),
        0,
        4,
        CompiledPipeline::placeholder(ModelId(1), "d", 4),
    );
    assert!(
        cache.get(ModelId(1), 0, 1).is_none(),
        "FIFO must evict insertion-1"
    );
    assert!(cache.get(ModelId(1), 0, 2).is_some());
    assert!(cache.get(ModelId(1), 0, 3).is_some());
    assert!(cache.get(ModelId(1), 0, 4).is_some());
}

#[test]
fn cache_hits_misses_evictions_counters_update_correctly() {
    let mut cache = PipelineCache::new(EvictionPolicy::Lru, 2);
    cache.insert(
        ModelId(1),
        0,
        1,
        CompiledPipeline::placeholder(ModelId(1), "a", 1),
    );
    cache.insert(
        ModelId(1),
        0,
        2,
        CompiledPipeline::placeholder(ModelId(1), "b", 2),
    );
    assert!(cache.get(ModelId(1), 0, 1).is_some());
    assert!(cache.get(ModelId(1), 0, 2).is_some());
    assert!(cache.get(ModelId(1), 0, 999).is_none());
    cache.insert(
        ModelId(1),
        0,
        3,
        CompiledPipeline::placeholder(ModelId(1), "c", 3),
    );
    cache.insert(
        ModelId(1),
        0,
        4,
        CompiledPipeline::placeholder(ModelId(1), "d", 4),
    );
    let stats = cache.stats();
    assert_eq!(stats.hits, 2);
    assert!(stats.misses >= 1);
    assert!(
        stats.evictions >= 1,
        "two evictions expected, got {}",
        stats.evictions
    );
    assert_eq!(stats.size, 2);
}

#[test]
fn cache_same_key_different_fingerprint_hash_are_distinct_entries() {
    let mut cache = PipelineCache::new(EvictionPolicy::Lru, 8);
    cache.insert(
        ModelId(1),
        0,
        100,
        CompiledPipeline::placeholder(ModelId(1), "fp-A", 100),
    );
    cache.insert(
        ModelId(1),
        0,
        200,
        CompiledPipeline::placeholder(ModelId(1), "fp-B", 200),
    );
    assert_eq!(cache.get(ModelId(1), 0, 100).unwrap().shader_source, "fp-A");
    assert_eq!(cache.get(ModelId(1), 0, 200).unwrap().shader_source, "fp-B");
}

#[test]
fn cache_write_through_persists_entries_that_can_be_reloaded() {
    let dir = std::env::temp_dir().join(format!("metal-runtime-test-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("cache.json");

    let mut cache = PipelineCache::new(EvictionPolicy::Lru, 4);
    cache.insert(
        ModelId(7),
        0,
        42,
        CompiledPipeline::placeholder(ModelId(7), "persisted", 42),
    );
    cache.write_through(&path).expect("write_through");

    let mut cache2 = PipelineCache::new(EvictionPolicy::Lru, 4);
    cache2.load_from_disk(&path).expect("load_from_disk");
    let got = cache2.get(ModelId(7), 0, 42);
    assert!(
        got.is_some(),
        "entry must survive write_through + load_from_disk"
    );
    assert_eq!(got.unwrap().shader_source, "persisted");

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(&dir);
}
