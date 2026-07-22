use turbo_quant::echokv::{EchoKVCache, EchoKVConfig};

fn config(
    max_cache_size: usize,
    attention_weight: f32,
    recency_weight: f32,
    eviction_threshold: f32,
) -> EchoKVConfig {
    EchoKVConfig {
        max_cache_size,
        attention_weight,
        recency_weight,
        eviction_threshold,
        recency_window: 128,
    }
}

#[test]
fn new_creates_empty_cache() {
    let cache = EchoKVCache::new(EchoKVConfig::default());
    assert_eq!(cache.len(), 0);
    assert!(cache.is_empty());
}

#[test]
fn insert_adds_entry_and_increases_len() {
    let mut cache = EchoKVCache::new(config(10, 0.7, 0.3, 0.05));
    assert!(cache.is_empty());
    cache.insert(0, 0.5);
    assert_eq!(cache.len(), 1);
    assert!(!cache.is_empty());
    cache.insert(1, 0.6);
    assert_eq!(cache.len(), 2);
}

#[test]
fn evict_removes_entries_below_threshold() {
    let mut cache = EchoKVCache::new(config(100, 1.0, 0.0, 0.5));
    cache.insert(0, 0.1); // score = 1.0*0.1 = 0.1 < 0.5
    cache.insert(1, 0.9); // score = 1.0*0.9 = 0.9 >= 0.5
    let evicted = cache.evict();
    assert_eq!(evicted.len(), 1);
    assert_eq!(evicted[0].position, 0);
    assert_eq!(cache.len(), 1);
    assert_eq!(cache.ranked_entries()[0].position, 1);
}

#[test]
fn evict_respects_max_cache_size() {
    let mut cache = EchoKVCache::new(config(3, 1.0, 0.0, 0.0));
    cache.insert(0, 0.9);
    cache.insert(1, 0.5);
    cache.insert(2, 0.3);
    cache.insert(3, 0.8);
    // Inserting 4th entry triggers eviction; max_cache_size=3, so 1 must be removed
    assert_eq!(cache.len(), 3);
    // Lowest attention score entry (position=2, score=0.3) should be evicted
    let positions: Vec<usize> = cache.ranked_entries().iter().map(|e| e.position).collect();
    assert!(!positions.contains(&2));
}

#[test]
fn score_entry_computes_correct_combined_score() {
    // attention_weight=1.0, recency_weight=0.0 → combined = 1.0 * attention_weight
    // recency_score is still computed (position 0 == current_position 0 → rec=1.0)
    // but is zeroed out by recency_weight=0.0 in the combined formula
    let cache = EchoKVCache::new(config(100, 1.0, 0.0, 0.0));
    let entry = cache.score_entry(0, 0.7);
    assert_eq!(entry.position, 0);
    assert!((entry.attention_score - 0.7).abs() < 1e-6);
    // recency_score is 1.0 (position matches current_position), but not used in combined
    assert!((entry.recency_score - 1.0).abs() < 1e-6);
    assert!((entry.combined_score - 0.7).abs() < 1e-6);
}

#[test]
fn score_entry_uses_recency() {
    // position = current_position → recency = 1.0
    let mut cache = EchoKVCache::new(config(100, 0.0, 1.0, 0.0));
    cache.insert(5, 0.0); // sets current_position = 5
    let entry = cache.score_entry(5, 0.0);
    assert!((entry.recency_score - 1.0).abs() < 1e-6);
    assert!((entry.combined_score - 1.0).abs() < 1e-6);
}

#[test]
fn ranked_entries_returns_sorted_by_score() {
    let mut cache = EchoKVCache::new(config(100, 1.0, 0.0, 0.0));
    cache.insert(0, 0.3);
    cache.insert(1, 0.9);
    cache.insert(2, 0.6);
    let ranked = cache.ranked_entries();
    assert_eq!(ranked.len(), 3);
    assert!(ranked[0].combined_score >= ranked[1].combined_score);
    assert!(ranked[1].combined_score >= ranked[2].combined_score);
    assert_eq!(ranked[0].position, 1);
    assert_eq!(ranked[1].position, 2);
    assert_eq!(ranked[2].position, 0);
}

#[test]
fn retained_mass_sums_attention_scores() {
    let mut cache = EchoKVCache::new(config(100, 1.0, 0.0, 0.0));
    cache.insert(0, 0.5);
    cache.insert(1, 0.3);
    cache.insert(2, 0.2);
    let mass = cache.retained_mass();
    assert!((mass - 1.0).abs() < 1e-6);
}

#[test]
fn max_cache_size_1_keeps_highest_scoring_entry() {
    let mut cache = EchoKVCache::new(config(1, 1.0, 0.0, 0.0));
    cache.insert(0, 0.3);
    assert_eq!(cache.len(), 1);
    cache.insert(1, 0.9);
    // After insert, max_cache_size=1 triggers eviction of lowest
    assert_eq!(cache.len(), 1);
    let ranked = cache.ranked_entries();
    assert_eq!(ranked[0].position, 1);
    assert!((ranked[0].attention_score - 0.9).abs() < 1e-6);
}

#[test]
fn eviction_threshold_1_0_evicts_all_entries() {
    // attention_weight=1.0, recency_weight=0.0 → combined = attention_score
    // All entries have attention_score ≤ 1.0, so none reach threshold of 1.0
    let mut cache = EchoKVCache::new(config(100, 1.0, 0.0, 1.0));
    cache.insert(0, 0.9);
    cache.insert(1, 0.99);
    let evicted = cache.evict();
    assert_eq!(evicted.len(), 2);
    assert_eq!(cache.len(), 0);
}

#[test]
fn eviction_threshold_0_0_no_eviction_below_threshold() {
    let mut cache = EchoKVCache::new(config(100, 1.0, 0.0, 0.0));
    cache.insert(0, 0.0);
    cache.insert(1, 0.001);
    let evicted = cache.evict();
    // threshold=0.0: no entries below 0.0, so none evicted by threshold
    assert_eq!(evicted.len(), 0);
    assert_eq!(cache.len(), 2);
}
