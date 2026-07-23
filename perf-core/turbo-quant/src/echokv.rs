use serde::{Deserialize, Serialize};

/// Score for a single KV cache entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntryScore {
    pub position: usize,
    pub attention_score: f32,
    pub recency_score: f32,
    pub combined_score: f32,
}

/// EchoKV configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EchoKVConfig {
    /// Maximum cache size (number of entries)
    pub max_cache_size: usize,
    /// Weight for attention-based scoring (0.0-1.0)
    pub attention_weight: f32,
    /// Weight for recency-based scoring (0.0-1.0)
    pub recency_weight: f32,
    /// Minimum score threshold — entries below this are evicted
    pub eviction_threshold: f32,
    /// Window of recent positions used for recency scoring
    pub recency_window: usize,
}

impl Default for EchoKVConfig {
    fn default() -> Self {
        Self {
            max_cache_size: 512,
            attention_weight: 0.7,
            recency_weight: 0.3,
            eviction_threshold: 0.05,
            recency_window: 128,
        }
    }
}

/// EchoKV cache manager
#[derive(Debug)]
pub struct EchoKVCache {
    config: EchoKVConfig,
    entries: Vec<EntryScore>,
    current_position: usize,
}

impl EchoKVCache {
    /// Create a new cache manager with the given configuration.
    pub fn new(config: EchoKVConfig) -> Self {
        Self {
            config,
            entries: Vec::new(),
            current_position: 0,
        }
    }

    /// Score a new entry based on attention weights
    pub fn score_entry(&self, position: usize, attention_weight: f32) -> EntryScore {
        let recency_score = if self.config.recency_window == 0 {
            0.0
        } else {
            let distance = self.current_position.saturating_sub(position);
            let half_window = self.config.recency_window as f32 / 2.0;
            let rec = 1.0 - (distance as f32 / half_window).min(1.0);
            rec.max(0.0)
        };
        let combined_score = self.config.attention_weight * attention_weight
            + self.config.recency_weight * recency_score;
        EntryScore {
            position,
            attention_score: attention_weight,
            recency_score,
            combined_score,
        }
    }

    /// Add entry and evict if over capacity
    pub fn insert(&mut self, position: usize, attention_weight: f32) {
        let entry = self.score_entry(position, attention_weight);
        self.entries.push(entry);
        self.current_position = self.current_position.max(position);
        if self.entries.len() > self.config.max_cache_size {
            self.evict();
        }
    }

    /// Evict entries below threshold or over capacity
    pub fn evict(&mut self) -> Vec<EntryScore> {
        let mut evicted = Vec::new();
        self.entries.retain(|e| {
            if e.combined_score < self.config.eviction_threshold {
                evicted.push(e.clone());
                false
            } else {
                true
            }
        });
        if self.entries.len() > self.config.max_cache_size {
            self.entries
                .sort_by(|a, b| a.combined_score.partial_cmp(&b.combined_score).unwrap());
            while self.entries.len() > self.config.max_cache_size {
                evicted.push(self.entries.remove(0));
            }
        }
        evicted
    }

    /// Get current cache size
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get all entries sorted by score (highest first)
    pub fn ranked_entries(&self) -> Vec<&EntryScore> {
        let mut sorted: Vec<&EntryScore> = self.entries.iter().collect();
        sorted.sort_by(|a, b| b.combined_score.partial_cmp(&a.combined_score).unwrap());
        sorted
    }

    /// Get total attention mass retained
    pub fn retained_mass(&self) -> f32 {
        self.entries.iter().map(|e| e.attention_score).sum()
    }
}
