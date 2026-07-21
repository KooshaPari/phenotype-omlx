//! Engine state — observable per-instance counters and history.
//!
//! The engine wraps `Arc<Mutex<...>>` at the FFI/Python boundary, but the
//! inner state itself is plain data: `EngineState`. Tests, snapshots, and
//! debugger tooling all interact with this struct directly, without any
//! lock overhead.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

/// Maximum number of accepted tokens retained in the rolling history.
pub const HISTORY_CAP: usize = 1024;

/// Plain, copyable snapshot of an engine's runtime state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EngineState {
    /// Tokens currently held in the target KV cache.
    pub kv_len: usize,
    /// Cumulative number of drafted tokens across all steps.
    pub drafted_total: u64,
    /// Cumulative number of accepted tokens across all steps.
    pub accepted_total: u64,
    /// Accepted tokens emitted by the most recent step.
    pub last_step_accepted: usize,
    /// Drafted tokens consumed by the most recent step.
    pub last_step_drafted: usize,
    /// Rolling history of the last [`HISTORY_CAP`] accepted tokens.
    /// Most recent is at the back.
    pub history: VecDeque<u32>,
}

impl Default for EngineState {
    fn default() -> Self {
        Self::new()
    }
}

impl EngineState {
    /// Construct a zero-valued state with empty history.
    pub fn new() -> Self {
        Self {
            kv_len: 0,
            drafted_total: 0,
            accepted_total: 0,
            last_step_accepted: 0,
            last_step_drafted: 0,
            history: VecDeque::new(),
        }
    }

    /// Independent copy (history is also duplicated, not aliased).
    pub fn snapshot(&self) -> Self {
        Self {
            kv_len: self.kv_len,
            drafted_total: self.drafted_total,
            accepted_total: self.accepted_total,
            last_step_accepted: self.last_step_accepted,
            last_step_drafted: self.last_step_drafted,
            history: self.history.clone(),
        }
    }

    /// Reset every counter and clear the history.
    pub fn reset(&mut self) {
        self.kv_len = 0;
        self.drafted_total = 0;
        self.accepted_total = 0;
        self.last_step_accepted = 0;
        self.last_step_drafted = 0;
        self.history.clear();
    }

    /// Record `n` additional KV tokens (typically the prompt + accepted tokens).
    pub fn extend_kv(&mut self, n: usize) {
        self.kv_len = self.kv_len.saturating_add(n);
    }

    /// Push a single accepted token onto the rolling history, dropping the
    /// oldest entry if we exceed the cap.
    pub fn push_accepted(&mut self, token: u32) {
        if self.history.len() >= HISTORY_CAP {
            self.history.pop_front();
        }
        self.history.push_back(token);
    }

    /// Update both per-step and cumulative counters in one call.
    pub fn record_step(&mut self, drafted: usize, accepted: usize) {
        self.last_step_drafted = drafted;
        self.last_step_accepted = accepted;
        self.drafted_total = self.drafted_total.saturating_add(drafted as u64);
        self.accepted_total = self.accepted_total.saturating_add(accepted as u64);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_is_zero() {
        let s = EngineState::new();
        assert_eq!(s.kv_len, 0);
        assert_eq!(s.drafted_total, 0);
        assert_eq!(s.accepted_total, 0);
        assert_eq!(s.last_step_accepted, 0);
        assert_eq!(s.last_step_drafted, 0);
        assert!(s.history.is_empty());
    }

    #[test]
    fn snapshot_is_independent() {
        let mut a = EngineState::new();
        a.kv_len = 8;
        a.history.push_back(1);
        let b = a.snapshot();
        a.history.push_back(2);
        assert_eq!(b.history, VecDeque::from([1]));
    }

    #[test]
    fn reset_clears_everything() {
        let mut s = EngineState::new();
        s.extend_kv(32);
        s.push_accepted(99);
        s.record_step(4, 3);
        s.reset();
        assert_eq!(s.kv_len, 0);
        assert!(s.history.is_empty());
        assert_eq!(s.drafted_total, 0);
        assert_eq!(s.accepted_total, 0);
    }

    #[test]
    fn push_accepted_caps_at_history_cap() {
        let mut s = EngineState::new();
        for i in 0..(HISTORY_CAP + 100) as u32 {
            s.push_accepted(i);
        }
        assert_eq!(s.history.len(), HISTORY_CAP);
        // newest preserved, oldest dropped
        assert_eq!(s.history[0], 100);
        assert_eq!(s.history[HISTORY_CAP - 1], (HISTORY_CAP + 99) as u32);
    }

    #[test]
    fn record_step_accumulates_counters() {
        let mut s = EngineState::new();
        s.record_step(4, 3);
        s.record_step(2, 1);
        assert_eq!(s.drafted_total, 6);
        assert_eq!(s.accepted_total, 4);
        assert_eq!(s.last_step_drafted, 2);
        assert_eq!(s.last_step_accepted, 1);
    }
}
