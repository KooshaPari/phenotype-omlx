//! Fleet discovery + heartbeat protocol for multi-node OMLX clusters.
//!
//! Nodes can advertise over mDNS/DNS-SD, register themselves in a shared
//! directory, and send heartbeats with capability tuples (vLLM, MLX, etc.).

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::time::{SystemTime, UNIX_EPOCH};

/// Hardware and software capabilities advertised by a fleet node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeCapabilities {
    pub backends: Vec<String>, // ["mlx", "vllm", "sglang", "tensorrt", "llamacpp", "metal"]
    pub models: Vec<String>,   // ["Qwen2.5-7B-Instruct", ...]
    pub device: String,        // "Apple M1 Pro"
    pub memory_gb: f32,
    pub cuda: bool,
    pub metal: bool,
}

/// Periodic heartbeat message sent by each fleet node to advertise liveness and load.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Heartbeat {
    pub node_id: String,
    pub addr: String,
    pub port: u16,
    pub ts_ms: u64,
    pub caps: NodeCapabilities,
    pub inflight: usize,
}

/// Trait for fleet membership management — announce, discover, and evict nodes.
pub trait Fleet: Send + Sync {
    /// Register or update a node's heartbeat in the fleet directory.
    fn announce(&self, hb: Heartbeat) -> Result<(), String>;
    /// Return all non-stale peer heartbeats known to this fleet instance.
    fn peers(&self) -> Vec<Heartbeat>;
    /// Remove a node from the fleet directory by its identifier.
    fn remove(&self, node_id: &str) -> Result<(), String>;
}

/// In-memory fleet registry — used as a fallback when no shared directory
/// (Redis, Consul, etcd) is configured. Useful for single-binary dev runs.
pub struct InMemoryFleet {
    pub nodes: parking_lot::RwLock<BTreeMap<String, Heartbeat>>,
    pub ttl_ms: u64,
}

impl InMemoryFleet {
    /// Create a new in-memory fleet registry with the given TTL in milliseconds.
    pub fn new(ttl_ms: u64) -> Self {
        Self {
            nodes: parking_lot::RwLock::new(BTreeMap::new()),
            ttl_ms,
        }
    }
}

impl Fleet for InMemoryFleet {
    fn announce(&self, hb: Heartbeat) -> Result<(), String> {
        self.nodes.write().insert(hb.node_id.clone(), hb);
        Ok(())
    }
    fn peers(&self) -> Vec<Heartbeat> {
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        let ttl = self.ttl_ms;
        self.nodes
            .read()
            .values()
            .filter(|hb| now_ms.saturating_sub(hb.ts_ms) <= ttl)
            .cloned()
            .collect()
    }
    fn remove(&self, node_id: &str) -> Result<(), String> {
        self.nodes.write().remove(node_id);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn caps() -> NodeCapabilities {
        NodeCapabilities {
            backends: vec!["mlx".into()],
            models: vec!["test".into()],
            device: "test-device".into(),
            memory_gb: 16.0,
            cuda: false,
            metal: true,
        }
    }

    fn hb(node_id: &str, ts_ms: u64) -> Heartbeat {
        Heartbeat {
            node_id: node_id.into(),
            addr: "127.0.0.1".into(),
            port: 8080,
            ts_ms,
            caps: caps(),
            inflight: 0,
        }
    }

    fn now_ms() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64
    }

    #[test]
    fn new_creates_empty_fleet() {
        let fleet = InMemoryFleet::new(5000);
        assert!(fleet.peers().is_empty());
    }

    #[test]
    fn add_peer_then_peers_returns_it() {
        let fleet = InMemoryFleet::new(5_000);
        fleet.announce(hb("node-a", now_ms())).unwrap();
        let peers = fleet.peers();
        assert_eq!(peers.len(), 1);
        assert_eq!(peers[0].node_id, "node-a");
    }

    #[test]
    fn peers_filters_stale_entries() {
        let fleet = InMemoryFleet::new(1000);
        let stale_ts = now_ms().saturating_sub(5000); // 5s ago, TTL=1s
        let fresh_ts = now_ms();

        fleet.announce(hb("stale-node", stale_ts)).unwrap();
        fleet.announce(hb("fresh-node", fresh_ts)).unwrap();

        let peers = fleet.peers();
        assert_eq!(peers.len(), 1);
        assert_eq!(peers[0].node_id, "fresh-node");
    }

    #[test]
    fn remove_peer_removes_it() {
        let fleet = InMemoryFleet::new(5_000);
        fleet.announce(hb("node-a", now_ms())).unwrap();
        fleet.announce(hb("node-b", now_ms())).unwrap();
        fleet.remove("node-a").unwrap();

        let peers = fleet.peers();
        assert_eq!(peers.len(), 1);
        assert_eq!(peers[0].node_id, "node-b");
    }

    #[test]
    fn peers_returns_empty_when_all_peers_stale() {
        let fleet = InMemoryFleet::new(500);
        let stale_ts = now_ms().saturating_sub(10_000);

        fleet.announce(hb("node-1", stale_ts)).unwrap();
        fleet.announce(hb("node-2", stale_ts)).unwrap();
        fleet.announce(hb("node-3", stale_ts)).unwrap();

        let peers = fleet.peers();
        assert!(
            peers.is_empty(),
            "all peers are stale, fleet should be empty"
        );
    }
}
