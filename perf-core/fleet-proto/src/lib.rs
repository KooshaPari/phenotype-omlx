//! Fleet discovery + heartbeat protocol for multi-node OMLX clusters.
//!
//! Nodes can advertise over mDNS/DNS-SD, register themselves in a shared
//! directory, and send heartbeats with capability tuples (vLLM, MLX, etc.).
//!
//! JSON-RPC 2.0 surface (`rpc` module): `capacity.fit`, `device.heartbeat`,
//! `fleet.peers`, `ping` — see ADR-006 federated synthetic monolith.

mod rpc;

pub use rpc::{dispatch, dispatch_str, RpcRequest, RpcResponse, RpcState};

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeCapabilities {
    pub backends: Vec<String>, // ["mlx", "vllm", "sglang", "tensorrt", "llamacpp", "metal"]
    pub models: Vec<String>,   // ["Qwen2.5-7B-Instruct", ...]
    pub device: String,        // "Apple M1 Pro"
    pub memory_gb: f32,
    pub cuda: bool,
    pub metal: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Heartbeat {
    pub node_id: String,
    pub addr: String,
    pub port: u16,
    pub ts_ms: u64,
    pub caps: NodeCapabilities,
    pub inflight: usize,
}

pub trait Fleet: Send + Sync {
    fn announce(&self, hb: Heartbeat) -> Result<(), String>;
    fn peers(&self) -> Vec<Heartbeat>;
    fn remove(&self, node_id: &str) -> Result<(), String>;
}

/// In-memory fleet registry — used as a fallback when no shared directory
/// (Redis, Consul, etcd) is configured. Useful for single-binary dev runs.
pub struct InMemoryFleet {
    pub nodes: parking_lot::RwLock<BTreeMap<String, Heartbeat>>,
    pub ttl_ms: u64,
}

impl InMemoryFleet {
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
