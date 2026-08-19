//! Minimal JSON-RPC 2.0 over serde_json for the federated monolith.

use crate::{Heartbeat, InMemoryFleet, NodeCapabilities, Fleet};
use pheno_capacity::{model_fits_in, vram_estimate, Dtype};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Deserialize)]
pub struct RpcRequest {
    pub jsonrpc: String,
    pub method: String,
    #[serde(default)]
    pub params: Value,
    pub id: Option<Value>,
}

#[derive(Debug, Serialize)]
pub struct RpcResponse {
    pub jsonrpc: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<RpcError>,
    pub id: Option<Value>,
}

#[derive(Debug, Serialize)]
pub struct RpcError {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

pub struct RpcState {
    pub fleet: Arc<InMemoryFleet>,
}

impl RpcState {
    pub fn new(ttl_ms: u64) -> Self {
        Self {
            fleet: Arc::new(InMemoryFleet::new(ttl_ms)),
        }
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn parse_dtype(s: &str) -> Result<Dtype, String> {
    match s.to_ascii_uppercase().as_str() {
        "F32" | "FP32" => Ok(Dtype::F32),
        "F16" | "FP16" => Ok(Dtype::F16),
        "BF16" => Ok(Dtype::Bf16),
        "I8" | "INT8" => Ok(Dtype::I8),
        "I4" | "INT4" | "Q4" => Ok(Dtype::I4),
        other => Err(format!("unknown dtype: {other}")),
    }
}

/// `capacity.fit` — { params, available_bytes, dtype }
fn capacity_fit(params: &Value) -> Result<Value, String> {
    let n = params
        .get("params")
        .and_then(|v| v.as_u64())
        .ok_or("params (u64) required")?;
    let available = params
        .get("available_bytes")
        .and_then(|v| v.as_u64())
        .ok_or("available_bytes (u64) required")?;
    let dtype_s = params
        .get("dtype")
        .and_then(|v| v.as_str())
        .unwrap_or("F16");
    let dtype = parse_dtype(dtype_s)?;
    let need = vram_estimate(n, dtype);
    Ok(json!({
        "fits": model_fits_in(n, available, dtype),
        "vram_estimate_bytes": need,
        "available_bytes": available,
        "dtype": dtype_s,
        "params": n,
    }))
}

/// `device.heartbeat` — announce into in-memory fleet; returns peers count.
fn device_heartbeat(state: &RpcState, params: &Value) -> Result<Value, String> {
    let node_id = params
        .get("node_id")
        .and_then(|v| v.as_str())
        .unwrap_or("local")
        .to_string();
    let addr = params
        .get("addr")
        .and_then(|v| v.as_str())
        .unwrap_or("127.0.0.1")
        .to_string();
    let port = params.get("port").and_then(|v| v.as_u64()).unwrap_or(0) as u16;
    let device = params
        .get("device")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();
    let memory_gb = params
        .get("memory_gb")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0) as f32;
    let cuda = params.get("cuda").and_then(|v| v.as_bool()).unwrap_or(false);
    let metal = params.get("metal").and_then(|v| v.as_bool()).unwrap_or(false);
    let backends: Vec<String> = params
        .get("backends")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();

    let hb = Heartbeat {
        node_id: node_id.clone(),
        addr,
        port,
        ts_ms: now_ms(),
        caps: NodeCapabilities {
            backends,
            models: vec![],
            device,
            memory_gb,
            cuda,
            metal,
        },
        inflight: 0,
    };
    state.fleet.announce(hb).map_err(|e| e)?;
    let peers = state.fleet.peers();
    Ok(json!({
        "ok": true,
        "node_id": node_id,
        "peers": peers.len(),
        "subject": "pheno.device.heartbeat",
    }))
}

fn fleet_peers(state: &RpcState) -> Result<Value, String> {
    Ok(serde_json::to_value(state.fleet.peers()).map_err(|e| e.to_string())?)
}

pub fn dispatch(state: &RpcState, req: RpcRequest) -> RpcResponse {
    if req.jsonrpc != "2.0" {
        return RpcResponse {
            jsonrpc: "2.0",
            result: None,
            error: Some(RpcError {
                code: -32600,
                message: "Invalid Request: jsonrpc must be 2.0".into(),
                data: None,
            }),
            id: req.id,
        };
    }

    let outcome = match req.method.as_str() {
        "capacity.fit" => capacity_fit(&req.params),
        "device.heartbeat" => device_heartbeat(state, &req.params),
        "fleet.peers" => fleet_peers(state),
        "ping" => Ok(json!("pong")),
        _ => Err(format!("Method not found: {}", req.method)),
    };

    match outcome {
        Ok(result) => RpcResponse {
            jsonrpc: "2.0",
            result: Some(result),
            error: None,
            id: req.id,
        },
        Err(msg) => {
            let code = if msg.starts_with("Method not found") {
                -32601
            } else {
                -32602
            };
            RpcResponse {
                jsonrpc: "2.0",
                result: None,
                error: Some(RpcError {
                    code,
                    message: msg,
                    data: None,
                }),
                id: req.id,
            }
        }
    }
}

pub fn dispatch_str(state: &RpcState, raw: &str) -> String {
    match serde_json::from_str::<RpcRequest>(raw) {
        Ok(req) => serde_json::to_string(&dispatch(state, req)).unwrap_or_else(|e| {
            json!({
                "jsonrpc": "2.0",
                "error": {"code": -32603, "message": e.to_string()},
                "id": null
            })
            .to_string()
        }),
        Err(e) => json!({
            "jsonrpc": "2.0",
            "error": {"code": -32700, "message": format!("Parse error: {e}")},
            "id": null
        })
        .to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_and_heartbeat() {
        let state = RpcState::new(60_000);
        let fit = dispatch_str(
            &state,
            r#"{"jsonrpc":"2.0","method":"capacity.fit","params":{"params":7000000000,"available_bytes":25769803776,"dtype":"F16"},"id":1}"#,
        );
        assert!(fit.contains("\"fits\":true"));
        let hb = dispatch_str(
            &state,
            r#"{"jsonrpc":"2.0","method":"device.heartbeat","params":{"node_id":"desk","device":"3090 Ti","cuda":true,"memory_gb":24},"id":2}"#,
        );
        assert!(hb.contains("\"ok\":true"));
    }
}
