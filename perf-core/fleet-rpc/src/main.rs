//! fleet-rpc — JSON-RPC serve + heartbeat file/NATS bridge.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use fleet_proto::{dispatch_str, RpcState};
use serde_json::json;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};

#[derive(Parser, Debug)]
#[command(name = "fleet-rpc", about = "Federated monolith RPC + heartbeat bridge")]
struct Cli {
    #[command(subcommand)]
    cmd: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Line-delimited JSON-RPC 2.0 on TCP (default 127.0.0.1:7070)
    Serve {
        #[arg(long, default_value = "127.0.0.1:7070")]
        bind: String,
    },
    /// One-shot RPC call from CLI
    Call {
        #[arg(long, default_value = "127.0.0.1:7070")]
        addr: String,
        method: String,
        #[arg(long, default_value = "{}")]
        params: String,
    },
    /// Probe GPUs → write heartbeat.json + append jsonl; try NATS if up
    Bridge {
        #[arg(long, default_value = "platform/federation/out/heartbeat.json")]
        out: PathBuf,
        #[arg(long, default_value = "platform/federation/out/heartbeats.jsonl")]
        jsonl: PathBuf,
        #[arg(long, default_value = "nats://127.0.0.1:4222")]
        nats: String,
        /// Also POST into local fleet-rpc serve (if running)
        #[arg(long)]
        rpc: Option<String>,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Commands::Serve { bind } => serve(&bind).await?,
        Commands::Call {
            addr,
            method,
            params,
        } => {
            let p: serde_json::Value = serde_json::from_str(&params)?;
            let req = json!({
                "jsonrpc": "2.0",
                "method": method,
                "params": p,
                "id": 1
            });
            let resp = rpc_roundtrip(&addr, &req.to_string()).await?;
            println!("{resp}");
        }
        Commands::Bridge {
            out,
            jsonl,
            nats,
            rpc,
        } => bridge(&out, &jsonl, &nats, rpc.as_deref()).await?,
    }
    Ok(())
}

async fn serve(bind: &str) -> Result<()> {
    let state = Arc::new(RpcState::new(120_000));
    let listener = TcpListener::bind(bind)
        .await
        .with_context(|| format!("bind {bind}"))?;
    eprintln!("fleet-rpc listening on {bind} (newline JSON-RPC)");
    loop {
        let (sock, peer) = listener.accept().await?;
        let state = Arc::clone(&state);
        tokio::spawn(async move {
            if let Err(e) = handle_conn(sock, state).await {
                eprintln!("conn {peer}: {e:#}");
            }
        });
    }
}

async fn handle_conn(sock: TcpStream, state: Arc<RpcState>) -> Result<()> {
    let (r, mut w) = sock.into_split();
    let mut lines = BufReader::new(r).lines();
    while let Some(line) = lines.next_line().await? {
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }
        let out = dispatch_str(state.as_ref(), &line);
        w.write_all(out.as_bytes()).await?;
        w.write_all(b"\n").await?;
    }
    Ok(())
}

async fn rpc_roundtrip(addr: &str, req: &str) -> Result<String> {
    let addr: SocketAddr = addr.parse()?;
    let mut stream = TcpStream::connect(addr).await?;
    stream.write_all(req.as_bytes()).await?;
    stream.write_all(b"\n").await?;
    let mut lines = BufReader::new(stream).lines();
    let resp = lines
        .next_line()
        .await?
        .context("no response from fleet-rpc")?;
    Ok(resp)
}

fn nvidia_snapshot() -> serde_json::Value {
    let candidates = [
        "nvidia-smi",
        r"C:\Windows\System32\nvidia-smi.exe",
        r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
    ];
    for bin in candidates {
        let out = Command::new(bin)
            .args([
                "--query-gpu=index,name,uuid,memory.total,memory.free,memory.used,utilization.gpu,driver_version",
                "--format=csv,noheader,nounits",
            ])
            .output();
        let Ok(o) = out else { continue };
        if !o.status.success() {
            continue;
        }
        let text = String::from_utf8_lossy(&o.stdout);
        let mut gpus = Vec::new();
        for line in text.lines() {
            let parts: Vec<_> = line.split(',').map(|s| s.trim()).collect();
            if parts.len() >= 8 {
                let name = parts[1];
                let name_s: &str = name;
                let cc = if name_s.contains("1080") || name_s.contains("Pascal") {
                    "6.1"
                } else if name_s.contains("2080") || name_s.contains("T4") {
                    "7.5"
                } else if name_s.contains("3080") || name_s.contains("3090") {
                    "8.6"
                } else if name_s.contains("4080") || name_s.contains("4090") {
                    "8.9"
                } else if name_s.contains("5090") {
                    "12.0"
                } else {
                    "unknown"
                };
                let cuda_max = if name_s.contains("1080") || name_s.contains("Pascal") {
                    "12.9"
                } else {
                    "13.4"
                };
                let lanes_supported: Vec<&str> = if name_s.contains("1080") || name_s.contains("Pascal") {
                    vec!["pytorch-cu129", "triton-61", "flashinfer-jit-cu129", "aphrodite-pascal", "directml"]
                } else {
                    vec!["sglang", "vllm", "pytorch-cu13", "triton", "flashinfer", "flashattn"]
                };
                let lanes_blocked: Vec<&str> = if name_s.contains("1080") || name_s.contains("Pascal") {
                    vec!["pytorch-cu128", "vllm (sm_70+)", "flashattn2 (sm_80+)", "tensorrt (sm_75+)"]
                } else {
                    vec![]
                };
                gpus.push(json!({
                    "nvidia_smi_index": parts[0].parse::<u32>().unwrap_or(0),
                    "name": name,
                    "uuid": parts[2],
                    "memory_total": format!("{} MiB", parts[3]),
                    "memory_free": format!("{} MiB", parts[4]),
                    "memory_used": format!("{} MiB", parts[5]),
                    "utilization_gpu": format!("{} %", parts[6]),
                    "driver": parts[7],
                    "compute_cap": cc,
                    "cuda_runtime_max": cuda_max,
                    "lanes_supported": lanes_supported,
                    "lanes_blocked": lanes_blocked,
                }));
            }
        }
        if !gpus.is_empty() {
            return json!(gpus);
        }
    }
    json!([])
}

fn wsl_snapshot() -> serde_json::Value {
    // Probe default WSL distro (Windows host only). On Linux this is a no-op.
    let out = if cfg!(windows) {
        Command::new("wsl").args(["-l", "-v", "--all"]).output().ok()
    } else {
        None
    };
    let Some(o) = out else { return json!(null); };
    if !o.status.success() { return json!(null); }
    let text = String::from_utf8_lossy(&o.stdout);
    // First line is header ("  NAME      STATE    VERSION"); skip, then first row "*Fedora..." is the default.
    let mut distros: Vec<String> = Vec::new();
    let mut default_distro: Option<String> = None;
    for (i, line) in text.lines().enumerate() {
        if i == 0 { continue; }
        let trimmed = line.trim().trim_start_matches('*').trim();
        if trimmed.is_empty() { continue; }
        let name = trimmed.split_whitespace().next().unwrap_or("").to_string();
        if name.is_empty() { continue; }
        if line.trim_start().starts_with('*') && default_distro.is_none() {
            default_distro = Some(name.clone());
        }
        distros.push(name);
    }
    json!({
        "default": default_distro.unwrap_or_else(|| "unknown".into()),
        "distros": distros,
    })
}

async fn bridge(out: &Path, jsonl: &Path, nats_url: &str, rpc: Option<&str>) -> Result<()> {
    let host = hostname();
    let gpus = nvidia_snapshot();
    let wsl = wsl_snapshot();
    let wsl_default = wsl.get("default").and_then(|v| v.as_str()).unwrap_or("unknown");
    let payload = json!({
        "schema": "pheno.device.heartbeat/v1",
        "ts": chrono_now(),
        "host": host,
        "platform": std::env::consts::OS,
        "machine": std::env::consts::ARCH,
        "gpus": gpus,
        "wsl_distro": wsl_default,
        "wsl": wsl,
        "tb2_terminus_active_lane": std::env::var("TB2_TERMINUS_LANE").unwrap_or_else(|_| "cuda:0 (3090 Ti) SGLang Qwen3.5-9B :30000".into()),
        "tb2_terminus_policy": std::env::var("TB2_TERMINUS_POLICY").unwrap_or_else(|_| "local-only, no omniroute".into()),
        "source": "fleet-rpc-bridge",
        "subject": "pheno.device.heartbeat",
    });

    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    if let Some(parent) = jsonl.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(out, serde_json::to_string_pretty(&payload)?)?;
    {
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(jsonl)?;
        writeln!(f, "{}", serde_json::to_string(&payload)?)?;
    }
    eprintln!("wrote {}", out.display());
    eprintln!("appended {}", jsonl.display());

    // File-bus stand-in when Docker/NATS is down
    let bus = out
        .parent()
        .unwrap_or(Path::new("."))
        .join("nats-bus")
        .join("pheno.device.heartbeat.jsonl");
    if let Some(p) = bus.parent() {
        std::fs::create_dir_all(p)?;
    }
    {
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&bus)?;
        writeln!(f, "{}", serde_json::to_string(&payload)?)?;
    }
    eprintln!("file-bus {}", bus.display());

    if let Some(addr) = rpc {
        let mem = gpus
            .as_array()
            .and_then(|a| a.first())
            .and_then(|g| g.get("memory_total"))
            .and_then(|v| v.as_str())
            .and_then(|s| s.replace(" MiB", "").trim().parse::<f64>().ok())
            .map(|mib| mib / 1024.0)
            .unwrap_or(0.0);
        let name = gpus
            .as_array()
            .and_then(|a| a.first())
            .and_then(|g| g.get("name"))
            .and_then(|v| v.as_str())
            .unwrap_or("gpu");
        let req = json!({
            "jsonrpc": "2.0",
            "method": "device.heartbeat",
            "params": {
                "node_id": host,
                "device": name,
                "memory_gb": mem,
                "cuda": true,
                "backends": ["sglang", "vllm", "pytorch"]  // llama-cpp-cuda removed 2026-07-25
            },
            "id": 1
        });
        match rpc_roundtrip(addr, &req.to_string()).await {
            Ok(r) => eprintln!("rpc ok: {r}"),
            Err(e) => eprintln!("rpc skip: {e:#}"),
        }
    }

    // Live NATS publish (Podman pheno-nats); file-bus remains durable backup
    match publish_nats(nats_url, "pheno.device.heartbeat", &payload).await {
        Ok(()) => eprintln!("nats ok → {nats_url} subject=pheno.device.heartbeat"),
        Err(e) => eprintln!("nats skip: {e:#} (file-bus still written)"),
    }

    Ok(())
}

async fn publish_nats(url: &str, subject: &str, payload: &serde_json::Value) -> Result<()> {
    let client = async_nats::connect(url)
        .await
        .with_context(|| format!("connect {url}"))?;
    let bytes = bytes::Bytes::from(serde_json::to_vec(payload)?);
    client
        .publish(subject.to_string(), bytes)
        .await
        .context("publish")?;
    client.flush().await.context("flush")?;
    Ok(())
}

fn hostname() -> String {
    std::env::var("COMPUTERNAME")
        .or_else(|_| std::env::var("HOSTNAME"))
        .unwrap_or_else(|_| "unknown".into())
}

fn chrono_now() -> String {
    // RFC3339-ish without extra crate
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("{secs}")
}
