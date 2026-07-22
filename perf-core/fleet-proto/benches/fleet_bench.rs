use fleet_proto::{Fleet, Heartbeat, InMemoryFleet, NodeCapabilities};
use std::time::Instant;
use std::time::{SystemTime, UNIX_EPOCH};

const BENCH_ITERS: usize = 200_000;

fn caps() -> NodeCapabilities {
    NodeCapabilities {
        backends: vec!["mlx".into()],
        models: vec!["Qwen2.5-7B".into()],
        device: "bench-device".into(),
        memory_gb: 32.0,
        cuda: false,
        metal: true,
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

fn make_heartbeat(id: usize) -> Heartbeat {
    Heartbeat {
        node_id: format!("node-{}", id),
        addr: "127.0.0.1".into(),
        port: 8080,
        ts_ms: now_ms(),
        caps: caps(),
        inflight: 0,
    }
}

fn bench_add_peer_throughput() {
    let fleet = InMemoryFleet::new(60_000);

    let start = Instant::now();
    for i in 0..BENCH_ITERS {
        let hb = make_heartbeat(i);
        fleet.announce(hb).unwrap();
    }
    let elapsed = start.elapsed();
    let ns_per = elapsed.as_nanos() as f64 / BENCH_ITERS as f64;
    let throughput = BENCH_ITERS as f64 / elapsed.as_secs_f64();

    println!("[fleet_bench] InMemoryFleet::add_peer (announce) throughput");
    println!("  peers:      {BENCH_ITERS}");
    println!("  total:      {:.3?}", elapsed);
    println!("  per_call:   {:.1} ns", ns_per);
    println!("  throughput:  {:.0} ops/s", throughput);
    println!();
}

fn bench_peers_at_scale(counts: &[usize]) {
    for &count in counts {
        let fleet = InMemoryFleet::new(60_000);
        let ts = now_ms();
        for i in 0..count {
            let hb = Heartbeat {
                node_id: format!("node-{}", i),
                addr: "127.0.0.1".into(),
                port: 8080,
                ts_ms: ts,
                caps: caps(),
                inflight: 0,
            };
            fleet.announce(hb).unwrap();
        }

        let iters = match count {
            0..=100 => 50_000,
            101..=1000 => 20_000,
            _ => 5_000,
        };

        // Warmup
        for _ in 0..iters {
            let _ = fleet.peers();
        }

        let start = Instant::now();
        for _ in 0..iters {
            let _ = fleet.peers();
        }
        let elapsed = start.elapsed();
        let ns = elapsed.as_nanos() as f64 / iters as f64;

        println!(
            "[fleet_bench] peers() with {} peers: {:.1} ns/call  ({:.0} ops/s, {} returned)",
            count,
            ns,
            iters as f64 / elapsed.as_secs_f64(),
            fleet.peers().len()
        );
    }
    println!();
}

fn main() {
    println!("=== fleet-proto performance benchmarks ===\n");
    bench_add_peer_throughput();
    println!("[fleet_bench] peers() at scale:");
    bench_peers_at_scale(&[100, 1_000, 10_000]);
    println!("=== done ===");
}
