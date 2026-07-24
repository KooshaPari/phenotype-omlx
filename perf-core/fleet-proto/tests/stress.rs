use fleet_proto::{Fleet, Heartbeat, InMemoryFleet, NodeCapabilities};
use std::time::{SystemTime, UNIX_EPOCH};

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

fn caps() -> NodeCapabilities {
    NodeCapabilities {
        backends: vec!["mlx".into()],
        models: vec!["Qwen2.5-7B".into()],
        device: "stress-device".into(),
        memory_gb: 32.0,
        cuda: false,
        metal: true,
    }
}

fn hb(node_id: &str, ts_ms: u64) -> Heartbeat {
    Heartbeat {
        node_id: node_id.into(),
        addr: "127.0.0.1".into(),
        port: 9000,
        ts_ms,
        caps: caps(),
        inflight: 0,
    }
}

#[test]
fn stress_1000_peers_announce_and_list() {
    let fleet = InMemoryFleet::new(3_600_000);
    let ts = now_ms();

    for i in 0..1000 {
        fleet.announce(hb(&format!("peer-{}", i), ts)).unwrap();
    }

    let peers = fleet.peers();
    assert_eq!(peers.len(), 1000);
}

#[test]
fn stress_ttl_eviction_1000() {
    let fleet = InMemoryFleet::new(100);
    let stale_ts = 0;

    for i in 0..1000 {
        fleet
            .announce(hb(&format!("peer-{}", i), stale_ts))
            .unwrap();
    }

    let peers = fleet.peers();
    assert_eq!(peers.len(), 0, "All 1000 peers should be evicted by TTL");
}

#[test]
fn stress_remove_all_peers() {
    let fleet = InMemoryFleet::new(3_600_000);
    let ts = now_ms();

    for i in 0..500 {
        fleet.announce(hb(&format!("peer-{}", i), ts)).unwrap();
    }

    for i in 0..500 {
        fleet.remove(&format!("peer-{}", i)).unwrap();
    }

    assert_eq!(fleet.peers().len(), 0);
}

#[test]
fn stress_rapid_announce_remove_cycle() {
    let fleet = InMemoryFleet::new(3_600_000);
    let ts = now_ms();

    for cycle in 0..100 {
        for i in 0..10 {
            fleet
                .announce(hb(&format!("cycle-{}-peer-{}", cycle, i), ts))
                .unwrap();
        }
        let peers = fleet.peers();
        assert!(
            peers.len() <= 10 * (cycle + 1),
            "expected at most {} peers, got {}",
            10 * (cycle + 1),
            peers.len()
        );
    }
}

#[test]
fn stress_interleaved_read_write() {
    let fleet = InMemoryFleet::new(3_600_000);
    let ts = now_ms();

    for i in 0..200 {
        fleet.announce(hb(&format!("peer-{}", i), ts)).unwrap();
    }

    for i in 0..200 {
        if i % 3 == 0 {
            fleet.remove(&format!("peer-{}", i)).unwrap();
        }
        let _peers = fleet.peers();
    }

    let remaining = fleet.peers();
    assert_eq!(
        remaining.len(),
        200 - 67,
        "every 3rd of 200 peers removed = 67 removed, 133 remain"
    );
}

#[test]
fn stress_duplicate_announce_overwrites() {
    let fleet = InMemoryFleet::new(3_600_000);
    let ts = now_ms();

    for _ in 0..5000 {
        fleet.announce(hb("same-node", ts)).unwrap();
    }

    let peers = fleet.peers();
    assert_eq!(
        peers.len(),
        1,
        "5000 announces of same node should collapse to 1"
    );
}

#[test]
fn stress_ttl_boundary_fresh_vs_stale() {
    let fleet = InMemoryFleet::new(500);
    let ts = now_ms();

    for i in 0..1000 {
        let peer_ts = if i % 2 == 0 { ts } else { 0 };
        fleet.announce(hb(&format!("peer-{}", i), peer_ts)).unwrap();
    }

    let peers = fleet.peers();
    assert_eq!(peers.len(), 500, "only fresh peers (half) should survive");
}
