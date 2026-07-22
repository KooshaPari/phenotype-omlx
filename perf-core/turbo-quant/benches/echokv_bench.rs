use std::time::Instant;
use turbo_quant::echokv::{EchoKVCache, EchoKVConfig};

const BENCH_ITERS: usize = 100_000;

fn bench_insert_throughput() {
    let config = EchoKVConfig {
        max_cache_size: 4096,
        ..Default::default()
    };
    let mut cache = EchoKVCache::new(config);

    let start = Instant::now();
    for i in 0..BENCH_ITERS {
        let attn = (i as f32 % 100.0) / 100.0;
        cache.insert(i, attn);
    }
    let elapsed = start.elapsed();
    let throughput = BENCH_ITERS as f64 / elapsed.as_secs_f64();
    let ns_per = elapsed.as_nanos() as f64 / BENCH_ITERS as f64;

    println!("[echokv_bench] EchoKVCache::insert throughput");
    println!("  entries:   {BENCH_ITERS}");
    println!("  total:     {:.3?}", elapsed);
    println!("  per_call:  {:.1} ns", ns_per);
    println!("  throughput: {:.0} entries/s", throughput);
    println!("  final_len: {}", cache.len());
    println!();
}

fn bench_evict_throughput() {
    let config = EchoKVConfig {
        max_cache_size: 100,
        eviction_threshold: 0.5,
        ..Default::default()
    };

    let iters = 50_000;
    let mut total_evicted = 0usize;

    let start = Instant::now();
    for _ in 0..iters {
        let mut cache = EchoKVCache::new(config.clone());
        // Insert 200 entries to force eviction
        for j in 0..200 {
            let attn = (j as f32 % 100.0) / 100.0;
            cache.insert(j, attn);
        }
        let evicted = cache.evict();
        total_evicted += evicted.len();
    }
    let elapsed = start.elapsed();
    let ns_per = elapsed.as_nanos() as f64 / iters as f64;

    println!("[echokv_bench] EchoKVCache::evict (200 entries, max=100)");
    println!("  iters:       {iters}");
    println!("  total:       {:.3?}", elapsed);
    println!("  per_call:    {:.1} ns", ns_per);
    println!("  avg_evicted: {:.1}", total_evicted as f64 / iters as f64);
    println!();
}

fn bench_ranked_entries(sizes: &[usize]) {
    for &size in sizes {
        let config = EchoKVConfig {
            max_cache_size: size,
            ..Default::default()
        };
        let mut cache = EchoKVCache::new(config);

        for i in 0..size {
            let attn = (i as f32 % 100.0) / 100.0;
            cache.insert(i, attn);
        }

        // Warmup
        let iters = if size <= 1024 { 10_000 } else { 2_000 };
        for _ in 0..iters {
            let _ = cache.ranked_entries();
        }

        let start = Instant::now();
        for _ in 0..iters {
            let _ = cache.ranked_entries();
        }
        let elapsed = start.elapsed();
        let ns = elapsed.as_nanos() as f64 / iters as f64;

        println!(
            "[echokv_bench] ranked_entries (size={size}): {:.1} ns/call  ({:.0} ops/s)",
            ns,
            iters as f64 / elapsed.as_secs_f64()
        );
    }
    println!();
}

fn main() {
    println!("=== turbo-quant EchoKV performance benchmarks ===\n");
    bench_insert_throughput();
    bench_evict_throughput();
    println!("[echokv_bench] ranked_entries by cache size:");
    bench_ranked_entries(&[64, 256, 1024, 4096]);
    println!("=== done ===");
}
