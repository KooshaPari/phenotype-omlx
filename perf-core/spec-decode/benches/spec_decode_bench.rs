use spec_decode::{
    create_parallel_trees, dedup_preserve, DraftTree, EngineState, ParallelTreeConfig,
};
use std::time::Instant;

const WARMUP_ITERS: usize = 1000;
const BENCH_ITERS: usize = 100_000;

fn bench_push_accepted() {
    let mut state = EngineState::new();
    // Pre-fill to capacity so every push triggers pop_front + push_back
    for i in 0..1024u32 {
        state.push_accepted(i);
    }

    let start = Instant::now();
    for i in 0..BENCH_ITERS {
        state.push_accepted(i as u32);
    }
    let elapsed = start.elapsed();
    let ns_per_call = elapsed.as_nanos() as f64 / BENCH_ITERS as f64;
    let throughput = BENCH_ITERS as f64 / elapsed.as_secs_f64();

    println!(
        "[spec_decode_bench] push_accepted (VecDeque, cap={})",
        spec_decode::HISTORY_CAP
    );
    println!("  iters:     {BENCH_ITERS}");
    println!("  total:     {:.3?}", elapsed);
    println!("  per_call:  {:.1} ns", ns_per_call);
    println!("  throughput: {:.0} ops/s", throughput);
    println!();
}

fn bench_dedup_preserve_vs_naive() {
    let input: Vec<u32> = (0..256).cycle().take(1024).collect();

    // Warmup
    for _ in 0..WARMUP_ITERS {
        let _ = dedup_preserve(input.clone());
    }

    // HashSet-based (our impl)
    let start = Instant::now();
    for _ in 0..BENCH_ITERS {
        let _ = dedup_preserve(input.clone());
    }
    let elapsed_hashset = start.elapsed();
    let ns_hashset = elapsed_hashset.as_nanos() as f64 / BENCH_ITERS as f64;

    // Naive Vec-based approach for comparison
    let naive_dedup = |xs: Vec<u32>| -> Vec<u32> {
        let mut out = Vec::with_capacity(xs.len());
        for x in xs {
            if !out.contains(&x) {
                out.push(x);
            }
        }
        out
    };

    let start = Instant::now();
    for _ in 0..BENCH_ITERS {
        let _ = naive_dedup(input.clone());
    }
    let elapsed_naive = start.elapsed();
    let ns_naive = elapsed_naive.as_nanos() as f64 / BENCH_ITERS as f64;

    println!("[spec_decode_bench] dedup_preserve — HashSet vs naive Vec (1024 items, 256 unique)");
    println!(
        "  HashSet:   {:.1} ns/call  ({:.0} ops/s)",
        ns_hashset,
        BENCH_ITERS as f64 / elapsed_hashset.as_secs_f64()
    );
    println!(
        "  naive Vec: {:.1} ns/call  ({:.0} ops/s)",
        ns_naive,
        BENCH_ITERS as f64 / elapsed_naive.as_secs_f64()
    );
    println!("  speedup:   {:.1}x", ns_naive / ns_hashset);
    println!();
}

fn bench_draft_tree_from_eagle3() {
    // Realistic logits: 4 depths, 3 candidates each
    let logits: Vec<Vec<(u32, f32)>> = vec![
        vec![(1, 0.5), (2, 0.3), (3, 0.2)],
        vec![(10, 0.4), (11, 0.35), (12, 0.25)],
        vec![(20, 0.6), (21, 0.3), (22, 0.1)],
        vec![(30, 0.7), (31, 0.2), (32, 0.1)],
    ];

    let iters = 50_000;

    // Warmup
    for _ in 0..WARMUP_ITERS {
        let _ = DraftTree::from_eagle3_predictions(0, logits.clone(), 8, 3);
    }

    let start = Instant::now();
    for _ in 0..iters {
        let tree = DraftTree::from_eagle3_predictions(0, logits.clone(), 8, 3);
        std::hint::black_box(tree.node_count());
    }
    let elapsed = start.elapsed();
    let ns = elapsed.as_nanos() as f64 / iters as f64;

    println!("[spec_decode_bench] DraftTree::from_eagle3_predictions (4 depths × 3 branches)");
    println!("  iters:     {iters}");
    println!("  total:     {:.3?}", elapsed);
    println!("  per_call:  {:.1} ns", ns);
    println!();
}

fn bench_create_parallel_trees() {
    let logits: Vec<Vec<(u32, f32)>> = vec![
        vec![
            (1, 0.4),
            (2, 0.3),
            (3, 0.2),
            (4, 0.1),
            (5, 0.35),
            (6, 0.25),
            (7, 0.2),
            (8, 0.1),
            (9, 0.3),
            (10, 0.25),
            (11, 0.2),
            (12, 0.15),
        ],
        vec![(20, 0.5), (21, 0.3), (22, 0.2)],
        vec![(30, 0.6), (31, 0.3), (32, 0.1)],
    ];
    let config = ParallelTreeConfig {
        num_parallel_branches: 4,
        max_depth: 8,
        max_branches_per_node: 3,
        probability_threshold: 0.01,
    };

    let iters = 20_000;

    // Warmup
    for _ in 0..WARMUP_ITERS {
        let _ = create_parallel_trees(0, logits.clone(), &config);
    }

    let start = Instant::now();
    for _ in 0..iters {
        let trees = create_parallel_trees(0, logits.clone(), &config);
        std::hint::black_box(trees.len());
    }
    let elapsed = start.elapsed();
    let ns = elapsed.as_nanos() as f64 / iters as f64;

    println!("[spec_decode_bench] create_parallel_trees (12 root candidates, 4 branches, depth 8)");
    println!("  iters:     {iters}");
    println!("  total:     {:.3?}", elapsed);
    println!("  per_call:  {:.1} ns", ns);
    println!();
}

fn main() {
    println!("=== spec-decode performance benchmarks ===\n");
    bench_push_accepted();
    bench_dedup_preserve_vs_naive();
    bench_draft_tree_from_eagle3();
    bench_create_parallel_trees();
    println!("=== done ===");
}
