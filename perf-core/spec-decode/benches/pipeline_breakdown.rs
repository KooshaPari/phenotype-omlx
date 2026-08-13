use criterion::{criterion_group, criterion_main, Criterion};

fn bench_pipeline_breakdown(c: &mut Criterion) {
    let mut group = c.benchmark_group("pipeline_breakdown");

    // 1. Token embedding lookup (simulated — batch of 32 random lookups in 151K vocab)
    group.bench_function("token_embedding", |b| {
        let vocab: Vec<u32> = (0..151936).collect();
        let indices: Vec<usize> = (0..32).map(|i| (i * 4733) % 151936).collect();
        b.iter(|| {
            for &idx in &indices {
                let _ = vocab[idx];
            }
        })
    });

    // 2. Proposal generation
    let logits: Vec<(u32, f32)> = (0..128).map(|i| (i, 1.0 / 128.0)).collect();
    group.bench_function("proposal_sort_128", |b| {
        b.iter(|| {
            let mut l = logits.clone();
            l.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            l
        })
    });

    // 3. Tree construction
    group.bench_function("tree_construction", |b| {
        b.iter(|| {
            let branch_logits = vec![
                vec![(1u32, 0.5), (2, 0.3), (3, 0.2)],
                vec![(4, 0.4), (5, 0.4), (6, 0.2)],
                vec![(7, 0.3), (8, 0.4), (9, 0.3)],
            ];
            spec_decode::tree_proposal::DraftTree::from_eagle3_predictions(0, branch_logits, 3, 3)
        })
    });

    // 4. Subsequence search
    group.bench_function("find_subseq_1k", |b| {
        let hay: Vec<u32> = (0..1000).collect();
        let needle = [500u32, 501, 502];
        b.iter(|| {
            let mut ml = 0usize;
            let mut s = 0usize;
            for (i, &h) in hay.iter().enumerate() {
                if h == needle[ml] {
                    if ml == 0 {
                        s = i;
                    }
                    ml += 1;
                    if ml == needle.len() {
                        return Some(s);
                    }
                } else {
                    ml = 0;
                }
            }
            None::<usize>
        })
    });

    // 5. Token dedup
    group.bench_function("dedup_256", |b| {
        let tokens: Vec<u32> = (0..256).flat_map(|x| [x, x, x]).collect();
        b.iter(|| {
            let mut seen = std::collections::HashSet::new();
            tokens
                .iter()
                .filter(|t| seen.insert(**t))
                .cloned()
                .collect::<Vec<u32>>()
        })
    });

    group.finish();
}

criterion_group!(benches, bench_pipeline_breakdown);
criterion_main!(benches);
