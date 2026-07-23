use criterion::{criterion_group, criterion_main, Criterion};

fn find_subseq_brute_force(hay: &[u32], needle: &[u32]) -> Option<usize> {
    if needle.is_empty() || hay.len() < needle.len() {
        return None;
    }
    'outer: for i in 0..=hay.len() - needle.len() {
        for j in 0..needle.len() {
            if hay[i + j] != needle[j] {
                continue 'outer;
            }
        }
        return Some(i);
    }
    None
}

fn find_subseq_sliding(hay: &[u32], needle: &[u32]) -> Option<usize> {
    if needle.is_empty() || hay.len() < needle.len() {
        return None;
    }
    let mut match_len = 0usize;
    let mut start = 0usize;
    for (i, &h) in hay.iter().enumerate() {
        if h == needle[match_len] {
            if match_len == 0 {
                start = i;
            }
            match_len += 1;
            if match_len == needle.len() {
                return Some(start);
            }
        } else {
            match_len = 0;
        }
    }
    None
}

fn bench_find_subseq(c: &mut Criterion) {
    let hay: Vec<u32> = (0..10000).collect();
    let needle = vec![5000u32, 5001, 5002, 5003, 5004];

    let mut group = c.benchmark_group("find_subseq");
    group.bench_function("brute_force_10k", |b| {
        b.iter(|| find_subseq_brute_force(&hay, &needle))
    });
    group.bench_function("sliding_10k", |b| {
        b.iter(|| find_subseq_sliding(&hay, &needle))
    });

    let hay_100k: Vec<u32> = (0..100000).collect();
    let needle_100k = vec![50000u32, 50001, 50002];
    group.bench_function("brute_force_100k", |b| {
        b.iter(|| find_subseq_brute_force(&hay_100k, &needle_100k))
    });
    group.bench_function("sliding_100k", |b| {
        b.iter(|| find_subseq_sliding(&hay_100k, &needle_100k))
    });
    group.finish();
}

criterion_group!(benches, bench_find_subseq);
criterion_main!(benches);
