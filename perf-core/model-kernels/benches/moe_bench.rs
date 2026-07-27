use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use model_kernels::common::Lcg;
use model_kernels::moe::{
    grouped_gemm, grouped_gemm_tiled, moe_dispatch, router_topk, shared_expert, weighted_reduce,
    weighted_reduce_tiled,
};

fn gen_logits(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = Lcg::new(seed);
    (0..n).map(|_| rng.next_signed()).collect()
}

fn gen_matrix(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
    let mut rng = Lcg::new(seed);
    (0..rows * cols).map(|_| rng.next_signed()).collect()
}

fn bench_router_topk(c: &mut Criterion) {
    let mut group = c.benchmark_group("moe_router_topk");

    for (num_experts, top_k) in [(8, 2), (16, 2), (32, 4), (64, 8), (128, 8)] {
        let logits = gen_logits(num_experts, 0xCAFE);
        group.bench_with_input(
            BenchmarkId::new("router_topk", format!("{num_experts}e_{top_k}k")),
            &(num_experts, top_k, &logits),
            |bench, &(ne, tk, lg)| {
                bench.iter(|| black_box(router_topk(black_box(lg), ne, tk, 0).unwrap()))
            },
        );
    }
    group.finish();
}

fn bench_dispatch(c: &mut Criterion) {
    let mut group = c.benchmark_group("moe_dispatch");

    for num_tokens in [32, 128, 512] {
        let num_experts = 8;
        let top_k = 2;
        let logits = gen_logits(num_experts, 0xD15);
        let mut assignments = Vec::with_capacity(num_tokens * top_k);
        let mut token_indices = Vec::with_capacity(num_tokens * top_k);
        for t in 0..num_tokens {
            let picks = router_topk(&logits, num_experts, top_k, t as u64).unwrap();
            for (e, s) in picks {
                assignments.push((e, s));
                token_indices.push(t);
            }
        }
        group.bench_with_input(
            BenchmarkId::new("dispatch", format!("{num_tokens}tok")),
            &(&token_indices, &assignments, num_experts),
            |bench, &(ti, asgn, ne)| {
                bench.iter(|| {
                    black_box(moe_dispatch(black_box(ti), black_box(asgn), ne, 1.25).unwrap())
                })
            },
        );
    }
    group.finish();
}

fn bench_grouped_gemm(c: &mut Criterion) {
    let mut group = c.benchmark_group("moe_grouped_gemm");

    let shapes: &[(usize, usize, usize, usize)] = &[
        (32, 128, 128, 8),
        (128, 128, 128, 8),
        (128, 256, 256, 8),
        (512, 128, 128, 8),
    ];

    for &(num_tokens, k_dim, n_dim, num_experts) in shapes {
        let act = gen_matrix(num_tokens, k_dim, 0xA1);
        let wt = gen_matrix(num_experts * k_dim, n_dim, 0xB2);
        let buckets: Vec<Vec<usize>> = (0..num_experts)
            .map(|e| (0..num_tokens).filter(|&t| t % num_experts == e).collect())
            .collect();
        let label = format!("{num_tokens}tok_k{k_dim}_n{n_dim}");

        group.bench_with_input(
            BenchmarkId::new("scalar", &label),
            &(&act, &wt, &buckets, k_dim, n_dim, num_tokens),
            |bench, &(a_ref, b_ref, bk, k, n, m)| {
                let mut out = vec![0.0f32; num_tokens * n];
                bench.iter(|| {
                    out.fill(0.0);
                    grouped_gemm(
                        black_box(a_ref),
                        black_box(b_ref),
                        black_box(bk),
                        m,
                        k,
                        n,
                        &mut out,
                    )
                    .unwrap();
                    black_box(())
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("tiled", &label),
            &(&act, &wt, &buckets, k_dim, n_dim, num_tokens),
            |bench, &(a_ref, b_ref, bk, k, n, m)| {
                let mut out = vec![0.0f32; num_tokens * n];
                bench.iter(|| {
                    out.fill(0.0);
                    grouped_gemm_tiled(
                        black_box(a_ref),
                        black_box(b_ref),
                        black_box(bk),
                        m,
                        k,
                        n,
                        &mut out,
                    )
                    .unwrap();
                    black_box(())
                })
            },
        );
    }
    group.finish();
}

fn bench_weighted_reduce(c: &mut Criterion) {
    let mut group = c.benchmark_group("moe_weighted_reduce");

    for (num_tokens, experts_per_token, hidden) in
        [(32, 2, 128), (128, 2, 128), (128, 4, 256), (512, 2, 128)]
    {
        let expert_outs = gen_matrix(num_tokens * experts_per_token, hidden, 0xE1);
        let weights: Vec<f32> = {
            let mut rng = Lcg::new(0xE2);
            (0..num_tokens * experts_per_token)
                .map(|_| rng.next_f32())
                .collect()
        };
        let label = format!("{num_tokens}tok_e{experts_per_token}_h{hidden}");

        group.bench_with_input(
            BenchmarkId::new("scalar", &label),
            &(
                &expert_outs,
                &weights,
                experts_per_token,
                hidden,
                num_tokens,
            ),
            |bench, &(eo, w, ept, h, nt)| {
                let mut out = vec![0.0f32; nt * h];
                bench.iter(|| {
                    out.fill(0.0);
                    weighted_reduce(black_box(eo), black_box(w), ept, h, &mut out).unwrap();
                    black_box(())
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("tiled", &label),
            &(
                &expert_outs,
                &weights,
                experts_per_token,
                hidden,
                num_tokens,
            ),
            |bench, &(eo, w, ept, h, nt)| {
                let mut out = vec![0.0f32; nt * h];
                bench.iter(|| {
                    out.fill(0.0);
                    weighted_reduce_tiled(black_box(eo), black_box(w), ept, h, &mut out).unwrap();
                    black_box(())
                })
            },
        );
    }
    group.finish();
}

fn bench_shared_expert(c: &mut Criterion) {
    let mut group = c.benchmark_group("moe_shared_expert");

    for (m, k_dim, n_dim) in [
        (32, 128, 128),
        (128, 128, 128),
        (128, 256, 256),
        (512, 128, 128),
    ] {
        let x = gen_matrix(m, k_dim, 0xF1);
        let w = gen_matrix(k_dim, n_dim, 0xF2);
        let label = format!("{m}tok_k{k_dim}_n{n_dim}");

        group.bench_with_input(
            BenchmarkId::new("shared_expert", &label),
            &(&x, &w, m, n_dim),
            |bench, &(x_ref, w_ref, m_val, n_val)| {
                let mut out = vec![0.0f32; m_val * n_val];
                bench.iter(|| {
                    out.fill(0.0);
                    shared_expert(black_box(x_ref), black_box(w_ref), &mut out).unwrap();
                    black_box(())
                })
            },
        );
    }
    group.finish();
}

fn bench_full_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("moe_full_pipeline");

    let num_tokens = 128;
    let num_experts = 8;
    let top_k = 2;
    let k_dim = 128;
    let n_dim = 128;
    let hidden = 128;

    let logits = gen_logits(num_experts, 0x1001);
    let act = gen_matrix(num_tokens, k_dim, 0x2002);
    let wt = gen_matrix(num_experts * k_dim, n_dim, 0x3003);
    let expert_outs = gen_matrix(num_tokens * top_k, hidden, 0x4004);
    let mut weights_buf = vec![0.0f32; num_tokens * top_k];

    group.bench_function("router+dispatch+gemm+reduce", |bench| {
        bench.iter(|| {
            let mut all_assignments = Vec::with_capacity(num_tokens * top_k);
            let mut all_token_indices = Vec::with_capacity(num_tokens * top_k);
            for t in 0..num_tokens {
                let picks = router_topk(black_box(&logits), num_experts, top_k, t as u64).unwrap();
                for (e, s) in picks {
                    all_assignments.push((e, s));
                    all_token_indices.push(t);
                }
            }
            let plan =
                moe_dispatch(&all_token_indices, &all_assignments, num_experts, 1.25).unwrap();
            let mut gemm_out = vec![0.0f32; num_tokens * n_dim];
            grouped_gemm_tiled(
                black_box(&act),
                black_box(&wt),
                black_box(&plan.expert_buckets),
                num_tokens,
                k_dim,
                n_dim,
                &mut gemm_out,
            )
            .unwrap();

            for (i, &(_e, w)) in all_assignments.iter().enumerate() {
                weights_buf[i] = w;
            }
            let mut reduce_out = vec![0.0f32; num_tokens * hidden];
            weighted_reduce_tiled(
                black_box(&expert_outs),
                black_box(&weights_buf),
                top_k,
                hidden,
                &mut reduce_out,
            )
            .unwrap();
            black_box(&reduce_out);
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_router_topk,
    bench_dispatch,
    bench_grouped_gemm,
    bench_weighted_reduce,
    bench_shared_expert,
    bench_full_pipeline,
);
criterion_main!(benches);
