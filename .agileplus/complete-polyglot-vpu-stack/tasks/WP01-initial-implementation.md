---
work_package_id: WP01
title: Compile gate + FR inventory
feature: Complete the phenotype-omlx polyglot vPU performance stack
feature_slug: complete-polyglot-vpu-stack
sequence: 1
state: in_progress
created_at: 2026-07-19T00:00:00Z
updated_at: 2026-07-19T07:00:00Z
---

# Work Package: Compile gate + FR inventory

## Feature
Complete the phenotype-omlx polyglot vPU performance stack (`complete-polyglot-vpu-stack`)

## Acceptance Criteria
- [x] FR→path matrix in `.agileplus/.../research.md`
- [x] Fix highest-leverage compile/test blocker with regression test
- [ ] Full `cargo test` perf-core green (blocked: `regress-baseline` dispatch_buckets envelope — out of scope)
- [ ] Split follow-on WPs in plan.md

## Changes (WP01)
1. `perf-core/kernel-registry/Cargo.toml` — enable `serde_json/float_roundtrip`
2. `perf-core/kernel-registry/src/quality.rs` — regression test `content_hash_survives_serde_round_trip_with_non_shortest_float`

## Verify
```bash
cd perf-core
cargo test -p kernel-registry --test governance_fuzz
cargo test -p kernel-registry content_hash_survives
cargo test -p eval-harness
```

## Proposed commit message (not committed — dirty unrelated WIP on branch)
```
fix(kernel-registry): preserve promotion content_hash across JSON float round-trip

Enable serde_json float_roundtrip so QualityEvidence scores deserialize
with the same bits used when the content hash was computed. Adds a
regression test for the proptest-found ULP drift case.
```

## Next
Proceed to WP02 (eval backend wiring) and WP04 (FFI worktree merge) in parallel.
