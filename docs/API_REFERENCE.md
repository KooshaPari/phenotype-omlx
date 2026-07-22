# phenotype-omlx API Reference

## `spec-decode`
High-performance speculative decoding engine for Apple Silicon.

`EngineState` — `{ history: Vec<AcceptedToken>, pending_drafts: usize }`.
`MedusaProposal` — `{ heads: Vec<MedusaHead>, tree: TreeTopology }`.
`dedup_preserve(proposals) -> usize` — remove dupes preserving order.

## `fleet-proto`
Fleet discovery and heartbeat protocol for multi-node OMLX clusters.

`InMemoryFleet` — TTL-based in-memory peer registry.
`Heartbeat` — `{ node_id, addr, port, ts_ms, caps, inflight }`.
`Fleet` trait — `announce(hb)`, `peers() -> Vec<Heartbeat>`, `remove(node_id)`.

```rust
let fleet = InMemoryFleet::new(5_000);
fleet.announce(heartbeat).unwrap();
let peers = fleet.peers();
```

## `kernel-registry`
`KernelRegistry` — candidate storage, lookup, selection traces.
`CandidateId(KernelKey, BackendKind)` — unique candidate identifier.
`Candidate` — `{ id, capabilities, record }`.
`SelectionPolicy` — `BestMetric | FastestCompile | LowestMemory`.

## `metal-runtime`
`flow_cfg_step_metal(pipeline, inputs) -> Result<StepOutput, FlowStepError>`.
`FlowStepError` — `Compile | Device | Timeout`.
`Pipeline` — compiled Metal pipeline handle.
