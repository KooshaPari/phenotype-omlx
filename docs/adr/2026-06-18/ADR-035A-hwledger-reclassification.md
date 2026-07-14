# ADR-035A: HwLedger Reclassification — Federated Service with Extractable pheno-capacity Lib

## Status
Accepted (inherited from hwLedger)

## Context
HwLedger was classified as a multi-stack federated service with a reusable Rust core
(capacity estimation / memory modeling). By ADR-035A the math core was extracted into
`pheno-capacity` while the UIs, sidecars, and tooling remained as the federated service.

With the merge into phenotype-omlx, those docs, ADRs, and design decisions are preserved
here so that the hwLedger fleet-logic and capacity-modeling work remains auditable.

## Related
- `docs/boundary/hwLedger.md` — original boundary file
- `docs/intent/hwLedger.md` — original intent file
