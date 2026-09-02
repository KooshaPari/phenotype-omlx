# Boundary: hwLedger (federated service)

**Repo:** `KooshaPari/hwLedger`
**Role in synthetic monolith:** spoke / federated service + optional OS GUIs
**Embed into omlx:** **no** (only `pheno-capacity` math embeds)

## In

- Hardware inventory persistence
- Fleet planner UI / sidecars
- Heartbeat publisher (via `hwledger-probe` plugin or sidecar)

## Out

- Inference engines (NanoVM plugins in omlx)
- Eval cockpit (bench-cockpit)
- Pure capacity math (`pheno-capacity` crate)

See ADR-035A + ADR-006.
