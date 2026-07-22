# SSD Readiness DAG and WBS

## Forward DAG

```text
A. normalized schema contract
├── B. committed synthetic CI fixtures
│   └── D. fixture manifest
└── C. release manifest schema
    └── E. manifest and JSONL validator
        ├── F. CI structure gate  <── B + D
        └── G. full release data gate
            └── H. pinned dataset acquisition (operator workflow)
                └── I. five verified processed corpora

F + G policy selection
└── J. readiness CLI contract
    ├── K. platform and SSD import checks (orthogonal)
    ├── L. Cargo and PyO3 build checks
    └── M. fresh-wheel environment
        ├── N. wheel import and FFI roundtrip
        └── O. selected dataset gate rerun
            └── P. release/readiness evidence bundle
```

`ci-structure` follows `A -> B -> D -> E -> F -> J -> M -> O` and remains hermetic.
`release-data` follows `A -> C -> E -> H -> I -> G -> J -> M -> O -> P` and never falls
back to fixtures.

## Work Breakdown

| ID | Deliverable | Depends on | Validation | Estimate |
|---|---|---|---|---|
| A | Versioned `{ "text": string }` JSONL contract | none | schema examples reviewed | 0.25 d |
| B | Five invented positive fixtures | A | offline parser pass | 0.25 d |
| C | Release manifest model | A | invalid/path traversal cases fail | 0.5 d |
| D | Fixture manifest with explicit synthetic class | B | release gate rejects it | 0.25 d |
| E | Pure validator: paths, JSONL, rows, bytes, SHA-256, provenance | A, C | focused unit suite | 1.0 d |
| F | `ci-structure` gate | B, D, E | clean offline CI pass | 0.25 d |
| H | Authorized, pinned acquisition procedure | C | revisions and command captured | 0.5 d |
| I | Five processed release artifacts and resolved manifest | H | independent hashes/counts | external |
| G | `release-data` gate | E, I | 5/5 or fail closed | 0.25 d |
| J | Readiness command modes, JSON output, stable exit codes | F, G policy | CLI contract tests | 0.75 d |
| K | Separate platform/import dimension | J | Apple/CUDA matrix tests | 0.25 d |
| L | Existing Cargo/PyO3 checks integrated without coupling | J | build checks pass | 0.25 d |
| M | Fresh-wheel environment creation | J, L | no source-tree imports | 0.5 d |
| N | Installed-wheel import and FFI roundtrip | M | numerical roundtrip pass | 0.25 d |
| O | Selected dataset gate in fresh environment | M, F or G | mode retained in report | 0.25 d |
| P | Immutable evidence bundle | N, O | manifest/report hashes recorded | 0.25 d |

## Test Work Packages

1. Positive fixtures for all five logical dataset identifiers.
2. Negative parser cases: malformed JSON, blank/empty file, missing/empty/non-string `text`,
   unexpected keys, and non-object rows.
3. Filesystem cases: absent root/file, absolute or escaping path, symlink escape, non-regular
   file, and duplicate resolved paths.
4. Manifest cases: missing dataset, unknown schema/class, floating revisions, count/size/hash
   mismatch, and synthetic fixture submitted to release mode.
5. CLI cases: explicit mode required, stable reason codes, distinct non-zero exits, JSON/human
   parity, no prompt leakage, and no network/filesystem writes.
6. Fresh-wheel case: build/install in empty venv, import canonical module, FFI roundtrip, then
   rerun the selected dataset gate with no source checkout on `PYTHONPATH`.

## Critical Path

```text
A -> C -> E -> H -> I -> G -> J -> M -> O -> P
```

The external critical-path item is `I`: no qualifying local processed corpus exists. Code may
complete through deterministic `ci-structure`, but release readiness remains red until the
authorized five-corpus acquisition produces independently verified manifest facts.

## Parallel Lanes

- `B + D + F` can proceed while release provenance is prepared.
- `K` and `L` can proceed after the CLI result model is stable.
- Negative validator tests can run in parallel with fixture creation.
- Fresh-wheel scaffolding can begin after CLI shape and wheel layout are fixed, but final `O`
  depends on the selected data gate.

## Completion Rules

- Structural CI completion is not release completion.
- Missing production data is reported as an external gate, not patched with generated rows.
- No task downloads or mutates corpora during readiness execution.
- `P` is complete only when its report identifies gate, artifact hashes, source revisions,
  wheel hash, tool versions, and every per-dataset result.
