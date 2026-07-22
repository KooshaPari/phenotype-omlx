# SSD Readiness Specifications

## Gate Model

Readiness exposes two explicit, non-interchangeable modes:

- `ci-structure`: hermetic validation using committed synthetic fixtures. It proves path,
  parser, normalization, schema, and diagnostic behavior only.
- `release-data`: validation of complete, provenance-pinned processed corpora. It is required
  before SSD benchmark, performance, quality, or production-readiness claims.

The selected mode must appear in machine-readable and human-readable output. There is no
automatic downgrade from `release-data` to `ci-structure`.

## Required Dataset Set

Both modes cover exactly these logical dataset identifiers:

1. `humaneval`
2. `alpaca`
3. `c4`
4. `gsm8k`
5. `ultrafeedback`

Release mode resolves their configured processed files beneath `SSD_DATASET_DIR`. The initial
upstream-compatible names are `<dataset>/<dataset>_data_10000.jsonl`, but the manifest is the
source of truth because `10000` is a requested cap rather than a guaranteed row count.

## Normalized Record Contract

Each processed file is UTF-8 JSON Lines. Every non-blank line is one JSON object containing:

```json
{"text": "a non-empty prompt string"}
```

Requirements:

- `text` is required, is a string, and is non-empty after trimming.
- blank lines, malformed JSON, arrays, scalars, null records, and missing/invalid `text` fail.
- additional keys fail in `ci-structure` so schema drift is detected. Release manifests may
  explicitly authorize additional keys in a future schema version; implicit tolerance is
  forbidden.
- files must contain at least one valid record.
- validation reports the one-based failing line without echoing prompt contents.

## CI Synthetic Fixture Contract

The repository may contain one tiny synthetic file per logical dataset. Each fixture:

- follows the normalized record contract;
- contains only invented, non-sensitive text;
- is identified by manifest field `class: synthetic-structure-fixture`;
- carries no upstream dataset name or benchmark attribution;
- is small enough for offline validation on every supported platform; and
- must be rejected by `release-data`, regardless of filename or row count.

CI also creates temporary negative fixtures during tests for malformed JSON, empty files,
missing `text`, empty `text`, extra keys, incorrect checksums, and missing dataset entries.

## Release Manifest Contract

Release mode consumes a version-controlled manifest definition whose resolved artifact values
can be updated only through explicit dataset preparation. Conceptual schema:

```json
{
  "schema_version": 1,
  "class": "release-dataset-set",
  "processor": {
    "repository": "https://github.com/tanishqkumar/ssd",
    "revision": "full immutable commit SHA",
    "command": "python scripts/get_data_from_hf.py --num-samples 10000"
  },
  "datasets": {
    "humaneval": {
      "path": "humaneval/humaneval_data_10000.jsonl",
      "source": "openai/openai_humaneval",
      "source_revision": "immutable dataset revision",
      "config": null,
      "split": "test",
      "rows": 164,
      "bytes": 12345,
      "sha256": "64 lowercase hexadecimal characters"
    }
  }
}
```

The manifest contains all five dataset entries. Every entry requires relative path, exact
source identifier, immutable source revision, config when applicable, split, actual positive
row count, positive byte count, and SHA-256. Absolute paths, traversal components, symlinks
escaping `SSD_DATASET_DIR`, floating revisions, missing hashes, and duplicate resolved paths
fail validation.

The example byte count and digest are illustrative, never accepted defaults. HumanEval's row
count reflects its smaller source split; validators must compare against pinned manifest
facts rather than infer counts from filenames.

## Provenance and Security

- Readiness never downloads, regenerates, mutates, or repairs data.
- Acquisition is a separate operator workflow using the pinned SSD processor revision and
  pinned upstream dataset revisions.
- SHA-256 and byte/row counts are verified before release readiness passes.
- Validation reads only regular files contained by the resolved dataset root.
- Error output never prints prompt text, credentials, tokens, or arbitrary record contents.
- Fixture classification is explicit and cryptographically covered by its own manifest; a
  renamed fixture cannot pass the release gate.
- A Hugging Face model cache is not a processed dataset root.

## Failure and Reporting Contract

Human-readable output emits one summary line per dataset with gate, logical identifier,
status, resolved relative path, rows checked, bytes, and a stable reason code. Final output
includes selected gate and passed/required count.

Machine-readable output contains the same fields without stack traces. Stable reason codes
include:

- `DATA_ROOT_MISSING`
- `MANIFEST_MISSING` / `MANIFEST_INVALID`
- `DATASET_ENTRY_MISSING`
- `FILE_MISSING` / `FILE_OUTSIDE_ROOT` / `FILE_NOT_REGULAR`
- `JSON_INVALID` / `SCHEMA_INVALID` / `TEXT_EMPTY` / `FILE_EMPTY`
- `ROW_COUNT_MISMATCH` / `BYTE_COUNT_MISMATCH` / `CHECKSUM_MISMATCH`
- `SOURCE_UNPINNED` / `PROCESSOR_UNPINNED`
- `FIXTURE_FORBIDDEN_IN_RELEASE`

Any required dataset failure makes the gate fail. Platform/CUDA unavailability is reported
separately and cannot convert a dataset failure into success. Structural and release-data
failures have distinct non-zero CLI exit codes; usage/configuration errors use another code.

## Acceptance Criteria

1. `ci-structure` passes offline from a clean checkout on macOS and Linux using only committed
   synthetic fixtures.
2. Every negative fixture case fails with the expected stable reason code and dataset/line
   location, without leaking record text.
3. `release-data` rejects the CI fixtures and fails when any of five entries, files, immutable
   revisions, counts, or checksums is missing or mismatched.
4. A fully pinned, schema-valid five-dataset corpus passes and reports 5/5 with verified rows,
   bytes, and hashes.
5. No mode performs network access or filesystem mutation.
6. CUDA dependency status, SSD import status, CI structure readiness, and release dataset
   readiness are independently reported.

## ARUs: Assumptions, Risks, and Uncertainties

- **Assumption:** Upstream's current normalized consumer contract is one `text` string field.
  **Mitigation:** pin the processor commit and version this manifest/schema contract.
- **Risk:** Upstream changes filenames, source identifiers, or preprocessing. **Mitigation:**
  manifest paths and revisions are authoritative; schema changes require an explicit version.
- **Risk:** A small or fabricated file is renamed to look like a full corpus. **Mitigation:**
  release class, immutable provenance, actual counts, bytes, and SHA-256 all must match.
- **Risk:** Dataset licenses or redistribution terms prohibit committing full corpora.
  **Mitigation:** commit only synthetic fixtures and manifests; acquire release corpora through
  the authorized operator workflow.
- **Uncertainty:** Exact immutable Hugging Face dataset revisions and final artifact hashes are
  not yet recorded. **Mitigation:** release mode remains failing until an authorized acquisition
  run records and independently verifies them.
- **Risk:** Treating Apple-Silicon CUDA absence as overall SSD readiness hides data failures.
  **Mitigation:** platform capability and dataset gates remain orthogonal result dimensions.
