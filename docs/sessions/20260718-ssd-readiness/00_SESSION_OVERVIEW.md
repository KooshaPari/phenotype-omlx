# SSD Readiness Session Overview

## Goal

Define an honest, deterministic SSD readiness contract with two explicit gates:

- a hermetic CI gate that validates the processed-dataset path and JSONL schema using
  committed synthetic fixtures; and
- a release gate that validates complete, provenance-pinned upstream datasets before
  publishing performance or production-readiness claims.

The CI fixture is structural test data only. It must never be reported as benchmark data,
substituted for a release corpus, or used to claim SSD quality or throughput.

## Current Evidence

- `scripts/phenotype-omlx-ready` currently sets `SSD_HF_CACHE` and `SSD_DATASET_DIR`, then
  imports the CUDA reference. A missing dataset can therefore surface during an import that
  appears to be a generic stack check.
- The default local dataset root is `~/.cache/ssd/datasets`; none of the five expected
  processed files was present during the 2026-07-18 audit.
- Upstream `tanishqkumar/ssd` documents `SSD_DATASET_DIR` as the parent of dataset
  subdirectories and generates JSONL records with one `text` string field.
- The upstream preparation script emits dataset-specific files under `humaneval/`,
  `alpaca/`, `c4/`, `gsm8k/`, and `ultrafeedback/`. A requested `10000` is a processing cap,
  not proof that every source contains 10,000 rows; HumanEval is smaller.
- Upstream source mappings are: HumanEval `prompt`, Alpaca `instruction` plus optional
  `input`, C4 `text`, GSM8K `question`, and UltraFeedback `instruction`, each normalized to
  `{ "text": ... }`.

## Success Criteria

1. CI runs offline and deterministically against committed, clearly marked synthetic JSONL
   fixtures whose records contain exactly a non-empty string `text` field.
2. CI proves path discovery, JSONL parsing, schema rejection, empty-file rejection, and
   actionable diagnostics without requiring CUDA, Hugging Face access, or full corpora.
3. Release readiness requires every configured production corpus, its manifest, source
   dataset/revision, preprocessing command/version, row count, byte count, and SHA-256.
4. Release readiness fails closed for absent files, malformed records, manifest mismatch,
   checksum mismatch, unexpected source/revision, or fixture-marked data.
5. Output names the active gate (`ci-structure` or `release-data`), dataset root, per-dataset
   status, and a distinct non-zero exit result for failed structural versus release data
   validation.
6. No readiness path silently downloads data or treats CUDA unavailability as evidence that
   production datasets are valid.

## Scope

This session specifies dataset readiness only: fixture schema, full-corpus provenance,
validation behavior, reporting, and tests. It does not download datasets, add benchmark
results, change SSD inference behavior, or fabricate production data. Dataset acquisition is
an explicit operator/release workflow using the upstream preparation code and pinned source
revisions.
