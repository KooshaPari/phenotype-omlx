# Eval Interchange Contract v1.0

**Status:** Proposed
**Created:** 2026-07-21
**Producer:** pheno-harness
**Consumer:** phenotype-omlx eval-harness

---

## Purpose

Define a shared JSON schema that pheno-harness emits as benchmark result contracts and phenotype-omlx's eval-harness consumes for ingestion, comparison, and regression detection.

---

## Schema Definition

### Required Fields (producer MUST emit)

```jsonc
{
  // --- Envelope ---
  "contract_version": "1.0",          // REQUIRED -- semver string
  "artifact_kind": "EvaluationReport", // REQUIRED -- literal

  // --- Producer Provenance ---
  "producer": {
    "name": "pheno-harness",           // REQUIRED -- repo name
    "version": "5.0.0",               // REQUIRED -- semver of producer
    "commit_sha": "abc123..."          // REQUIRED -- git SHA (40 hex chars)
  },

  // --- Run Metadata ---
  "run": {
    "run_id": "00000000-...",          // REQUIRED -- UUID v4
    "started_at": "2026-01-01T00:00:00Z", // REQUIRED -- ISO 8601
    "model": "Qwen/Qwen3.5-0.8B",     // REQUIRED -- model identifier
    "variant": "stock",                // REQUIRED -- "stock" | "ours"
    "judge_mode": "deterministic"      // REQUIRED -- "deterministic" | "llm" | "hybrid"
  },

  // --- Suite Results ---
  "suites": [
    {
      "suite": "mmlu-pro",            // REQUIRED -- suite name
      "n": 25,                         // REQUIRED -- task count
      "passed": 25,                    // REQUIRED -- passed count
      "pass_at_1": 1.0,               // REQUIRED -- float [0.0, 1.0]
      "evidence_label": "live_verified" // REQUIRED -- evidence provenance
    }
  ],

  // --- Aggregated Totals ---
  "totals": {
    "cells": 500,                      // REQUIRED -- total cells across suites
    "passed": 500,                     // REQUIRED -- total passed
    "pass_at_1": 1.0                   // REQUIRED -- aggregate pass@1
  },

  // --- Integrity ---
  "hash_chain": {
    "top_level_sha256": "aa8d32...",   // REQUIRED -- SHA-256 of canonical JSON minus hash_chain
    "task_ids_sorted_sha256": "1605..." // REQUIRED -- SHA-256 of sorted task_id list
  }
}
```

### Optional Fields (producer MAY emit, consumer SHOULD ignore unknown)

```jsonc
{
  "matrix": { ... },       // experimental matrix metadata
  "comparator": { ... },   // A/B comparison results
  // additional properties on any nested object are allowed
}
```

---

## Evidence Labels

| Label | Meaning |
|-------|---------|
| `live_verified` | Result observed against running inference endpoint |
| `reported` | Result self-reported by producer (not independently verified) |
| `synthetic` | Result from synthetic/mock execution |

Consumer SHOULD warn when `evidence_label != "live_verified"`.

---

## Hash Chain Specification

1. **`top_level_sha256`**: Compute SHA-256 over the canonical JSON of the entire document *excluding* the `hash_chain` key. Canonical form: sorted keys, no whitespace, UTF-8.
2. **`task_ids_sorted_sha256`**: Collect all `task_id` values from all suites, sort lexicographically, join with newline, compute SHA-256 of the resulting string.

---

## Consumer Behavior

| Rule | Level | Description |
|------|-------|-------------|
| R-VERSION | MUST reject | If `contract_version` is missing or not `"1.0"` |
| R-HASHCHAIN | MUST reject | If `hash_chain` verification fails |
| R-PRODUCER | MUST reject | If `producer` block is missing required fields |
| R-SUITES | MUST reject | If `suites` is empty or missing |
| R-TOTALS | MUST reject | If `totals` is missing |
| W-EVIDENCE | SHOULD warn | If any `evidence_label` is not `"live_verified"` |
| W-UNKNOWN | MAY ignore | Unknown additional properties at any level |

---

## Migration from v0.1

The V5 contract (`contract_version: "0.1"`) does not conform to v1.0. Key differences:

| v0.1 field | v1.0 equivalent | Notes |
|------------|-----------------|-------|
| `producer.repo` | `producer.name` | Renamed |
| `producer.head` | `producer.commit_sha` | Renamed |
| `producer.branch` | (removed) | Not in contract spec |
| `producer.dirty_paths` | (removed) | Not in contract spec |
| `producer.host` | (removed) | Not in contract spec |
| `run.evidence_label` | moved to suite-level | Per-suite evidence provenance |
| `matrix` | optional | Not required by consumer |
| `comparator` | optional | Not required by consumer |

---

## JSON Schema (machine-readable)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "urn:phenotype:eval-interchange:v1.0",
  "type": "object",
  "required": ["contract_version", "artifact_kind", "producer", "run", "suites", "totals", "hash_chain"],
  "properties": {
    "contract_version": { "type": "string", "const": "1.0" },
    "artifact_kind": { "type": "string", "const": "EvaluationReport" },
    "producer": {
      "type": "object",
      "required": ["name", "version", "commit_sha"],
      "properties": {
        "name": { "type": "string" },
        "version": { "type": "string" },
        "commit_sha": { "type": "string", "pattern": "^[0-9a-f]{40}$" }
      },
      "additionalProperties": true
    },
    "run": {
      "type": "object",
      "required": ["run_id", "started_at", "model", "variant", "judge_mode"],
      "properties": {
        "run_id": { "type": "string" },
        "started_at": { "type": "string", "format": "date-time" },
        "model": { "type": "string" },
        "variant": { "type": "string", "enum": ["stock", "ours"] },
        "judge_mode": { "type": "string", "enum": ["deterministic", "llm", "hybrid"] }
      },
      "additionalProperties": true
    },
    "suites": {
      "type": "array",
      "minItems": 1,
      "items": {
        "type": "object",
        "required": ["suite", "n", "passed", "pass_at_1", "evidence_label"],
        "properties": {
          "suite": { "type": "string" },
          "n": { "type": "integer", "minimum": 0 },
          "passed": { "type": "integer", "minimum": 0 },
          "pass_at_1": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
          "evidence_label": { "type": "string", "enum": ["live_verified", "reported", "synthetic"] }
        },
        "additionalProperties": true
      }
    },
    "totals": {
      "type": "object",
      "required": ["cells", "passed", "pass_at_1"],
      "properties": {
        "cells": { "type": "integer", "minimum": 0 },
        "passed": { "type": "integer", "minimum": 0 },
        "pass_at_1": { "type": "number", "minimum": 0.0, "maximum": 1.0 }
      },
      "additionalProperties": true
    },
    "hash_chain": {
      "type": "object",
      "required": ["top_level_sha256", "task_ids_sorted_sha256"],
      "properties": {
        "top_level_sha256": { "type": "string", "pattern": "^[0-9a-f]{64}$" },
        "task_ids_sorted_sha256": { "type": "string", "pattern": "^[0-9a-f]{64}$" }
      },
      "additionalProperties": false
    },
    "matrix": { "type": "object" },
    "comparator": { "type": "object" }
  },
  "additionalProperties": true
}
```
