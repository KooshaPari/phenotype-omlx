# FR-7 vPU dashboard contract (D0)

**Owner chat:** Salmon (Chat 7)  
**Status:** ACTIVE — implemented in this tree  
**Evidence class:** live verified via `scripts/fr7_dashboard_smoke.py`

## Routes (canonical oMLX)

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/vpu/` | Panel HTML |
| GET | `/vpu/health` | Liveness/readiness |
| GET | `/health` | Alias of `/vpu/health` |
| GET | `/vpu/api/v1/status` | Versioned status JSON |

Default bind: `127.0.0.1:8787` via `omlx-research vpu-dashboard`.

## Status JSON schema

Committed: `perf-core/vpu/dashboard/schema/status.v1.json`  
Example payload produced by `/vpu/api/v1/status`.

Required fields: `schema_version`, `build_head`, `polyglot_tiers`, `eval_snapshot_id`, `promotion_snapshot_id`, `errors`, `owner`.

## Health contract

- `200` + `{"ok": true, ...}` when process is serving
- Fail-closed: missing dashboard assets ⇒ `503` + `{"ok": false, "error": "..."}`

## Explicit non-substitutes

pheno-harness Go dashboard, static preview HTML, and research_panel reuse do **not** satisfy FR-7.
