# 15m loop — audit or do nxt (subagent fan-out)

**Interval:** 15m
**Sentinel:** `AGENT_LOOP_TICK_audit_nxt`
**Policy:** Each tick, spawn **multiple Task subagents in parallel** (hub, Mac V5, pairings/Langfuse, shell artifacts). Prefer doing concrete work over status-only.

**Runtime:** Podman only — never Docker Engine.

**Open (needs Mac online):**
- Pull real V5 EvaluationReport → `apps/bench-cockpit/data` + `BENCH_DATA`
