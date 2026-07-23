# LLM observability alternatives — why Langfuse wins for Phenotype

**Verdict (2026-07):** Langfuse remains the absolute best default for
bench-cockpit / Phenotype eval observability. **Operate on Cloud Hobby first**;
self-host only when caps force it or a meaningful fork appears. Keep Phoenix and
Braintrust as *side* tools only if a specific gap appears.

## Decision criteria (Phenotype)

| Criterion | Weight | Why it matters here |
|-----------|--------|---------------------|
| Self-host MIT / no SaaS meter | High (overflow) | Prefer Cloud Hobby until caps; self-host when fleet volume exceeds Hobby |
| Tracing + datasets + LLM-as-judge + playground | Critical | Parity with LangSmith evaluators we are migrating off |
| Custom LLM connections (Minimax Anthropic adapter) | Critical | Hosted judges already wired to `Minimax` / `Minimax-M3` |
| Agent surface (MCP + CLI + Agent Skill) | High | Agents must R/W traces/scores/prompts without leaving Cursor/Claude |
| Podman / Apple Container (never Docker) | High | Org hard constraint |
| Cost / lock-in | High | Prefer OSS primary; SaaS only as optional mirror |
| Harbor / portage fit | Medium | Offline judges + generation seed already land in Langfuse |

## Comparison matrix

| Tool | License / self-host | Traces | Prompts | Hosted LLM judges | Datasets / experiments | MCP/CLI/agents | Notes for us |
|------|---------------------|--------|---------|-------------------|------------------------|----------------|--------------|
| **Langfuse** | MIT core; unlimited self-host | Yes | Yes | Yes (custom providers) | Yes | Yes (first-class) | **Primary.** Same codebase as Cloud |
| Arize Phoenix | ELv2 (source-available) | Yes (OTel) | Weak | Evals strong | Partial | Weaker | Best *side* OTel/RAG debugger |
| Helicone | OSS proxy | Gateway-first | No | Weak | Weak | Limited | Fast cost proxy, not eval platform |
| Braintrust | SaaS; self-host enterprise | Yes | Partial | Eval-first | Strong CI gates | Partial | Great eval CI; paid / lock-in |
| W&B Weave | Apache (W&B stack) | Yes | Partial | Yes | Yes | W&B-centric | Only if already on W&B |
| LangSmith | SaaS | Yes | Hub | Yes | Yes | Limited vs LF | **Legacy.** Cost + less control |
| PostHog LLM | MIT | Product+LLM | Weak | Weak | Weak | Product analytics | Wrong center of gravity |
| OpenLLMetry | Apache | Instrumentation | No | No | No | OTel only | Pipe into Langfuse/Phoenix backend |

Sources: Langfuse pricing / self-host docs; 2026 comparisons (Lushbinary, PostHog,
Open-Techstack, SideGuy). Re-verify pricing before budget commits.

## Why not switch

1. **Already integrated:** cockpit BFF, seed+generation, Minimax hosted judges,
   score configs, observation rules, offline judge fallback.
2. **Cloud Hobby first, self-host as overflow:** same product features; remove
   freemium pain only when units/seats/retention actually block work.
3. **Agent path is native:** MCP (`/api/public/mcp`), `langfuse-cli`, Agent Skill
   (`npx skills add langfuse/skills`) — matches Phenotype agent-first workflow.
4. **Phoenix** wins only if we standardize purely on OTel collectors and accept
   weaker prompt/playground/judge UX.
5. **Braintrust** wins only for eval-gate CI as a *secondary* system — self-host
   is enterprise-gated; contradicts OSS/control mandate.

## Recommended topology

```text
                    ┌─────────────────────────────┐
   bench-cockpit ──►│ Langfuse PRIMARY (self-host) │  ◄── MCP / CLI / Skill
   Harbor / judges  │ unlimited retention + data   │
                    └──────────────┬──────────────┘
                                   │ optional dual-write / migrate
                                   ▼
                    ┌─────────────────────────────┐
                    │ Langfuse Cloud (US) mirror   │  Hobby for UI share / backup
                    └─────────────────────────────┘
   Side (optional): Phoenix for deep OTel/RAG debug; never replace primary.
```

## Revisit triggers

Re-open this decision only if:

- Hosted Minimax preflight remains unusable and another OSS stack has better
  structured-output judges with Anthropic-compat custom bases, **or**
- We standardize org-wide on OTel→ClickHouse and drop prompt/playground needs, **or**
- Self-host ops cost exceeds Cloud Core/Pro *and* we accept SaaS retention limits.

Until then: ship Langfuse deeper (MCP/CLI/self-host), not a parallel primary.
