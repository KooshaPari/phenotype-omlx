/** Resolve RLVR-AF scalars for a bench cell. */

export type RlvrSource = 'harness' | 'trace' | 'derived' | 'unavailable';

export interface RlvrResolved {
  source: RlvrSource;
  /** Authoritative only for harness/trace. Derived/unavailable are non-authoritative. */
  authoritative: boolean;
  composite: number;
  l0: number;
  l1: number;
  l2: number;
  l3: number;
  tournamentDelta: number;
  verifiable: boolean;
  passed: boolean;
  breakdown: Record<string, number>;
}

export interface ResolveRlvrOptions {
  /**
   * Opt-in: synthesize L0–L3 from quality/perf stubs (intent/hallu/judge/…).
   * Off by default — missing harness rlvr_* must not look like proven quality.
   */
  allowDerived?: boolean;
}

function num(v: unknown, fallback = 0): number {
  const n = typeof v === 'number' ? v : Number(v);
  return Number.isFinite(n) ? n : fallback;
}

function clamp01(v: number): number {
  return Math.max(0, Math.min(1, v));
}

function fromHarness(c: Record<string, unknown>): RlvrResolved | null {
  const composite =
    c.rlvr_composite ?? c.RLVRReward ?? c.rlvr_reward;
  if (composite == null && c.rlvr_l0 == null && c.RLVRRewardBreakdown == null) {
    return null;
  }
  const breakdown = (c.rlvr_reward_breakdown ??
    c.RLVRRewardBreakdown ??
    {}) as Record<string, number>;
  const l0 = num(c.rlvr_l0 ?? breakdown.l0);
  const l1 = num(c.rlvr_l1 ?? breakdown.l1);
  const l2 = num(c.rlvr_l2 ?? breakdown.l2);
  const l3 = num(c.rlvr_l3 ?? breakdown.l3);
  const comp =
    composite != null
      ? num(composite)
      : (l0 + l1 + l2 + l3) / 4;
  return {
    source: 'harness',
    authoritative: true,
    composite: comp,
    l0,
    l1,
    l2,
    l3,
    tournamentDelta: num(c.rlvr_tournament_delta ?? c.RLVRTournamentDelta),
    verifiable: Boolean(c.rlvr_verifiable ?? c.RLVRVerifiable),
    passed: Boolean(c.rlvr_passed ?? c.RLVRPassed ?? (comp >= 0.5)),
    breakdown,
  };
}

function fromTrace(c: Record<string, unknown>): RlvrResolved | null {
  const trace = c.progress_trace;
  if (!Array.isArray(trace)) return null;
  for (let i = trace.length - 1; i >= 0; i--) {
    const row = trace[i];
    if (!row || typeof row !== 'object') continue;
    const o = row as Record<string, unknown>;
    const kind = o.kind ?? o.type ?? o.event;
    if (kind === 'reward' || o.composite != null || o.l0 != null) {
      const l0 = num(o.l0);
      const l1 = num(o.l1);
      const l2 = num(o.l2);
      const l3 = num(o.l3);
      const composite =
        o.composite != null ? num(o.composite) : (l0 + l1 + l2 + l3) / 4;
      const breakdown =
        o.breakdown && typeof o.breakdown === 'object'
          ? (o.breakdown as Record<string, number>)
          : {};
      return {
        source: 'trace',
        authoritative: true,
        composite,
        l0,
        l1,
        l2,
        l3,
        tournamentDelta: num(o.tournament_delta ?? o.tournamentDelta),
        verifiable: true,
        passed: composite >= 0.5,
        breakdown,
      };
    }
  }
  return null;
}

/**
 * Provisional L0–L3 from quality/perf metrics when harness has no RLVR.
 * Non-authoritative — for debugging only. Prefer resolveRlvr without allowDerived.
 */
export function deriveRlvr(c: Record<string, unknown>): RlvrResolved {
  const format = clamp01(num(c.format_compliance_rate));
  const pc = clamp01(num(c.partial_credit));
  const judge = num(c.judge_score);
  const pass = clamp01(num(c.pass_at_1));
  const intent = clamp01(num(c.intent_preservation_rate));
  const tool = clamp01(num(c.tool_call_success_rate));
  const hallu = num(c.hallucination_count);
  const wall = Math.max(0, num(c.wall_clock_s));
  const tps = Math.max(0, num(c.tokens_per_second));

  const l0 = format; // schema / format
  const l1 = judge > 0 ? clamp01(judge) : Math.max(pc, pass * 0.5); // correctness
  const l2 = clamp01(0.6 * intent + 0.4 * tool) * (hallu > 0 ? 0.85 : 1); // intent/tools
  const effTps = tps > 0 ? clamp01(tps / 80) : 0;
  const effWall = wall > 0 ? clamp01(1 - wall / 60) : 0;
  const l3 = clamp01(0.55 * effTps + 0.45 * effWall); // efficiency

  const composite = 0.15 * l0 + 0.4 * l1 + 0.25 * l2 + 0.2 * l3;

  return {
    source: 'derived',
    authoritative: false,
    composite,
    l0,
    l1,
    l2,
    l3,
    tournamentDelta: 0,
    verifiable: false,
    passed: pass >= 0.999 || pc >= 0.5,
    breakdown: {
      json: format,
      tool,
      patch: pc,
      tests: pass,
      output_cap: clamp01(1 - Math.min(1, num(c.verbosity_tokens, 0) / 2000)),
      context_budget: intent,
      escalation: hallu > 0 ? 0.2 : 0.8,
      tokens_saved: l3,
    },
  };
}

/** Fail-loud placeholder: no harness/trace RLVR — do not invent scores. */
export function unavailableRlvr(): RlvrResolved {
  return {
    source: 'unavailable',
    authoritative: false,
    composite: Number.NaN,
    l0: Number.NaN,
    l1: Number.NaN,
    l2: Number.NaN,
    l3: Number.NaN,
    tournamentDelta: Number.NaN,
    verifiable: false,
    passed: false,
    breakdown: {},
  };
}

export function formatRlvrScore(v: number, digits = 3): string {
  return Number.isFinite(v) ? v.toFixed(digits) : '—';
}

/**
 * Resolve RLVR for a cell.
 * Prefer harness rlvr_*, then progress_trace reward spans.
 * Default: unavailable (NaN scores) when both missing — never silently treat
 * stub intent/hallu/judge as proven RLVR. Pass `{ allowDerived: true }` to
 * opt into non-authoritative synthesis for debugging.
 */
export function resolveRlvr(
  cell: unknown,
  opts: ResolveRlvrOptions = {},
): RlvrResolved {
  const c = (cell && typeof cell === 'object' ? cell : {}) as Record<string, unknown>;
  const harness = fromHarness(c);
  if (harness) return harness;
  const trace = fromTrace(c);
  if (trace) return trace;
  if (opts.allowDerived) return deriveRlvr(c);
  return unavailableRlvr();
}
