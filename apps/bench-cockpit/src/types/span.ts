/** Discriminated ProgressTrace span union with legacy fallback. */

export type SpanKind =
  | 'turn'
  | 'llm'
  | 'tool'
  | 'verifier'
  | 'reward'
  | 'raw';

export interface SpanBase {
  id: string;
  kind: SpanKind;
  name?: string;
  startMs?: number;
  endMs?: number;
  parentId?: string;
}

export interface TurnSpan extends SpanBase {
  kind: 'turn';
  turn: number;
  role?: string;
  content?: string;
}

export interface LlmSpan extends SpanBase {
  kind: 'llm';
  model?: string;
  tokensIn?: number;
  tokensOut?: number;
}

export interface ToolSpan extends SpanBase {
  kind: 'tool';
  toolName?: string;
  ok?: boolean;
}

export interface VerifierSpan extends SpanBase {
  kind: 'verifier';
  passed?: boolean;
  reward?: number;
}

export interface RewardSpan extends SpanBase {
  kind: 'reward';
  /** Primary RLVR scalar: nested L0/L1/L2/L3 composite. */
  composite?: number;
  l0?: number;
  l1?: number;
  l2?: number;
  l3?: number;
  breakdown?: Record<string, number>;
  tournamentDelta?: number;
}

export interface RawSpan extends SpanBase {
  kind: 'raw';
  payload: unknown;
}

export type Span = TurnSpan | LlmSpan | ToolSpan | VerifierSpan | RewardSpan | RawSpan;

function asNum(v: unknown): number | undefined {
  return typeof v === 'number' && Number.isFinite(v) ? v : undefined;
}

function asStr(v: unknown): string | undefined {
  return typeof v === 'string' ? v : undefined;
}

/** Strict decode → fallback `raw` for legacy ProgressTrace rows. */
export function decodeSpan(row: unknown, index: number): Span {
  if (!row || typeof row !== 'object') {
    return { id: `raw-${index}`, kind: 'raw', payload: row };
  }
  const o = row as Record<string, unknown>;
  const id = asStr(o.id) ?? `span-${index}`;
  const kind = asStr(o.kind) ?? asStr(o.type) ?? asStr(o.event);

  if (kind === 'turn' || o.turn != null) {
    return {
      id,
      kind: 'turn',
      turn: asNum(o.turn) ?? index,
      role: asStr(o.role),
      content: asStr(o.content) ?? asStr(o.text),
      startMs: asNum(o.start_ms) ?? asNum(o.startMs),
      endMs: asNum(o.end_ms) ?? asNum(o.endMs),
      parentId: asStr(o.parent_id) ?? asStr(o.parentId),
    };
  }
  if (kind === 'llm' || o.model != null) {
    return {
      id,
      kind: 'llm',
      model: asStr(o.model),
      tokensIn: asNum(o.tokens_in) ?? asNum(o.tokensIn),
      tokensOut: asNum(o.tokens_out) ?? asNum(o.tokensOut),
      name: asStr(o.name),
    };
  }
  if (kind === 'tool' || o.tool_name != null || o.toolName != null) {
    return {
      id,
      kind: 'tool',
      toolName: asStr(o.tool_name) ?? asStr(o.toolName),
      ok: typeof o.ok === 'boolean' ? o.ok : undefined,
      name: asStr(o.name),
    };
  }
  if (kind === 'verifier' || o.reward != null || o.passed != null) {
    return {
      id,
      kind: 'verifier',
      passed: typeof o.passed === 'boolean' ? o.passed : undefined,
      reward: asNum(o.reward),
      name: asStr(o.name),
    };
  }
  if (kind === 'reward' || o.composite != null || o.l0 != null) {
    return {
      id,
      kind: 'reward',
      composite: asNum(o.composite),
      l0: asNum(o.l0),
      l1: asNum(o.l1),
      l2: asNum(o.l2),
      l3: asNum(o.l3),
      breakdown:
        o.breakdown && typeof o.breakdown === 'object'
          ? (o.breakdown as Record<string, number>)
          : undefined,
      tournamentDelta: asNum(o.tournament_delta) ?? asNum(o.tournamentDelta),
    };
  }
  return { id, kind: 'raw', payload: row, name: asStr(o.name) };
}

export function decodeTrace(rows: unknown[] | undefined): Span[] {
  if (!Array.isArray(rows)) return [];
  return rows.map(decodeSpan);
}
