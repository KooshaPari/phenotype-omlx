import { Cell } from '../types';

function firstString(...vals: unknown[]): string | null {
  for (const v of vals) {
    if (typeof v === 'string' && v.trim()) return v.trim();
  }
  return null;
}

function meta(cell: Cell, ...keys: string[]): string | null {
  const m = cell.metadata;
  if (!m) return null;
  for (const k of keys) {
    const v = m[k];
    if (typeof v === 'string' && v.trim()) return v.trim();
  }
  return null;
}

/** Display title: enriched export → metadata → task_id. */
export function taskTitle(cell: Cell | null, fallbackId: string): string {
  if (!cell) return fallbackId;
  return (
    firstString(cell.task_title, cell.title, meta(cell, 'task_title', 'title')) ?? fallbackId
  );
}

/** Assignment description when present (top-level or metadata). */
export function taskDescription(cell: Cell | null): string | null {
  if (!cell) return null;
  return firstString(
    cell.description,
    cell.task_description,
    meta(cell, 'description', 'task_description'),
  );
}

/** Acceptance criteria / rubric text (Canvas-style grading section). */
export function taskAcceptance(cell: Cell | null): string | null {
  if (!cell) return null;
  return firstString(
    cell.acceptance,
    cell.acceptance_criteria,
    cell.rubric,
    meta(cell, 'acceptance', 'acceptance_criteria', 'rubric'),
  );
}

/** Prefer progress_trace; fall back to chat_trace (dual-read). */
export function resolveTraceRows(cell: Cell): unknown[] | undefined {
  if (Array.isArray(cell.progress_trace) && cell.progress_trace.length > 0) {
    return cell.progress_trace;
  }
  if (Array.isArray(cell.chat_trace) && cell.chat_trace.length > 0) {
    return cell.chat_trace;
  }
  if (Array.isArray(cell.progress_trace)) return cell.progress_trace;
  if (Array.isArray(cell.chat_trace)) return cell.chat_trace;
  return undefined;
}

/** True when V5 / export stripped IO or transcript bodies. */
export function isTraceTruncated(cell: Cell, spansLen: number): boolean {
  const flag = meta(cell, 'trace_truncated', 'truncated', 'io_truncated');
  if (flag && /^(1|true|yes|truncated)$/i.test(flag)) return true;
  if (cell.metadata?.evidence_label === 'truncated') return true;
  // Spans present but no conversational content and no IO — typical V5 strip.
  const hasIO = Boolean(cell.prompt?.trim() || cell.reply?.trim());
  if (spansLen === 0 && !hasIO) return true;
  return false;
}

export type OutcomeKey =
  | 'pass_at_1'
  | 'gen_ok'
  | 'verified_pass_at_1'
  | 'partial_credit'
  | 'judge';

export const OUTCOME_EXPLAIN: Record<
  OutcomeKey,
  { label: string; blurb: string; fmt: (v: number | null) => string }
> = {
  pass_at_1: {
    label: 'pass_at_1',
    blurb:
      'Legacy primary pass rate (0–1). On many V5 runs this equals generation-ok / reported success, not live verified grading.',
    fmt: (v) => (v == null ? '—' : `${(v * 100).toFixed(1)}%`),
  },
  gen_ok: {
    label: 'gen_ok',
    blurb:
      'Whether the model produced a scorable generation (format / substring / harness OK). Prefer this over pass_at_1 when evidence is reported/synthetic.',
    fmt: (v) => (v == null ? '—' : `${(v * 100).toFixed(1)}%`),
  },
  verified_pass_at_1: {
    label: 'verified_pass_at_1',
    blurb:
      'Live-verified pass when the harness graded against expected answers or an external judge with evidence_label=live_verified. Absent on truncated V5 exports.',
    fmt: (v) => (v == null ? '—' : `${(v * 100).toFixed(1)}%`),
  },
  partial_credit: {
    label: 'partial_credit',
    blurb:
      'Soft quality in [0,1] from rubric / semantic / reward layers. More informative than binary pass when tasks are partially correct.',
    fmt: (v) => (v == null ? '—' : v.toFixed(3)),
  },
  judge: {
    label: 'judge',
    blurb:
      'LLM-as-judge or harness judge_score in [0,1]. Often zeroed or missing on V5 contract dumps — treat as untrusted until a judge arm is wired.',
    fmt: (v) => (v == null ? '—' : v.toFixed(3)),
  },
};
