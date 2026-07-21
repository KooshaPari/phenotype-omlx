/* ── View & filter enums ───────────────────────────────────────────── */

export type ViewType =
  | 'overview'
  | 'suites'
  | 'suite'
  | 'task'
  | 'cells'
  | 'comparison'
  | 'failures'
  | 'calibration'
  | 'viz'
  | 'throughput'
  | 'rlvr'
  | 'audit'
  | 'langsmith';

export type FailMode = 'all' | 'timeout' | 'low-pc' | 'hallucination';

export type SortDir = 1 | -1;

export type GroupBy = 'none' | 'suite' | 'difficulty' | 'variant' | 'status';

/* ── Variant summary metrics ──────────────────────────────────────── */

export interface VariantSummary {
  pass_at_1: number;
  gen_ok: number;
  verified_pass_at_1: number;
  mean_wall_clock_s: number;
  mean_partial_credit: number;
  mean_format_compliance: number;
  n_hallucinations: number;
  mean_tokens_read: number;
  mean_cost_usd: number;
  mean_peak_rss_mb: number;
  mean_energy_joules: number;
  mean_first_token_ms: number;
  mean_retry_count: number;
  success_rate: number;
  timeout_rate: number;
  /** Catch-all for any extra metric the server sends. */
  [k: string]: number;
}

/* ── Summary data ─────────────────────────────────────────────────── */

export interface SummaryData {
  meta: { model: string; n_cells: number; n_suites: number };
  by_variant: {
    stock: VariantSummary;
    ours: VariantSummary;
  };
}

/* Keep the old name around so existing imports don't break. */
export type Summary = SummaryData;

/* ── Individual benchmark cell ────────────────────────────────────── */

export interface Cell {
  task_id: string;
  variant: 'stock' | 'ours';
  suite: string;
  difficulty: string;
  task_type: string;
  ok: boolean;
  wall_clock_s: number;
  tokens_per_second: number;
  first_token_latency_ms: number;
  pass_at_1: number;
  gen_ok?: number;
  verified_pass_at_1?: number;
  partial_credit: number;
  format_compliance_rate: number;
  judge_score: number;
  intent_preservation_rate: number;
  tool_call_success_rate: number;
  hallucination_count: number;
  retry_count: number;
  total_tokens_in: number;
  total_tokens_out: number;
  cost_usd: number;
  peak_rss_mb: number;
  peak_gpu_mem_mb: number;
  energy_proxy_joules: number;
  created_at: string;
  completed_at?: string;
  semantic?: Record<string, number>;
  failure_analysis?: {
    primary_factor?: string;
    confidence?: number;
    [k: string]: any;
  };
  reply?: string;
  prompt?: string;
  error_message?: string;
  error_code?: string;
  model_name?: string;
  model_version?: string;
  temperature?: number;
  max_tokens?: number;
  top_p?: number;
  seed?: number;
  scoring_method?: string;
  system_prompt_hash?: string;
  progress_trace?: any[];
  expected_answer?: string;
  metadata?: Record<string, string>;
  [k: string]: any;
}

/* ── Payload envelope (from WebSocket) ────────────────────────────── */

export interface LintWarning {
  code: 'degenerate_cell' | 'all_variants_pass' | 'missing_judge_score' | string;
  severity: 'error' | 'warning' | 'info';
  message: string;
  cells?: string[];
}

export interface BenchPayload {
  serverTs: string;
  lintRunTs?: string;
  warnings?: LintWarning[];
  data: { summary: SummaryData; cells: Cell[] };
}

/* ── History ring buffer entry ────────────────────────────────────── */

export interface HistoryEntry {
  receivedAt: string;
  summary: SummaryData;
  cellCount: number;
}

/* ── Diff between consecutive pushes ──────────────────────────────── */

export interface DiffInfo {
  added: number;
  removed: number;
  changed: number;
  ts: string;
}

/* ── Sparkline data point ─────────────────────────────────────────── */

export interface TrendPoint {
  stock: number;
  ours: number;
  ts: string;
}

/* ── Auto-detected insight ────────────────────────────────────────── */

export interface Insight {
  /** Insight identifier (stable for dismiss persistence). */
  kind: string;
  level: 'good' | 'warn' | 'bad';
  text: string;
  /** Optional navigation target (e.g. "cells?suite=X"). */
  jumpTo?: string;
}

/* ── Filter set shape used by the state ───────────────────────────── */

export interface CellFilters {
  suite: Set<string>;
  difficulty: Set<string>;
  variant: Set<string>;
}
