import { Cell, VariantSummary } from '../types';

/** Evidence labels that indicate live verified grading (not reported/synthetic). */
const VERIFIED_EVIDENCE = new Set(['live_verified', 'verified']);

export function isVerifiedEvidence(cell: Cell): boolean {
  const label = cell.metadata?.evidence_label ?? '';
  return VERIFIED_EVIDENCE.has(label);
}

/** Generation-ok score: prefer explicit gen_ok, fall back to legacy pass_at_1. */
export function effectiveGenOk(cell: Cell): number {
  if (cell.gen_ok != null && !Number.isNaN(cell.gen_ok)) {
    return cell.gen_ok;
  }
  return cell.pass_at_1;
}

/** Verified pass when present and meaningful, else null. */
export function effectiveVerifiedPass(cell: Cell): number | null {
  const v = cell.verified_pass_at_1;
  if (v == null || Number.isNaN(v)) return null;
  if (v > 0 || isVerifiedEvidence(cell)) return v;
  return null;
}

/** Whether the cell has a trusted verified-pass signal for quality aggregates. */
export function hasVerifiedPass(cell: Cell): boolean {
  return effectiveVerifiedPass(cell) != null;
}

/** Primary quality pass: verified when available, otherwise gen_ok / pass_at_1. */
export function qualityPass(cell: Cell): number {
  const verified = effectiveVerifiedPass(cell);
  if (verified != null) return verified;
  return effectiveGenOk(cell);
}

export function qualityPassLabel(cell: Cell, untrusted = false): string {
  if (hasVerifiedPass(cell)) return 'Verified';
  return untrusted ? 'Gen ok' : 'Pass@1';
}

function summaryGenOk(v: VariantSummary): number {
  return v.gen_ok ?? v.pass_at_1;
}

function summaryVerified(v: VariantSummary): number | null {
  const raw = v.verified_pass_at_1;
  if (raw == null || Number.isNaN(raw) || raw <= 0) return null;
  return raw;
}

export function summaryQualityPass(v: VariantSummary): number {
  const verified = summaryVerified(v);
  if (verified != null) return verified;
  return summaryGenOk(v);
}

export function summaryHasVerified(v: VariantSummary): boolean {
  return summaryVerified(v) != null;
}

export function summaryQualityLabel(v: VariantSummary, untrusted = false): string {
  if (summaryHasVerified(v)) return 'Verified';
  return untrusted ? 'Gen ok' : 'Pass@1';
}

export function meanQualityPass(cells: Cell[]): number {
  if (!cells.length) return 0;
  return cells.reduce((s, c) => s + qualityPass(c), 0) / cells.length;
}

export function meanGenOk(cells: Cell[]): number {
  if (!cells.length) return 0;
  return cells.reduce((s, c) => s + effectiveGenOk(c), 0) / cells.length;
}
