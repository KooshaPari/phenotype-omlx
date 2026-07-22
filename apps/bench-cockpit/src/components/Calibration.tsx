import React, { useMemo, useState } from 'react';
import { Cell, LintWarning, BenchPayload } from '../types';

interface Props {
  cells: Cell[];
  warnings?: LintWarning[];
  lintRunTs?: string;
}

interface CalibratedRow {
  suite: string;
  variant: 'stock' | 'ours';
  n: number;
  rawPass: number;
  expectedRandom: number;
  calibratedPass: number;
}

interface PerItemHistRow {
  task_id: string;
  suite: string;
  variants: Array<{ variant: string; pass_at_1: number }>;
  flag: 'degenerate' | 'all_variants_pass' | null;
}

function mean(a: number[]): number {
  return a.length ? a.reduce((s, x) => s + x, 0) / a.length : 0;
}

// Estimate expected-random baseline for a suite from its task_type mix.
// Conservative defaults when task_type is missing. These are kept
// deliberately low so the calibration surfaces over-claiming rather
// than hiding it; tune via SPEC-005 eval-pillars upstream.
const RANDOM_BASELINE_BY_TASK_TYPE: Record<string, number> = {
  arithmetic: 0.0,
  exact_match: 0.0,
  multi_choice: 0.25,
  multiple_choice: 0.25,
  code_exec: 0.0,
  code_generate: 0.0,
  reading_comprehension: 0.2,
  open_qa: 0.05,
  instruction_following: 0.1,
};

function baselineForCell(c: Cell): number {
  const tt = (c.task_type || '').toLowerCase();
  if (tt in RANDOM_BASELINE_BY_TASK_TYPE) {
    return RANDOM_BASELINE_BY_TASK_TYPE[tt];
  }
  // 3+ choice MC fallback
  return 0.25;
}

function computeCalibrated(cells: Cell[]): CalibratedRow[] {
  const byKey = new Map<string, Cell[]>();
  for (const c of cells) {
    const k = `${c.suite}::${c.variant}`;
    if (!byKey.has(k)) byKey.set(k, []);
    byKey.get(k)!.push(c);
  }
  const rows: CalibratedRow[] = [];
  for (const [k, group] of byKey) {
    const [suite, variant] = k.split('::') as [string, 'stock' | 'ours'];
    const n = group.length;
    const rawPass = mean(group.map(c => c.pass_at_1));
    const expectedRandom = mean(group.map(baselineForCell));
    const denom = Math.max(1e-6, 1 - expectedRandom);
    const calibratedPass = Math.max(0, Math.min(1, (rawPass - expectedRandom) / denom));
    rows.push({ suite, variant, n, rawPass, expectedRandom, calibratedPass });
  }
  return rows.sort((a, b) => a.suite.localeCompare(b.suite) || a.variant.localeCompare(b.variant));
}

function computePerItemHist(cells: Cell[]): PerItemHistRow[] {
  const byKey = new Map<string, Cell[]>();
  for (const c of cells) {
    if (!byKey.has(c.task_id)) byKey.set(c.task_id, []);
    byKey.get(c.task_id)!.push(c);
  }
  const rows: PerItemHistRow[] = [];
  for (const [task_id, group] of byKey) {
    const variants = group.map(c => ({ variant: c.variant, pass_at_1: c.pass_at_1 }));
    let flag: PerItemHistRow['flag'] = null;
    if (variants.length > 1 && variants.every(v => v.pass_at_1 >= 0.999)) {
      flag = 'all_variants_pass';
    } else if (group.some(c => c.pass_at_1 >= 0.999 && (c.wall_clock_s < 0.05 || !c.prompt))) {
      flag = 'degenerate';
    }
    rows.push({ task_id, suite: group[0]?.suite || '?', variants, flag });
  }
  return rows.sort((a, b) => a.suite.localeCompare(b.suite) || a.task_id.localeCompare(b.task_id));
}

const SEV_RANK: Record<string, number> = { error: 0, warning: 1, info: 2 };

export default function Calibration({ cells, warnings, lintRunTs }: Props) {
  const [showRaw, setShowRaw] = useState(true);
  const calibrated = useMemo(() => computeCalibrated(cells), [cells]);
  const perItem = useMemo(() => computePerItemHist(cells), [cells]);
  const sortedWarnings = useMemo(() => {
    return [...(warnings || [])].sort((a, b) =>
      (SEV_RANK[a.severity] ?? 9) - (SEV_RANK[b.severity] ?? 9)
    );
  }, [warnings]);

  const flaggedDegenerate = perItem.filter(r => r.flag === 'degenerate');
  const flaggedAllPass = perItem.filter(r => r.flag === 'all_variants_pass');

  // Per-suite per-variant pair for the bar chart
  const bySuite = useMemo(() => {
    const m = new Map<string, { stock?: CalibratedRow; ours?: CalibratedRow }>();
    for (const r of calibrated) {
      if (!m.has(r.suite)) m.set(r.suite, {});
      m.get(r.suite)![r.variant] = r;
    }
    return [...m.entries()].sort((a, b) => a[0].localeCompare(b[0]));
  }, [calibrated]);

  return (
    <div className="view-content">
      {/* Lint warnings banner */}
      {sortedWarnings.length > 0 && (
        <div className="cal-warnings" data-testid="cal-warnings">
          <div className="cal-warnings-header">
            <span className="cal-warnings-title">
              {sortedWarnings.length} lint {sortedWarnings.length === 1 ? 'warning' : 'warnings'}
            </span>
            {lintRunTs && <span className="cal-warnings-ts">last run {new Date(lintRunTs).toLocaleTimeString()}</span>}
          </div>
          {sortedWarnings.map((w, i) => (
            <div key={i} className={`cal-warn cal-warn-${w.severity}`}>
              <span className="cal-warn-sev">{w.severity}</span>
              <span className="cal-warn-code">{w.code}</span>
              <span className="cal-warn-msg">{w.message}</span>
              {w.cells && w.cells.length > 0 && (
                <details className="cal-warn-cells">
                  <summary>{w.cells.length} cell(s)</summary>
                  <ul>
                    {w.cells.slice(0, 50).map((c, j) => <li key={j}>{c}</li>)}
                    {w.cells.length > 50 && <li>... and {w.cells.length - 50} more</li>}
                  </ul>
                </details>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Toggle raw vs calibrated */}
      <div className="cal-toolbar">
        <span className="cal-toolbar-label">Display:</span>
        <button
          className={`cal-toggle ${showRaw ? 'on' : ''}`}
          onClick={() => setShowRaw(true)}
        >raw pass@1</button>
        <button
          className={`cal-toggle ${!showRaw ? 'on' : ''}`}
          onClick={() => setShowRaw(false)}
        >calibrated (random-baseline subtracted)</button>
      </div>

      {/* Per-suite calibration bars */}
      <h3 className="section-title">Calibration per Suite</h3>
      <div className="cal-suite-grid">
        {bySuite.map(([suite, grp]) => {
          const sRow = grp.stock;
          const oRow = grp.ours;
          if (!sRow || !oRow) return null;
          const sVal = showRaw ? sRow.rawPass : sRow.calibratedPass;
          const oVal = showRaw ? oRow.rawPass : oRow.calibratedPass;
          const delta = oVal - sVal;
          const cls = delta > 0.001 ? 'positive' : delta < -0.001 ? 'negative' : 'neutral';
          return (
            <div key={suite} className="cal-suite-card">
              <div className="cal-suite-name">{suite}</div>
              <div className="cal-suite-row">
                <span className="swatch stock" />stock
                <div className="cal-bar-wrap">
                  <div className="cal-bar stock" style={{ width: `${sVal * 100}%` }} />
                </div>
                <span className="cal-val">{(sVal * 100).toFixed(1)}%</span>
              </div>
              <div className="cal-suite-row">
                <span className="swatch ours" />ours
                <div className="cal-bar-wrap">
                  <div className="cal-bar ours" style={{ width: `${oVal * 100}%` }} />
                </div>
                <span className="cal-val">{(oVal * 100).toFixed(1)}%</span>
              </div>
              <div className="cal-suite-meta">
                <span>n={sRow.n}+{oRow.n}</span>
                <span className="cal-suite-random">random ~ {(sRow.expectedRandom * 100).toFixed(0)}%</span>
                <span className={`cal-delta ${cls}`}>Δ {delta >= 0 ? '+' : ''}{(delta * 100).toFixed(1)}pp</span>
              </div>
            </div>
          );
        })}
        {bySuite.length === 0 && (
          <div className="cal-empty">No cells available for calibration.</div>
        )}
      </div>

      {/* Degenerate / all-pass flagged items */}
      <h3 className="section-title">Flagged Items</h3>
      <div className="cal-flag-summary">
        <span className="cal-flag-chip cal-flag-error">
          degenerate: <strong>{flaggedDegenerate.length}</strong>
        </span>
        <span className="cal-flag-chip cal-flag-warning">
          all-variants-pass: <strong>{flaggedAllPass.length}</strong>
        </span>
      </div>
      {(flaggedDegenerate.length + flaggedAllPass.length) > 0 && (
        <div className="cal-flag-list">
          {[...flaggedDegenerate, ...flaggedAllPass].slice(0, 100).map((row, i) => (
            <div key={`${row.task_id}-${i}`} className={`cal-flag-row cal-flag-${row.flag}`}>
              <span className="cal-flag-task">{row.task_id}</span>
              <span className="cal-flag-suite">{row.suite}</span>
              <span className="cal-flag-variants">
                {row.variants.map(v => `${v.variant}:${(v.pass_at_1 * 100).toFixed(0)}%`).join(' / ')}
              </span>
              <span className="cal-flag-reason">
                {row.flag === 'degenerate' ? 'sub-50ms wall-clock or empty prompt' : 'all variants 100%'}
              </span>
            </div>
          ))}
          {(flaggedDegenerate.length + flaggedAllPass.length) > 100 && (
            <div className="cal-flag-more">
              ... and {(flaggedDegenerate.length + flaggedAllPass.length) - 100} more flagged items
            </div>
          )}
        </div>
      )}
    </div>
  );
}
