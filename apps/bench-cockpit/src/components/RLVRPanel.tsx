import React, { useMemo } from 'react';
import { Cell } from '../types';
import { resolveRlvr } from '../lib/rlvr';

interface Props {
  cells: Cell[];
}

/**
 * RLVR-AF panel.
 * Primary scalar: nested L0/L1/L2/L3 composite.
 * Drill-down: 8-component breakdown.
 * Secondary: tournament delta.
 */
export default function RLVRPanel({ cells }: Props) {
  const rows = useMemo(() => {
    return cells.map((c) => {
      const r = resolveRlvr(c);
      return { cell: c, ...r };
    });
  }, [cells]);

  const verifiableShare = cells.length
    ? rows.filter((r) => r.verifiable).length / Math.max(1, cells.length)
    : 0;
  const sourceCounts = useMemo(() => {
    const m = { harness: 0, trace: 0, derived: 0 };
    for (const r of rows) m[r.source]++;
    return m;
  }, [rows]);

  const components = [
    'json',
    'tool',
    'patch',
    'tests',
    'output_cap',
    'context_budget',
    'escalation',
    'tokens_saved',
  ];

  if (!cells.length) {
    return (
      <div className="empty-state" data-testid="rlvr-empty">
        No cells loaded.
      </div>
    );
  }

  return (
    <div className="view-stack" data-testid="rlvr-view">
      <div className="viz-panel">
        <div className="viz-toolbar">
          <span className="viz-title">RLVR-AF · primary = L0–L3 composite</span>
          <span className="viz-hint">
            verifiable {(verifiableShare * 100).toFixed(0)}% · sources harness {sourceCounts.harness} /
            trace {sourceCounts.trace} / derived {sourceCounts.derived}
          </span>
        </div>
        {sourceCounts.derived === rows.length && (
          <div className="warn-banner">
            No harness <code>rlvr_*</code> fields — showing <b>derived</b> L0–L3 from quality/perf
            metrics. Wire progress_trace reward spans or top-level rlvr_* for true RLVR-AF.
          </div>
        )}
        <table className="heat-table">
          <thead>
            <tr>
              <th>task</th>
              <th>variant</th>
              <th>src</th>
              <th>composite</th>
              <th>L0</th>
              <th>L1</th>
              <th>L2</th>
              <th>L3</th>
              <th>tourn Δ</th>
              <th>pass</th>
            </tr>
          </thead>
          <tbody>
            {rows.slice(0, 100).map((r, i) => (
              <tr key={i}>
                <td>{r.cell.task_id}</td>
                <td>{r.cell.variant}</td>
                <td><span className="badge">{r.source}</span></td>
                <td>{r.composite.toFixed(3)}</td>
                <td>{r.l0.toFixed(2)}</td>
                <td>{r.l1.toFixed(2)}</td>
                <td>{r.l2.toFixed(2)}</td>
                <td>{r.l3.toFixed(2)}</td>
                <td>{r.tournamentDelta.toFixed(3)}</td>
                <td className={r.passed ? 'good' : 'bad'}>{r.passed ? '✓' : '✗'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="viz-panel">
        <div className="viz-toolbar">
          <span className="viz-title">8-component breakdown (drill-down)</span>
        </div>
        <div className="rlvr-bars">
          {components.map((k) => {
            const vals = rows.map((r) => Number(r.breakdown[k] ?? 0));
            const mean = vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 0;
            return (
              <div className="rlvr-bar-row" key={k}>
                <span className="l">{k}</span>
                <div className="bar-wrap">
                  <div className="bar mid" style={{ width: `${Math.min(100, mean * 100)}%` }} />
                </div>
                <span className="v">{mean.toFixed(3)}</span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
