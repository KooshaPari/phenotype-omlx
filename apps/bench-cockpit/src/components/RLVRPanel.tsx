import React, { useMemo, useState } from 'react';
import { Cell } from '../types';
import { formatRlvrScore, resolveRlvr } from '../lib/rlvr';

interface Props {
  cells: Cell[];
}

/**
 * RLVR-AF panel.
 * Primary scalar: nested L0/L1/L2/L3 composite.
 * Drill-down: 8-component breakdown.
 * Secondary: tournament delta.
 *
 * Missing harness rlvr_* → source unavailable (not fake 100%/zeros).
 * Derived synthesis is opt-in via the debug toggle (non-authoritative).
 */
export default function RLVRPanel({ cells }: Props) {
  const [allowDerived, setAllowDerived] = useState(false);

  const rows = useMemo(() => {
    return cells.map((c) => {
      const r = resolveRlvr(c, { allowDerived });
      return { cell: c, ...r };
    });
  }, [cells, allowDerived]);

  const verifiableShare = cells.length
    ? rows.filter((r) => r.verifiable).length / Math.max(1, cells.length)
    : 0;
  const sourceCounts = useMemo(() => {
    const m = { harness: 0, trace: 0, derived: 0, unavailable: 0 };
    for (const r of rows) m[r.source]++;
    return m;
  }, [rows]);

  const nonAuthoritative = sourceCounts.derived + sourceCounts.unavailable;
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
            trace {sourceCounts.trace} / derived {sourceCounts.derived} / unavailable{' '}
            {sourceCounts.unavailable}
          </span>
          <label className="viz-hint" style={{ display: 'inline-flex', gap: '0.35rem', alignItems: 'center' }}>
            <input
              type="checkbox"
              data-testid="rlvr-allow-derived"
              checked={allowDerived}
              onChange={(e) => setAllowDerived(e.target.checked)}
            />
            show derived (non-authoritative)
          </label>
        </div>
        {sourceCounts.unavailable > 0 && (
          <div className="warn-banner" data-testid="rlvr-unavailable-banner">
            {sourceCounts.unavailable} cell{sourceCounts.unavailable === 1 ? '' : 's'} missing harness{' '}
            <code>rlvr_*</code> / trace reward — scores shown as <b>—</b> (not proven quality). Wire
            top-level rlvr_* or progress_trace reward spans. Derived synthesis is opt-in only.
          </div>
        )}
        {sourceCounts.derived > 0 && (
          <div className="warn-banner" data-testid="rlvr-derived-banner">
            Showing <b>derived</b> L0–L3 from quality/perf stubs (intent / hallu / judge / …) —{' '}
            <b>non-authoritative</b>, not harness RLVR-AF.
          </div>
        )}
        {nonAuthoritative === 0 && sourceCounts.harness + sourceCounts.trace === rows.length && (
          <div className="good-banner" data-testid="rlvr-authoritative-banner">
            All cells resolved from harness or trace RLVR.
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
              <tr key={i} data-rlvr-source={r.source}>
                <td>{r.cell.task_id}</td>
                <td>{r.cell.variant}</td>
                <td>
                  <span className={`badge${r.authoritative ? '' : ' badge-warn'}`}>{r.source}</span>
                </td>
                <td>{formatRlvrScore(r.composite)}</td>
                <td>{formatRlvrScore(r.l0, 2)}</td>
                <td>{formatRlvrScore(r.l1, 2)}</td>
                <td>{formatRlvrScore(r.l2, 2)}</td>
                <td>{formatRlvrScore(r.l3, 2)}</td>
                <td>{formatRlvrScore(r.tournamentDelta)}</td>
                <td className={r.source === 'unavailable' ? 'faint' : r.passed ? 'good' : 'bad'}>
                  {r.source === 'unavailable' ? '—' : r.passed ? '✓' : '✗'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="viz-panel">
        <div className="viz-toolbar">
          <span className="viz-title">8-component breakdown (drill-down)</span>
          {sourceCounts.unavailable === rows.length && !allowDerived && (
            <span className="viz-hint">unavailable — no breakdown until harness rlvr_* present</span>
          )}
        </div>
        <div className="rlvr-bars">
          {components.map((k) => {
            const vals = rows
              .map((r) => Number(r.breakdown[k] ?? Number.NaN))
              .filter((v) => Number.isFinite(v));
            const mean = vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : Number.NaN;
            return (
              <div className="rlvr-bar-row" key={k}>
                <span className="l">{k}</span>
                <div className="bar-wrap">
                  <div
                    className="bar mid"
                    style={{
                      width: Number.isFinite(mean) ? `${Math.min(100, mean * 100)}%` : '0%',
                    }}
                  />
                </div>
                <span className="v">{formatRlvrScore(mean)}</span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
