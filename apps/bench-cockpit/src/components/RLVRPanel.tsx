import React, { useMemo } from 'react';
import { Cell } from '../types';

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
    return cells
      .filter((c) => c.rlvr_reward != null || c.rlvr_composite != null || c.RLVRReward != null)
      .map((c) => {
        const composite =
          c.rlvr_composite ??
          c.RLVRReward ??
          c.rlvr_reward ??
          0;
        const breakdown =
          c.rlvr_reward_breakdown ??
          c.RLVRRewardBreakdown ??
          {};
        return {
          cell: c,
          composite: Number(composite) || 0,
          l0: Number(c.rlvr_l0 ?? breakdown.l0 ?? 0),
          l1: Number(c.rlvr_l1 ?? breakdown.l1 ?? 0),
          l2: Number(c.rlvr_l2 ?? breakdown.l2 ?? 0),
          l3: Number(c.rlvr_l3 ?? breakdown.l3 ?? 0),
          tournamentDelta: Number(c.rlvr_tournament_delta ?? c.RLVRTournamentDelta ?? 0),
          verifiable: Boolean(c.rlvr_verifiable ?? c.RLVRVerifiable),
          passed: Boolean(c.rlvr_passed ?? c.RLVRPassed),
          breakdown: breakdown as Record<string, number>,
        };
      });
  }, [cells]);

  const verifiableShare = cells.length
    ? rows.filter((r) => r.verifiable).length / Math.max(1, cells.length)
    : 0;

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

  if (!rows.length) {
    return (
      <div className="empty-state" data-testid="rlvr-empty">
        No RLVR fields on cells yet. Wire harness JSON with rlvr_composite / rlvr_* keys.
        Primary scalar = L0/L1/L2/L3 composite.
      </div>
    );
  }

  return (
    <div className="view-stack" data-testid="rlvr-view">
      <div className="viz-panel">
        <div className="viz-toolbar">
          <span className="viz-title">RLVR-AF · primary = L0–L3 composite</span>
          <span className="viz-hint">
            verifiable share {(verifiableShare * 100).toFixed(0)}% · secondary = tournament Δ
          </span>
        </div>
        <table className="heat-table">
          <thead>
            <tr>
              <th>task</th>
              <th>variant</th>
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
