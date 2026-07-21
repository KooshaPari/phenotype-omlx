import React from 'react';
import { Cell, FailMode } from '../types';

interface Props {
  cells: Cell[];
  failMode: FailMode;
  onFailMode: (m: FailMode) => void;
  onSelect: (c: Cell) => void;
}

function cellStatus(c: Cell): 'ok' | 'fail' | 'timeout' {
  if (c.wall_clock_s >= 59 && (c.tokens_per_second || 0) === 0) return 'timeout';
  if (!c.ok || c.partial_credit < 0.5) return 'fail';
  return 'ok';
}

export default function Failures({ cells, failMode, onFailMode, onSelect }: Props) {
  let filtered = cells.filter(c => cellStatus(c) !== 'ok');
  if (failMode === 'timeout') filtered = filtered.filter(c => cellStatus(c) === 'timeout');
  if (failMode === 'low-pc') filtered = filtered.filter(c => c.partial_credit < 0.5);
  if (failMode === 'hallucination') filtered = filtered.filter(c => c.hallucination_count > 0);
  filtered.sort((a, b) => b.partial_credit - a.partial_credit || b.wall_clock_s - a.wall_clock_s);

  const keys = ['suite', 'task_id', 'variant', 'difficulty', 'wall_clock_s', 'partial_credit', 'hallucination_count', 'failure_analysis'];
  const labels = ['Suite', 'Task', 'Var', 'Diff', 'Wall', 'PC', 'Halluc', 'Factor'];

  const modes: Array<{ key: FailMode; label: string }> = [
    { key: 'all', label: 'All' },
    { key: 'timeout', label: 'Timeouts' },
    { key: 'low-pc', label: 'Low PC' },
    { key: 'hallucination', label: 'Hallucinations' },
  ];

  return (
    <div className="view-content">
      <div className="fails-toolbar">
        <span className="cells-count">{filtered.length} failures</span>
        <div className="fails-group">
          {modes.map(m => (
            <button key={m.key} className={`gt-btn ${failMode === m.key ? 'on' : ''}`} onClick={() => onFailMode(m.key)}>{m.label}</button>
          ))}
        </div>
      </div>
      <div className="fail-table-wrap">
        <table className="fail-table">
          <thead>
            <tr>{keys.map((k, i) => <th key={k}>{labels[i]}</th>)}</tr>
          </thead>
          <tbody>
            {filtered.slice(0, 500).map((c, i) => (
              <tr key={i} onClick={() => onSelect(c)}>
                <td>{c.suite}</td>
                <td className="mono">{c.task_id}</td>
                <td><span className={`vp ${c.variant}`}>{c.variant}</span></td>
                <td><span className={`dp ${c.difficulty}`}>{c.difficulty}</span></td>
                <td>{c.wall_clock_s.toFixed(1)}s</td>
                <td>{c.partial_credit.toFixed(2)}</td>
                <td>{c.hallucination_count}</td>
                <td className="faint">{c.failure_analysis?.primary_factor || '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
