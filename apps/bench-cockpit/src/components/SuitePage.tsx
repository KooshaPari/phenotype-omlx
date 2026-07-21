import React, { useMemo } from 'react';
import { Cell } from '../types';

interface Props {
  suite: string;
  cells: Cell[];
  onOpenTask: (taskId: string, variant: 'stock' | 'ours') => void;
  onBack: () => void;
}

function mean(vals: number[]): number {
  return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 0;
}

export default function SuitePage({ suite, cells, onOpenTask, onBack }: Props) {
  const suiteCells = useMemo(
    () => cells.filter((c) => c.suite === suite),
    [cells, suite],
  );
  const stock = suiteCells.filter((c) => c.variant === 'stock');
  const ours = suiteCells.filter((c) => c.variant === 'ours');
  const tasks = useMemo(() => {
    const m = new Map<string, { stock?: Cell; ours?: Cell }>();
    for (const c of suiteCells) {
      const cur = m.get(c.task_id) || {};
      if (c.variant === 'stock') cur.stock = c;
      else cur.ours = c;
      m.set(c.task_id, cur);
    }
    return [...m.entries()].sort((a, b) => a[0].localeCompare(b[0]));
  }, [suiteCells]);

  const sPc = mean(stock.map((c) => c.partial_credit));
  const oPc = mean(ours.map((c) => c.partial_credit));
  const sWall = mean(stock.map((c) => c.wall_clock_s));
  const oWall = mean(ours.map((c) => c.wall_clock_s));
  const sTps = mean(stock.map((c) => c.tokens_per_second || 0));
  const oTps = mean(ours.map((c) => c.tokens_per_second || 0));

  return (
    <div className="view-content suite-page">
      <div className="detail-nav">
        <button type="button" className="gt-btn" onClick={onBack}>← Suites</button>
        <h2 className="detail-title">{suite}</h2>
        <span className="faint">{tasks.length} tasks · {suiteCells.length} cells</span>
      </div>

      <div className="ov-grid">
        <div className="ov-card">
          <div className="ov-title">Partial credit</div>
          <div className="ov-metric">
            <span className="ov-stock">{sPc.toFixed(3)}</span>
            <span className="ov-sep">·</span>
            <span className="ov-ours">{oPc.toFixed(3)}</span>
          </div>
        </div>
        <div className="ov-card">
          <div className="ov-title">Wall</div>
          <div className="ov-metric">
            <span className="ov-stock">{sWall.toFixed(2)}s</span>
            <span className="ov-sep">·</span>
            <span className="ov-ours">{oWall.toFixed(2)}s</span>
          </div>
        </div>
        <div className="ov-card">
          <div className="ov-title">Tok/s</div>
          <div className="ov-metric">
            <span className="ov-stock">{sTps.toFixed(1)}</span>
            <span className="ov-sep">·</span>
            <span className="ov-ours">{oTps.toFixed(1)}</span>
          </div>
        </div>
      </div>

      <table className="heat-table suites-table">
        <thead>
          <tr>
            <th>Task</th>
            <th>PC s</th>
            <th>PC o</th>
            <th>Wall s</th>
            <th>Wall o</th>
            <th>Tok/s s</th>
            <th>Tok/s o</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {tasks.map(([taskId, pair]) => (
            <tr key={taskId} className="task-row">
              <td className="mono">{taskId}</td>
              <td className="heat-cell stock"><span className="v">{pair.stock ? pair.stock.partial_credit.toFixed(3) : '—'}</span></td>
              <td className="heat-cell ours"><span className="v">{pair.ours ? pair.ours.partial_credit.toFixed(3) : '—'}</span></td>
              <td className="mono faint">{pair.stock ? pair.stock.wall_clock_s.toFixed(2) : '—'}</td>
              <td className="mono faint">{pair.ours ? pair.ours.wall_clock_s.toFixed(2) : '—'}</td>
              <td className="mono faint">{pair.stock?.tokens_per_second?.toFixed(1) ?? '—'}</td>
              <td className="mono faint">{pair.ours?.tokens_per_second?.toFixed(1) ?? '—'}</td>
              <td>
                <button type="button" className="gt-btn" onClick={() => onOpenTask(taskId, pair.ours ? 'ours' : 'stock')}>
                  Open
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
