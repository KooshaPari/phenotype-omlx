import React, { useMemo, useState } from 'react';
import { Cell, HistoryEntry } from '../types';
import { resolveRlvr } from '../lib/rlvr';

interface Props {
  suite: string;
  taskId: string;
  cells: Cell[];
  history: HistoryEntry[];
  initialVariant?: 'stock' | 'ours';
  onBack: () => void;
  onOpenSuite: () => void;
}

export default function TaskPage({
  suite,
  taskId,
  cells,
  history,
  initialVariant = 'ours',
  onBack,
  onOpenSuite,
}: Props) {
  const variants = useMemo(() => {
    const set = new Set(
      cells.filter((c) => c.suite === suite && c.task_id === taskId).map((c) => c.variant),
    );
    return (['stock', 'ours'] as const).filter((v) => set.has(v));
  }, [cells, suite, taskId]);

  const [variant, setVariant] = useState<'stock' | 'ours'>(
    variants.includes(initialVariant) ? initialVariant : variants[0] || 'ours',
  );
  const [runIdx, setRunIdx] = useState(0); // 0 = live/current payload (latest history entry)

  const cell = useMemo(
    () => cells.find((c) => c.suite === suite && c.task_id === taskId && c.variant === variant) || null,
    [cells, suite, taskId, variant],
  );

  const oursTrend = useMemo(() => {
    // History only stores summary aggregates today — approximate ours PC/pass from summary.
    return history.map((h, i) => ({
      run: i + 1,
      at: h.receivedAt,
      pass: h.summary?.by_variant?.ours?.pass_at_1 ?? 0,
      pc: h.summary?.by_variant?.ours?.mean_partial_credit ?? 0,
      wall: h.summary?.by_variant?.ours?.mean_wall_clock_s ?? 0,
      n: h.cellCount,
    }));
  }, [history]);

  const rlvr = cell ? resolveRlvr(cell) : null;

  return (
    <div className="view-content task-page">
      <div className="detail-nav">
        <button type="button" className="gt-btn" onClick={onBack}>← Back</button>
        <button type="button" className="gt-btn" onClick={onOpenSuite}>Suite {suite}</button>
        <h2 className="detail-title mono">{taskId}</h2>
      </div>

      <div className="task-controls">
        <label>
          Variant
          <select value={variant} onChange={(e) => setVariant(e.target.value as 'stock' | 'ours')}>
            {variants.map((v) => (
              <option key={v} value={v}>{v}</option>
            ))}
          </select>
        </label>
        <label>
          Run
          <select value={runIdx} onChange={(e) => setRunIdx(Number(e.target.value))}>
            <option value={0}>live (current)</option>
            {history.map((h, i) => (
              <option key={i} value={i + 1}>
                hist #{i + 1} · {h.cellCount} cells · {new Date(h.receivedAt).toLocaleTimeString()}
              </option>
            ))}
          </select>
        </label>
        {runIdx > 0 && (
          <span className="faint">History stores suite-level summary only — cell body stays live.</span>
        )}
      </div>

      {!cell ? (
        <div className="empty-state">No cell for {suite}/{taskId} · {variant}</div>
      ) : (
        <div className="ov-grid">
          <div className="ov-card">
            <div className="ov-title">Scores</div>
            <div className="kv"><span className="k">pass@1</span><span className="v">{(cell.pass_at_1 * 100).toFixed(1)}%</span></div>
            <div className="kv"><span className="k">PC</span><span className="v">{cell.partial_credit.toFixed(3)}</span></div>
            <div className="kv"><span className="k">judge</span><span className="v">{cell.judge_score?.toFixed(3) ?? '—'}</span></div>
            <div className="kv"><span className="k">format</span><span className="v">{((cell.format_compliance_rate || 0) * 100).toFixed(0)}%</span></div>
          </div>
          <div className="ov-card">
            <div className="ov-title">Perf</div>
            <div className="kv"><span className="k">wall</span><span className="v">{cell.wall_clock_s.toFixed(2)}s</span></div>
            <div className="kv"><span className="k">tok/s</span><span className="v">{(cell.tokens_per_second || 0).toFixed(1)}</span></div>
            <div className="kv"><span className="k">ttft</span><span className="v">{cell.first_token_latency_ms ? (cell.first_token_latency_ms / 1000).toFixed(2) + 's' : '—'}</span></div>
          </div>
          {rlvr && (
            <div className="ov-card">
              <div className="ov-title">RLVR ({rlvr.source})</div>
              <div className="kv"><span className="k">composite</span><span className="v">{rlvr.composite.toFixed(3)}</span></div>
              <div className="kv"><span className="k">L0–L3</span><span className="v mono">{rlvr.l0.toFixed(2)} / {rlvr.l1.toFixed(2)} / {rlvr.l2.toFixed(2)} / {rlvr.l3.toFixed(2)}</span></div>
            </div>
          )}
        </div>
      )}

      <h3 className="section-title">Ours change over runs (summary ring)</h3>
      {oursTrend.length < 2 ? (
        <p className="faint">Need ≥2 history snapshots (reconnect / data reload) to show ours trend.</p>
      ) : (
        <table className="heat-table">
          <thead>
            <tr><th>Run</th><th>When</th><th>Ours P@1</th><th>Ours PC</th><th>Ours Wall</th><th>n</th></tr>
          </thead>
          <tbody>
            {oursTrend.map((r) => (
              <tr key={r.run}>
                <td>#{r.run}</td>
                <td className="faint">{new Date(r.at).toLocaleTimeString()}</td>
                <td>{(r.pass * 100).toFixed(1)}%</td>
                <td>{r.pc.toFixed(3)}</td>
                <td>{r.wall.toFixed(2)}s</td>
                <td>{r.n}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
