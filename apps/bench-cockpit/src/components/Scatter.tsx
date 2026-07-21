import React, { useMemo, useState } from 'react';
import { Cell } from '../types';

interface Props {
  cells: Cell[];
}

type Metric = 'pass_at_1' | 'judge_score' | 'wall_clock_s' | 'tokens_per_second';

/** Pareto-ish SVG scatter: Quality (y) vs Performance (x=wall). */
export default function Scatter({ cells }: Props) {
  const [metric, setMetric] = useState<Metric>('judge_score');
  const W = 640;
  const H = 360;
  const pad = 40;

  const pts = useMemo(() => {
    return cells
      .filter((c) => c.wall_clock_s > 0)
      .map((c) => {
        const yRaw =
          metric === 'pass_at_1'
            ? c.pass_at_1
            : metric === 'judge_score'
              ? c.judge_score
              : metric === 'tokens_per_second'
                ? c.tokens_per_second
                : c.wall_clock_s;
        return {
          x: Math.log10(Math.max(c.wall_clock_s, 1e-3)),
          y: yRaw,
          r: Math.max(3, Math.min(14, (c.peak_rss_mb || 100) / 80)),
          variant: c.variant,
          label: `${c.suite}/${c.task_id}`,
        };
      });
  }, [cells, metric]);

  const xs = pts.map((p) => p.x);
  const ys = pts.map((p) => p.y);
  const xmin = Math.min(...xs, 0);
  const xmax = Math.max(...xs, 1);
  const ymin = Math.min(...ys, 0);
  const ymax = Math.max(...ys, 1);

  const sx = (x: number) => pad + ((x - xmin) / (xmax - xmin || 1)) * (W - 2 * pad);
  const sy = (y: number) => H - pad - ((y - ymin) / (ymax - ymin || 1)) * (H - 2 * pad);

  return (
    <div className="viz-panel" data-testid="scatter">
      <div className="viz-toolbar">
        <span className="viz-title">Quality × Performance (Pareto)</span>
        <select value={metric} onChange={(e) => setMetric(e.target.value as Metric)}>
          <option value="judge_score">judge_score (y)</option>
          <option value="pass_at_1">pass@1 (y)</option>
          <option value="tokens_per_second">tok/s (y)</option>
        </select>
        <span className="viz-hint">x = log10(wall_s) · bubble = RSS</span>
      </div>
      <svg width={W} height={H} className="viz-svg">
        <line x1={pad} y1={H - pad} x2={W - pad} y2={H - pad} stroke="currentColor" opacity={0.3} />
        <line x1={pad} y1={pad} x2={pad} y2={H - pad} stroke="currentColor" opacity={0.3} />
        {pts.map((p, i) => (
          <circle
            key={i}
            cx={sx(p.x)}
            cy={sy(p.y)}
            r={p.r}
            fill={p.variant === 'ours' ? '#3ddc97' : '#6ea8fe'}
            opacity={0.75}
          >
            <title>{`${p.label} · ${p.variant}`}</title>
          </circle>
        ))}
      </svg>
      <div className="viz-legend">
        <span className="leg ours">ours</span>
        <span className="leg stock">stock</span>
      </div>
    </div>
  );
}
