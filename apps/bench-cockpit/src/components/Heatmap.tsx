import React, { useMemo, useState } from 'react';
import { Cell } from '../types';

interface Props {
  cells: Cell[];
}

type Metric =
  | 'pass_at_1'
  | 'judge_score'
  | 'intent_preservation_rate'
  | 'hallucination_count'
  | 'format_compliance_rate'
  | 'wall_clock_s';

function cellMetric(c: Cell, m: Metric): number {
  switch (m) {
    case 'pass_at_1':
      return c.pass_at_1;
    case 'judge_score':
      return c.judge_score;
    case 'intent_preservation_rate':
      return c.intent_preservation_rate;
    case 'hallucination_count':
      return c.hallucination_count;
    case 'format_compliance_rate':
      return c.format_compliance_rate;
    case 'wall_clock_s':
      return c.wall_clock_s;
  }
}

/** HELM-style variant × task heatmap. */
export default function Heatmap({ cells }: Props) {
  const [metric, setMetric] = useState<Metric>('pass_at_1');
  const { tasks, variants, grid, max } = useMemo(() => {
    const variants = [...new Set(cells.map((c) => c.variant))].sort();
    const tasks = [...new Set(cells.map((c) => `${c.suite}::${c.task_id}`))].sort();
    const grid = new Map<string, number>();
    let max = 0;
    for (const c of cells) {
      const k = `${c.variant}|${c.suite}::${c.task_id}`;
      const v = cellMetric(c, metric);
      grid.set(k, v);
      max = Math.max(max, v);
    }
    return { tasks, variants, grid, max: max || 1 };
  }, [cells, metric]);

  const color = (v: number) => {
    const t = Math.max(0, Math.min(1, v / max));
    const r = Math.round(30 + t * 200);
    const g = Math.round(40 + (1 - Math.abs(t - 0.5) * 2) * 120);
    const b = Math.round(80 + (1 - t) * 140);
    return `rgb(${r},${g},${b})`;
  };

  return (
    <div className="viz-panel" data-testid="heatmap">
      <div className="viz-toolbar">
        <span className="viz-title">HELM heatmap</span>
        <select value={metric} onChange={(e) => setMetric(e.target.value as Metric)}>
          <option value="pass_at_1">pass@1</option>
          <option value="judge_score">judge_score</option>
          <option value="intent_preservation_rate">intent</option>
          <option value="hallucination_count">hallucinations</option>
          <option value="format_compliance_rate">format</option>
          <option value="wall_clock_s">wall_s</option>
        </select>
      </div>
      <div className="heat-wrap">
        <table className="heat-table">
          <thead>
            <tr>
              <th>task</th>
              {variants.map((v) => (
                <th key={v}>{v}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {tasks.slice(0, 80).map((t) => (
              <tr key={t}>
                <td className="heat-label">{t}</td>
                {variants.map((v) => {
                  const val = grid.get(`${v}|${t}`);
                  return (
                    <td
                      key={v}
                      className="heat-cell"
                      style={{ background: val == null ? 'transparent' : color(val) }}
                      title={val == null ? '—' : val.toFixed(3)}
                    >
                      {val == null ? '' : val.toFixed(2)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
