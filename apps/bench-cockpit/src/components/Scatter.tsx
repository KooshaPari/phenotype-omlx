import React, { useMemo, useState } from 'react';
import { EChart } from '../lib/echart';
import { Cell } from '../types';
import { effectiveGenOk } from '../lib/metrics';

interface Props {
  cells: Cell[];
  onSelect?: (c: Cell) => void;
}

type YMetric = 'pass_at_1' | 'partial_credit' | 'tokens_per_second';

/** Pareto-ish scatter via echarts: quality/perf vs wall-clock. */
export default function Scatter({ cells, onSelect }: Props) {
  // Default partial_credit — judge_score is often all-zeros on V5 (reads as flat/black).
  const [metric, setMetric] = useState<YMetric>('partial_credit');

  const { option, byKey, nPts } = useMemo(() => {
    const byKey = new Map<string, Cell>();
    const stock: [number, number, string][] = [];
    const ours: [number, number, string][] = [];
    for (const c of cells) {
      if (!(c.wall_clock_s > 0)) continue;
      const y =
        metric === 'pass_at_1'
          ? effectiveGenOk(c)
          : metric === 'tokens_per_second'
            ? c.tokens_per_second
            : c.partial_credit;
      const key = `${c.variant}|${c.suite}|${c.task_id}`;
      byKey.set(key, c);
      const pt: [number, number, string] = [c.wall_clock_s, y, key];
      if (c.variant === 'ours') ours.push(pt);
      else stock.push(pt);
    }
    const option = {
      backgroundColor: '#12161f',
      tooltip: {
        trigger: 'item',
        formatter: (p: { value?: [number, number, string] }) => {
          const key = p.value?.[2];
          const c = key ? byKey.get(key) : undefined;
          if (!c) return '';
          return `${c.variant} · ${c.suite}/${c.task_id}<br/>wall ${c.wall_clock_s.toFixed(2)}s · y=${(p.value?.[1] ?? 0).toFixed(3)}`;
        },
      },
      legend: { data: ['stock', 'ours'], textStyle: { color: '#9aa3b2' } },
      grid: { left: 56, right: 24, top: 40, bottom: 48 },
      xAxis: {
        type: 'log',
        name: 'wall_clock_s (log)',
        nameLocation: 'middle',
        nameGap: 28,
        axisLabel: { color: '#9aa3b2' },
        axisLine: { lineStyle: { color: '#3a4254' } },
        splitLine: { lineStyle: { color: '#2a3140' } },
      },
      yAxis: {
        type: 'value',
        name: metric === 'pass_at_1' ? 'gen_ok' : metric,
        nameLocation: 'middle',
        nameGap: 40,
        axisLabel: { color: '#9aa3b2' },
        axisLine: { lineStyle: { color: '#3a4254' } },
        splitLine: { lineStyle: { color: '#2a3140' } },
      },
      series: [
        {
          name: 'stock',
          type: 'scatter',
          symbolSize: 10,
          itemStyle: { color: '#6b8cae' },
          data: stock,
        },
        {
          name: 'ours',
          type: 'scatter',
          symbolSize: 10,
          itemStyle: { color: '#e8a87c' },
          data: ours,
        },
      ],
    };
    return { option, byKey, nPts: stock.length + ours.length };
  }, [cells, metric]);

  return (
    <div className="viz-panel" data-testid="scatter">
      <div className="viz-toolbar">
        <span className="viz-title">Pareto scatter</span>
        <label>
          Y{' '}
          <select value={metric} onChange={(e) => setMetric(e.target.value as YMetric)}>
            <option value="partial_credit">partial_credit</option>
            <option value="pass_at_1">gen_ok</option>
            <option value="tokens_per_second">tokens_per_second</option>
          </select>
        </label>
      </div>
      {nPts === 0 ? (
        <div className="viz-empty">No points with wall_clock_s &gt; 0.</div>
      ) : (
        <EChart
          option={option}
          style={{ height: 380, width: '100%' }}
          opts={{ renderer: 'canvas' }}
          onEvents={{
            click: (p) => {
              const val = p.value as [number, number, string] | undefined;
              const key = val?.[2];
              const c = key ? byKey.get(key) : undefined;
              if (c && onSelect) onSelect(c);
            },
          }}
        />
      )}
    </div>
  );
}
