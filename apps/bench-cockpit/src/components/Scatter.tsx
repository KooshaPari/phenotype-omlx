import React, { useMemo, useState } from 'react';
import ReactECharts from 'echarts-for-react';
import type { ComponentType } from 'react';
import { Cell } from '../types';

// echarts-for-react types lag React 19 — cast once.
const Chart = ReactECharts as unknown as ComponentType<{
  option: Record<string, unknown>;
  style?: React.CSSProperties;
  opts?: { renderer?: string };
  onEvents?: Record<string, (p: { value?: [number, number, string] }) => void>;
}>;

interface Props {
  cells: Cell[];
  onSelect?: (c: Cell) => void;
}

type YMetric = 'pass_at_1' | 'judge_score' | 'tokens_per_second';

/** Pareto-ish scatter via echarts: quality/perf vs wall-clock. */
export default function Scatter({ cells, onSelect }: Props) {
  const [metric, setMetric] = useState<YMetric>('judge_score');

  const { option, byKey } = useMemo(() => {
    const byKey = new Map<string, Cell>();
    const stock: [number, number, string][] = [];
    const ours: [number, number, string][] = [];
    for (const c of cells) {
      if (!(c.wall_clock_s > 0)) continue;
      const y =
        metric === 'pass_at_1'
          ? c.pass_at_1
          : metric === 'tokens_per_second'
            ? c.tokens_per_second
            : c.judge_score;
      const key = `${c.variant}|${c.suite}|${c.task_id}`;
      byKey.set(key, c);
      const pt: [number, number, string] = [c.wall_clock_s, y, key];
      if (c.variant === 'ours') ours.push(pt);
      else stock.push(pt);
    }
    const option = {
      backgroundColor: 'transparent',
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
        splitLine: { lineStyle: { color: '#2a3140' } },
      },
      yAxis: {
        type: 'value',
        name: metric,
        nameLocation: 'middle',
        nameGap: 40,
        axisLabel: { color: '#9aa3b2' },
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
    return { option, byKey };
  }, [cells, metric]);

  return (
    <div className="scatter-panel">
      <div className="scatter-toolbar">
        <label>
          Y metric{' '}
          <select value={metric} onChange={(e) => setMetric(e.target.value as YMetric)}>
            <option value="judge_score">judge_score</option>
            <option value="pass_at_1">pass_at_1</option>
            <option value="tokens_per_second">tokens_per_second</option>
          </select>
        </label>
      </div>
      <Chart
        option={option}
        style={{ height: 380, width: '100%' }}
        opts={{ renderer: 'canvas' }}
        onEvents={{
          click: (p) => {
            const key = p.value?.[2];
            const c = key ? byKey.get(key) : undefined;
            if (c && onSelect) onSelect(c);
          },
        }}
      />
    </div>
  );
}
