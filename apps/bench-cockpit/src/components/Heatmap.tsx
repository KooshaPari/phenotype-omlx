import React, { useMemo, useState } from 'react';
import { EChart } from '../lib/echart';
import { Cell } from '../types';

interface Props {
  cells: Cell[];
  onSelect?: (c: Cell) => void;
}

type Metric =
  | 'pass_at_1'
  | 'judge_score'
  | 'intent_preservation_rate'
  | 'hallucination_count'
  | 'format_compliance_rate'
  | 'wall_clock_s'
  | 'tokens_per_second';

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
    case 'tokens_per_second':
      return c.tokens_per_second;
  }
}

/** HELM-style variant × task heatmap (echarts). Caps rows for readability. */
export default function Heatmap({ cells, onSelect }: Props) {
  const [metric, setMetric] = useState<Metric>('pass_at_1');
  const MAX_TASKS = 60;

  const { option, lookup } = useMemo(() => {
    const variants = [...new Set(cells.map((c) => c.variant))].sort();
    const allTasks = [...new Set(cells.map((c) => `${c.suite}::${c.task_id}`))].sort();
    const tasks = allTasks.slice(0, MAX_TASKS);
    const lookup = new Map<string, Cell>();
    for (const c of cells) {
      lookup.set(`${c.variant}|${c.suite}::${c.task_id}`, c);
    }

    const data: [number, number, number][] = [];
    let max = 0;
    for (let yi = 0; yi < tasks.length; yi++) {
      for (let xi = 0; xi < variants.length; xi++) {
        const c = lookup.get(`${variants[xi]}|${tasks[yi]}`);
        if (!c) continue;
        const v = cellMetric(c, metric);
        max = Math.max(max, v);
        data.push([xi, yi, v]);
      }
    }
    if (max <= 0) max = 1;

    const option = {
      backgroundColor: 'transparent',
      tooltip: {
        position: 'top',
        formatter: (p: { value?: [number, number, number] }) => {
          const [xi, yi, v] = p.value ?? [0, 0, 0];
          const task = tasks[yi] ?? '?';
          const variant = variants[xi] ?? '?';
          return `${variant} · ${task}<br/>${metric}: ${Number(v).toFixed(3)}`;
        },
      },
      grid: { left: 160, right: 24, top: 16, bottom: 48 },
      xAxis: {
        type: 'category',
        data: variants,
        splitArea: { show: true },
        axisLabel: { color: '#9aa3b2' },
      },
      yAxis: {
        type: 'category',
        data: tasks,
        axisLabel: {
          color: '#9aa3b2',
          fontSize: 10,
          formatter: (s: string) => (s.length > 28 ? `${s.slice(0, 26)}…` : s),
        },
      },
      visualMap: {
        min: 0,
        max,
        calculable: true,
        orient: 'horizontal',
        left: 'center',
        bottom: 0,
        textStyle: { color: '#9aa3b2' },
        inRange: {
          color: ['#1a2332', '#2d4a6f', '#3d7a5a', '#c9a227', '#c44'],
        },
      },
      series: [
        {
          name: metric,
          type: 'heatmap',
          data,
          label: {
            show: variants.length <= 3 && tasks.length <= 24,
            color: '#e8ecf1',
            fontSize: 9,
            formatter: (p: { value?: [number, number, number] }) =>
              p.value ? Number(p.value[2]).toFixed(2) : '',
          },
          emphasis: {
            itemStyle: { shadowBlur: 8, shadowColor: 'rgba(0,0,0,0.45)' },
          },
        },
      ],
    };

    return { option, lookup: { tasks, variants, cells: lookup } };
  }, [cells, metric]);

  const height = Math.min(720, Math.max(280, 28 + (option.yAxis as { data: string[] }).data.length * 18));

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
          <option value="tokens_per_second">tok/s</option>
        </select>
      </div>
      <EChart
        option={option}
        style={{ height, width: '100%' }}
        opts={{ renderer: 'canvas' }}
        onEvents={{
          click: (p) => {
            const val = p.value as [number, number, number] | undefined;
            if (!val || !onSelect) return;
            const [xi, yi] = val;
            const task = lookup.tasks[yi];
            const variant = lookup.variants[xi];
            if (!task || !variant) return;
            const c = lookup.cells.get(`${variant}|${task}`);
            if (c) onSelect(c);
          },
        }}
      />
    </div>
  );
}
