import React, { useRef, useCallback } from 'react';
import { useVirtualizer } from '@tanstack/react-virtual';
import { Cell } from '../types';
import { BenchState } from '../state/useBenchState';

interface Props {
  cells: Cell[];
  state: BenchState;
  onSort: (key: string) => void;
  onSelect: (cell: Cell | null) => void;
  onGroup: (g: BenchState['groupBy']) => void;
}

const COLS: Array<{ key: string; label: string; w: number; fmt: (v: any) => string }> = [
  { key: 'task_id', label: 'Task', w: 130, fmt: v => v },
  { key: 'variant', label: 'Var', w: 52, fmt: v => v },
  { key: 'suite', label: 'Suite', w: 100, fmt: v => v },
  { key: 'difficulty', label: 'Diff', w: 60, fmt: v => v },
  { key: 'ok', label: 'Status', w: 72, fmt: v => v ? 'ok' : 'fail' },
  { key: 'wall_clock_s', label: 'Wall', w: 70, fmt: v => v.toFixed(2) + 's' },
  { key: 'pass_at_1', label: 'P@1', w: 56, fmt: v => (v * 100).toFixed(0) + '%' },
  { key: 'partial_credit', label: 'PC', w: 56, fmt: v => v.toFixed(3) },
  { key: 'format_compliance_rate', label: 'Fmt', w: 56, fmt: v => (v * 100).toFixed(0) + '%' },
  { key: 'judge_score', label: 'Judge', w: 60, fmt: v => v ? v.toFixed(2) : '—' },
  { key: 'tokens_per_second', label: 'Tok/s', w: 60, fmt: v => v ? v.toFixed(1) : '—' },
  { key: 'hallucination_count', label: 'Hal', w: 44, fmt: v => v },
  { key: 'first_token_latency_ms', label: 'TTFT', w: 60, fmt: v => v ? (v / 1000).toFixed(2) + 's' : '—' },
  { key: 'cost_usd', label: '$', w: 64, fmt: v => '$' + v.toFixed(4) },
  { key: 'retry_count', label: 'Ret', w: 40, fmt: v => v },
  { key: 'peak_rss_mb', label: 'RSS', w: 56, fmt: v => v ? v.toFixed(0) + 'M' : '—' },
  { key: 'peak_gpu_mem_mb', label: 'GPU', w: 56, fmt: v => v ? v.toFixed(0) + 'M' : '—' },
  { key: 'energy_proxy_joules', label: 'J', w: 52, fmt: v => v ? v.toFixed(1) : '—' },
];

export default function CellsTable({ cells, state, onSort, onSelect, onGroup }: Props) {
  const parentRef = useRef<HTMLDivElement>(null);
  const sorted = useCallback(() => {
    const arr = [...cells];
    const k = state.sortKey;
    const dir = state.sortDir;
    arr.sort((a, b) => {
      const av = a[k], bv = b[k];
      if (k === 'ok') return dir * (Number(bv) - Number(av));
      if (typeof av === 'string' || typeof bv === 'string') return dir * String(av).localeCompare(String(bv));
      return dir * ((av || 0) - (bv || 0));
    });
    return arr;
  }, [cells, state.sortKey, state.sortDir])();

  const rows = useCallback(() => {
    if (state.groupBy === 'none') return sorted.map(c => ({ kind: 'cell' as const, cell: c }));
    const groups = new Map<string, Cell[]>();
    for (const c of sorted) {
      let gk: string;
      if (state.groupBy === 'status') gk = c.ok ? 'ok' : (c.wall_clock_s >= 59 && !c.tokens_per_second ? 'timeout' : 'fail');
      else if (state.groupBy === 'variant') gk = c.variant;
      else gk = c[state.groupBy] || 'other';
      if (!groups.has(gk)) groups.set(gk, []);
      groups.get(gk)!.push(c);
    }
    const result: Array<{ kind: 'group' | 'cell'; label?: string; cell?: Cell }> = [];
    for (const [gk, arr] of groups) {
      result.push({ kind: 'group', label: `${gk} (${arr.length})` });
      for (const c of arr) result.push({ kind: 'cell', cell: c });
    }
    return result;
  }, [sorted, state.groupBy])();

  const virtualizer = useVirtualizer({
    count: rows.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 28,
    overscan: 20,
  });

  const sortedKey = state.sortKey;
  const col = COLS.find(c => c.key === sortedKey);
  const colLabel = col?.label || sortedKey;

  return (
    <div className="view-content">
      <div className="cells-toolbar">
        <span className="cells-count">{cells.length} cells</span>
        <span className="cells-sort">sorted by {colLabel} {state.sortDir < 0 ? '▼' : '▲'}</span>
        <div className="cells-group">
          {(['none', 'suite', 'difficulty', 'variant', 'status'] as const).map(g => (
            <button key={g} className={`gt-btn ${state.groupBy === g ? 'on' : ''}`} onClick={() => onGroup(g)}>{g}</button>
          ))}
        </div>
      </div>
      <div className="cells-wrap">
        <div className="cells-header">
          {COLS.map(c => (
            <div key={c.key} className="ch" style={{ width: c.w }} onClick={() => onSort(c.key)}>
              {c.label}
              {sortedKey === c.key && <span className="csort">{state.sortDir < 0 ? '▼' : '▲'}</span>}
            </div>
          ))}
        </div>
        <div ref={parentRef} className="cells-scroll">
          <div style={{ height: virtualizer.getTotalSize(), position: 'relative' }}>
            {virtualizer.getVirtualItems().map(virtualItem => {
              const row = rows[virtualItem.index];
              if (row.kind === 'group') {
                return (
                  <div key={virtualItem.index} className="cg-row" style={{ height: 28, transform: `translateY(${virtualItem.start}px)` }}>
                    <span className="cg-label">{row.label}</span>
                  </div>
                );
              }
              const c = row.cell!;
              return (
                <div
                  key={virtualItem.index}
                  className={`cd-row ${state.selectedCell?.task_id === c.task_id && state.selectedCell?.variant === c.variant ? 'selected' : ''}`}
                  style={{ height: 28, transform: `translateY(${virtualItem.start}px)` }}
                  onClick={() => onSelect(c)}
                >
                  {COLS.map(col => (
                    <div key={col.key} className="cd" style={{ width: col.w }}>
                      {col.key === 'ok' ? (
                        <span className={`sp ${c.ok ? 'ok' : 'fail'}`}>{c.ok ? 'ok' : 'fail'}</span>
                      ) : col.key === 'variant' ? (
                        <span className={`vp ${c.variant}`}>{c.variant}</span>
                      ) : col.key === 'difficulty' ? (
                        <span className={`dp ${c.difficulty}`}>{c.difficulty}</span>
                      ) : col.key === 'wall_clock_s' ? (
                        <><span className="mini-bar" style={{ width: Math.min(c.wall_clock_s / 60 * 100, 100) + '%', background: c.wall_clock_s > 30 ? 'var(--amber)' : 'var(--accent-dim)' }} />{col.fmt(c[col.key])}</>
                      ) : (
                        col.fmt(c[col.key])
                      )}
                    </div>
                  ))}
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
