import React, { useMemo } from 'react';
import { Cell } from '../types';

interface Props {
  cells: Cell[];
  onSelect?: (c: Cell) => void;
}

/** Failure-mode matrix: domain × primary_factor. */
export default function FailMatrix({ cells, onSelect }: Props) {
  const { domains, factors, counts, max } = useMemo(() => {
    const domains = [...new Set(cells.map((c) => c.suite))].sort();
    const factors = [
      ...new Set(
        cells.map((c) => String(c.failure_analysis?.primary_factor || (c.ok ? 'ok' : 'unknown')))
      ),
    ].sort();
    const counts = new Map<string, { n: number; sample?: Cell }>();
    let max = 0;
    for (const c of cells) {
      const f = String(c.failure_analysis?.primary_factor || (c.ok ? 'ok' : 'unknown'));
      const k = `${c.suite}|${f}`;
      const prev = counts.get(k) || { n: 0 };
      prev.n += 1;
      prev.sample = c;
      counts.set(k, prev);
      max = Math.max(max, prev.n);
    }
    return { domains, factors, counts, max: max || 1 };
  }, [cells]);

  return (
    <div className="viz-panel" data-testid="fail-matrix">
      <div className="viz-toolbar">
        <span className="viz-title">Failure matrix</span>
        <span className="viz-hint">click cell → sample</span>
      </div>
      <div className="heat-wrap">
        <table className="heat-table">
          <thead>
            <tr>
              <th>suite</th>
              {factors.map((f) => (
                <th key={f}>{f}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {domains.map((d) => (
              <tr key={d}>
                <td className="heat-label">{d}</td>
                {factors.map((f) => {
                  const entry = counts.get(`${d}|${f}`);
                  const n = entry?.n ?? 0;
                  const t = n / max;
                  return (
                    <td
                      key={f}
                      className="heat-cell clickable"
                      style={{
                        background: n
                          ? `rgba(220, 80, 60, ${0.15 + t * 0.75})`
                          : 'transparent',
                      }}
                      onClick={() => entry?.sample && onSelect?.(entry.sample)}
                    >
                      {n || ''}
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
