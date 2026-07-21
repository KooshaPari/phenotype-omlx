import React, { useMemo } from 'react';
import { Cell } from '../types';

interface Props { cells: Cell[] }

function perSuite(cells: Cell[]): Map<string, Cell[]> {
  const m = new Map<string, Cell[]>();
  for (const c of cells) { const a = m.get(c.suite) || []; a.push(c); m.set(c.suite, a); }
  return m;
}
function mean(a: number[]) { return a.length ? a.reduce((s, x) => s + x, 0) / a.length : 0; }

export default function Suites({ cells }: Props) {
  const suites = useMemo(() => [...perSuite(cells).entries()].sort((a, b) => a[0].localeCompare(b[0])), [cells]);
  const metrics = [
    { key: 'wall_clock_s', label: 'Wall', fmt: (v: number) => v.toFixed(2) + 's', src: (c: Cell) => c.wall_clock_s },
    { key: 'pass_at_1', label: 'P@1', fmt: (v: number) => (v * 100).toFixed(0) + '%', src: (c: Cell) => c.pass_at_1 },
    { key: 'partial_credit', label: 'PC', fmt: (v: number) => v.toFixed(2), src: (c: Cell) => c.partial_credit },
    { key: 'format_compliance_rate', label: 'Fmt', fmt: (v: number) => (v * 100).toFixed(0) + '%', src: (c: Cell) => c.format_compliance_rate },
    { key: 'judge_score', label: 'Judge', fmt: (v: number) => v ? v.toFixed(2) : '—', src: (c: Cell) => c.judge_score ?? 0 },
    { key: 'tokens_per_second', label: 'Tok/s', fmt: (v: number) => v ? v.toFixed(1) : '—', src: (c: Cell) => c.tokens_per_second ?? 0 },
    { key: 'hallucination_count', label: 'Halluc', fmt: (v: number) => String(v), src: (c: Cell) => c.hallucination_count },
  ];

  return (
    <div className="view-content">
      <table className="heat-table">
        <thead>
          <tr><th rowSpan={2}>Suite</th>{metrics.map(m => <th key={m.key} colSpan={2}>{m.label}</th>)}</tr>
          <tr>{metrics.flatMap(m => [<th key={m.key + 's'} className="subhead">s</th>, <th key={m.key + 'o'} className="subhead">o</th>])}</tr>
        </thead>
        <tbody>
          {suites.map(([suite, arr]) => {
            const stock = arr.filter(c => c.variant === 'stock');
            const ours = arr.filter(c => c.variant === 'ours');
            return (
              <tr key={suite}>
                <td><b>{suite}</b><br /><span className="faint">{arr.length} cells</span></td>
                {metrics.map(m => {
                  const sMean = mean(stock.map(m.src).filter(v => v != null && v > 0));
                  const oMean = mean(ours.map(m.src).filter(v => v != null && v > 0));
                  const maxV = Math.max(sMean, oMean, 1);
                  return (
                    <React.Fragment key={m.key}>
                      <td className="hc stock"><div className="hc-bg" style={{ background: `rgba(248,113,113,${0.05 + (sMean / maxV) * 0.3})` }} /><span>{m.fmt(sMean)}</span></td>
                      <td className="hc ours"><div className="hc-bg" style={{ background: `rgba(96,165,250,${0.05 + (oMean / maxV) * 0.3})` }} /><span>{m.fmt(oMean)}</span></td>
                    </React.Fragment>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
