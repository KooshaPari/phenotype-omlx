import React from 'react';
import { Cell, Summary } from '../types';

interface Props {
  cells: Cell[];
  summary: Summary;
  onJumpToSuite: (suite: string) => void;
}

function mean(a: number[]): number {
  return a.length ? a.reduce((s, x) => s + x, 0) / a.length : 0;
}

function perSuite(cells: Cell[]): Map<string, { stock: Cell[]; ours: Cell[] }> {
  const m = new Map<string, { stock: Cell[]; ours: Cell[] }>();
  for (const c of cells) {
    if (!m.has(c.suite)) m.set(c.suite, { stock: [], ours: [] });
    m.get(c.suite)![c.variant as 'stock' | 'ours'].push(c);
  }
  return m;
}

function perDiff(cells: Cell[]): Map<string, { stock: Cell[]; ours: Cell[] }> {
  const m = new Map<string, { stock: Cell[]; ours: Cell[] }>();
  for (const c of cells) {
    if (!m.has(c.difficulty)) m.set(c.difficulty, { stock: [], ours: [] });
    m.get(c.difficulty)![c.variant as 'stock' | 'ours'].push(c);
  }
  return m;
}

export default function Overview({ cells, summary, onJumpToSuite }: Props) {
  const s = summary.by_variant.stock;
  const o = summary.by_variant.ours;
  const suites = perSuite(cells);
  const dims = perDiff(cells);
  const DIFFS = ['easy', 'medium', 'hard', 'ultra'];

  const metrics = [
    { key: 'pass_at_1', label: 'Pass@1', fmt: (v: number) => (v * 100).toFixed(1) + '%', better: 'up' },
    { key: 'mean_wall_clock_s', label: 'Wall', fmt: (v: number) => v.toFixed(2) + 's', better: 'down' },
    { key: 'mean_partial_credit', label: 'PC', fmt: (v: number) => v.toFixed(3), better: 'up' },
    { key: 'mean_format_compliance', label: 'Format', fmt: (v: number) => (v * 100).toFixed(0) + '%', better: 'up' },
  ];

  return (
    <div className="view-content">
      {/* Metric Cards */}
      <div className="ov-grid" id="ovGrid">
        {metrics.map(m => {
          const sv = s[m.key] ?? 0;
          const ov = o[m.key] ?? 0;
          const delta = ov - sv;
          const isBetter = (m.better === 'up' ? delta > 0 : delta < 0);
          const cls = Math.abs(delta) > 0.0001 ? (isBetter ? 'positive' : 'negative') : 'neutral';
          const arrow = Math.abs(delta) < 0.0001 ? '—' : (isBetter ? '▲ better' : '▼ worse');
          return (
            <div key={m.key} className="ov-card">
              <div className="ov-title">{m.label}</div>
              <div className="ov-metric">
                <span className="ov-stock">{m.fmt(sv)}</span>
                <span className="ov-sep">·</span>
                <span className="ov-ours">{m.fmt(ov)}</span>
              </div>
              <div className="ov-bar">
                <div className="seg stock" style={{ width: '50%' }} />
                <div className="seg ours" style={{ width: '50%' }} />
              </div>
              <div className="ov-meta">
                <span>Δ {delta >= 0 ? '+' : ''}{m.fmt(delta)}</span>
                <span className={cls}>{arrow}</span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Suite Grid */}
      <h3 className="section-title">Per Suite</h3>
      <div className="suite-grid" id="suiteGrid">
        {[...suites.entries()].sort((a, b) => a[0].localeCompare(b[0])).map(([suite, grp]) => {
          const sW = mean(grp.stock.map(c => c.wall_clock_s));
          const oW = mean(grp.ours.map(c => c.wall_clock_s));
          const sP = mean(grp.stock.map(c => c.pass_at_1));
          const oP = mean(grp.ours.map(c => c.pass_at_1));
          const dWall = oW - sW;
          const cls = dWall < 0 ? 'positive' : (dWall > 0 ? 'negative' : 'neutral');
          return (
            <div key={suite} className="suite-card" onClick={() => onJumpToSuite(suite)}>
              <div className="suite-name">{suite} <span className="n">{grp.stock.length + grp.ours.length}</span></div>
              <div className="suite-row"><span className="swatch stock" />stock <span className="suite-val">{sW.toFixed(2)}s</span></div>
              <div className="suite-row"><span className="swatch ours" />ours <span className="suite-val">{oW.toFixed(2)}s</span><span className={`suite-delta ${cls}`}>{dWall >= 0 ? '+' : ''}{dWall.toFixed(2)}s</span></div>
              <div className="suite-row"><span className="swatch" style={{ background: 'var(--green)' }} />p@1 <span className="suite-val">{(sP * 100).toFixed(0)}% / {(oP * 100).toFixed(0)}%</span></div>
            </div>
          );
        })}
      </div>

      {/* Per Difficulty */}
      <h3 className="section-title">Per Difficulty</h3>
      <div className="dim-grid" id="dimGrid">
        <div className="dim-card">
          {DIFFS.map(d => {
            const grp = dims.get(d);
            if (!grp || (!grp.stock.length && !grp.ours.length)) return null;
            const sW = mean(grp.stock.map(c => c.wall_clock_s));
            const oW = mean(grp.ours.map(c => c.wall_clock_s));
            const sP = mean(grp.stock.map(c => c.pass_at_1));
            const oP = mean(grp.ours.map(c => c.pass_at_1));
            const maxW = Math.max(sW, oW, 1);
            return (
              <div key={d}>
                <div className="dim-title">{d}</div>
                <div className="dim-row"><span className="label">Wall</span><div className="bar-wrap"><div className="bar stock" style={{ left: 0, width: `${(sW / maxW) * 100}%` }} /><div className="bar ours" style={{ left: 0, width: `${(oW / maxW) * 100}%` }} /></div><span className="val">{sW.toFixed(1)} / {oW.toFixed(1)}s</span></div>
                <div className="dim-row"><span className="label">P@1</span><div className="bar-wrap"><div className="bar stock" style={{ left: 0, width: `${sP * 100}%` }} /><div className="bar ours" style={{ left: 0, width: `${oP * 100}%` }} /></div><span className="val">{(sP * 100).toFixed(0)} / {(oP * 100).toFixed(0)}%</span></div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
