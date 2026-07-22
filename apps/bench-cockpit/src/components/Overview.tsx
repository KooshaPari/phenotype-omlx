import React from 'react';
import { Cell, Summary, SuiteCoverageRow } from '../types';
import { meanQualityPass, summaryQualityLabel, summaryQualityPass } from '../lib/metrics';
import { auxRoleLabel, auxVariants, isAblationVariant } from '../lib/arms';
import { SuiteCoverage } from './SuiteCoverage';
import { EMPTY_VARIANT_SUMMARY } from './VerdictStrip';

interface Props {
  cells: Cell[];
  summary: Summary;
  onJumpToSuite: (suite: string) => void;
  suiteCoverage?: SuiteCoverageRow[];
}

function mean(a: number[]): number {
  return a.length ? a.reduce((s, x) => s + x, 0) / a.length : 0;
}

function perSuite(cells: Cell[]): Map<string, { stock: Cell[]; ours: Cell[] }> {
  const m = new Map<string, { stock: Cell[]; ours: Cell[] }>();
  for (const c of cells) {
    if (c.variant !== 'stock' && c.variant !== 'ours') continue;
    if (!m.has(c.suite)) m.set(c.suite, { stock: [], ours: [] });
    m.get(c.suite)![c.variant].push(c);
  }
  return m;
}

function perDiff(cells: Cell[]): Map<string, { stock: Cell[]; ours: Cell[] }> {
  const m = new Map<string, { stock: Cell[]; ours: Cell[] }>();
  for (const c of cells) {
    if (c.variant !== 'stock' && c.variant !== 'ours') continue;
    if (!m.has(c.difficulty)) m.set(c.difficulty, { stock: [], ours: [] });
    m.get(c.difficulty)![c.variant].push(c);
  }
  return m;
}

export default function Overview({ cells, summary, onJumpToSuite, suiteCoverage }: Props) {
  const s = summary.by_variant.stock ?? EMPTY_VARIANT_SUMMARY;
  const o = summary.by_variant.ours ?? EMPTY_VARIANT_SUMMARY;
  const extraArms = auxVariants(Object.keys(summary.by_variant || {}));
  const suites = perSuite(cells);
  const dims = perDiff(cells);
  const DIFFS = ['easy', 'medium', 'hard', 'ultra'];
  const passLabel = summaryQualityLabel(s, true);

  const metrics = [
    { key: 'mean_partial_credit', label: 'PC', fmt: (v: number) => v.toFixed(3), better: 'up' as const, stock: s.mean_partial_credit, ours: o.mean_partial_credit },
    { key: 'mean_wall_clock_s', label: 'Wall', fmt: (v: number) => v.toFixed(2) + 's', better: 'down' as const, stock: s.mean_wall_clock_s, ours: o.mean_wall_clock_s },
    { key: 'mean_format_compliance', label: 'Format', fmt: (v: number) => (v * 100).toFixed(0) + '%', better: 'up' as const, stock: s.mean_format_compliance, ours: o.mean_format_compliance },
    {
      key: 'quality_pass',
      label: passLabel,
      fmt: (v: number) => (v * 100).toFixed(1) + '%',
      better: 'up' as const,
      stock: summaryQualityPass(s),
      ours: summaryQualityPass(o),
    },
  ];

  return (
    <div className="view-content">
      <div className="ov-grid" id="ovGrid">
        {metrics.map((m) => {
          const sv = m.stock ?? 0;
          const ov = m.ours ?? 0;
          const delta = ov - sv;
          const isBetter = m.better === 'up' ? delta > 0 : delta < 0;
          const cls = Math.abs(delta) > 0.0001 ? (isBetter ? 'positive' : 'negative') : 'neutral';
          const arrow = Math.abs(delta) < 0.0001 ? '—' : isBetter ? '▲ better' : '▼ worse';
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

      {extraArms.length > 0 && (
        <p className="muted" style={{ marginBottom: 12 }} data-testid="aux-roles-banner">
          Auxiliary roles in load (judge / evaluator / distiller — not peer models):{' '}
          <code className="mono">{extraArms.map(auxRoleLabel).join(', ')}</code>
        </p>
      )}

      <SuiteCoverage rows={suiteCoverage ?? []} onJumpToSuite={onJumpToSuite} />

      <h3 className="section-title">Per Suite</h3>
      <div className="suite-grid" id="suiteGrid">
        {[...suites.entries()].sort((a, b) => a[0].localeCompare(b[0])).map(([suite, grp]) => {
          const sW = mean(grp.stock.map((c) => c.wall_clock_s));
          const oW = mean(grp.ours.map((c) => c.wall_clock_s));
          const sP = meanQualityPass(grp.stock);
          const oP = meanQualityPass(grp.ours);
          const dWall = oW - sW;
          const cls = dWall < 0 ? 'positive' : dWall > 0 ? 'negative' : 'neutral';
          const suiteExtras = cells.filter(
            (c) => c.suite === suite && !isAblationVariant(c.variant),
          );
          const byArm = new Map<string, Cell[]>();
          for (const c of suiteExtras) {
            const arr = byArm.get(c.variant) || [];
            arr.push(c);
            byArm.set(c.variant, arr);
          }
          return (
            <div key={suite} className="suite-card" onClick={() => onJumpToSuite(suite)}>
              <div className="suite-name">{suite} <span className="n">{grp.stock.length + grp.ours.length}</span></div>
              {grp.stock.length > 0 && (
                <div className="suite-card-row"><span className="swatch stock" />stock <span className="suite-val">{sW.toFixed(2)}s · {(sP * 100).toFixed(0)}%</span></div>
              )}
              {grp.ours.length > 0 && (
                <div className="suite-card-row"><span className="swatch ours" />ours <span className="suite-val">{oW.toFixed(2)}s · {(oP * 100).toFixed(0)}%</span><span className={`suite-delta ${cls}`}>{dWall >= 0 ? '+' : ''}{dWall.toFixed(2)}s</span></div>
              )}
              {[...byArm.entries()].map(([arm, armCells]) => (
                <div key={arm} className="suite-card-row suite-aux-row faint">
                  <span className="swatch" style={{ background: 'var(--muted, #888)', opacity: 0.5 }} />
                  {auxRoleLabel(arm)}{' '}
                  <span className="suite-val">
                    {armCells.length} cell{armCells.length === 1 ? '' : 's'} · aux
                  </span>
                </div>
              ))}
              {!grp.stock.length && !grp.ours.length && suiteExtras.length === 0 && (
                <div className="suite-card-row faint">no cells</div>
              )}
            </div>
          );
        })}
      </div>

      <h3 className="section-title">Per Difficulty</h3>
      <div className="dim-grid" id="dimGrid">
        <div className="dim-card">
          {DIFFS.map((d) => {
            const grp = dims.get(d);
            if (!grp || (!grp.stock.length && !grp.ours.length)) return null;
            const sW = mean(grp.stock.map((c) => c.wall_clock_s));
            const oW = mean(grp.ours.map((c) => c.wall_clock_s));
            const sP = meanQualityPass(grp.stock);
            const oP = meanQualityPass(grp.ours);
            const maxW = Math.max(sW, oW, 1);
            return (
              <div key={d}>
                <div className="dim-title">{d}</div>
                <div className="dim-row"><span className="label">Wall</span><div className="bar-wrap"><div className="bar stock" style={{ left: 0, width: `${(sW / maxW) * 100}%` }} /><div className="bar ours" style={{ left: 0, width: `${(oW / maxW) * 100}%` }} /></div><span className="val">{sW.toFixed(1)} / {oW.toFixed(1)}s</span></div>
                <div className="dim-row"><span className="label">Pass</span><div className="bar-wrap"><div className="bar stock" style={{ left: 0, width: `${sP * 100}%` }} /><div className="bar ours" style={{ left: 0, width: `${oP * 100}%` }} /></div><span className="val">{(sP * 100).toFixed(0)} / {(oP * 100).toFixed(0)}%</span></div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
