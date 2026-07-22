import React, { useMemo } from 'react';
import { Cell } from '../types';

interface Props {
  cells: Cell[];
  onSelect?: (c: Cell | null) => void;
}

function perTask(cells: Cell[]): Map<string, { stock?: Cell; ours?: Cell }> {
  const m = new Map<string, { stock?: Cell; ours?: Cell }>();
  for (const c of cells) {
    if (!m.has(c.task_id)) m.set(c.task_id, {});
    const entry = m.get(c.task_id)!;
    entry[c.variant as 'stock' | 'ours'] = c;
  }
  return m;
}

export default function Comparison({ cells, onSelect }: Props) {
  const { scatterDots, wins, losses, histBins, histMax } = useMemo(() => {
    const pts: Array<{ task: string; sWall: number; oWall: number; sOk: boolean; oOk: boolean }> = [];
    const pairs = perTask(cells);
    for (const [task, v] of pairs) {
      if (v.stock && v.ours) {
        pts.push({ task, sWall: v.stock.wall_clock_s, oWall: v.ours.wall_clock_s, sOk: v.stock.ok, oOk: v.ours.ok });
      }
    }
    const w = pts.filter(p => p.oWall < p.sWall).length;
    const l = pts.length - w - pts.filter(p => p.oWall === p.sWall).length;
    const xMax = Math.max(0.1, ...pts.flatMap(p => [p.sWall, p.oWall]));
    const bins = 24;
    const binW = xMax / bins;
    const sBins = new Array(bins).fill(0);
    const oBins = new Array(bins).fill(0);
    for (const p of pts) {
      sBins[Math.min(bins - 1, Math.floor(p.sWall / binW))]++;
      oBins[Math.min(bins - 1, Math.floor(p.oWall / binW))]++;
    }
    return {
      scatterDots: pts,
      wins: w,
      losses: l,
      histBins: { bins, sBins, oBins, binW, xMax },
      histMax: Math.max(...sBins, ...oBins, 1),
    };
  }, [cells]);

  // Scatter SVG
  const scW = 400, scH = 200, scM = { l: 36, r: 8, t: 10, b: 22 };
  const xMax = Math.max(0.1, ...scatterDots.flatMap(p => [p.sWall, p.oWall]));
  const sx = (v: number) => scM.l + (v / xMax) * (scW - scM.l - scM.r);
  const sy = (v: number) => (scH - scM.b) - (v / xMax) * (scH - scM.t - scM.b);

  // Histogram SVG
  const hW = 400, hH = 120, hM = { l: 28, r: 8, t: 10, b: 20 };

  return (
    <div className="view-content">
      <div className="comp-layout">
        {/* Scatter */}
        <div className="comp-card">
          <div className="comp-card-title">Stock vs Ours Wall Time</div>
          <svg viewBox={`0 0 ${scW} ${scH}`} className="comp-svg">
            {[0, 1, 2, 3, 4].map(i => {
              const x = scM.l + (i / 4) * (scW - scM.l - scM.r);
              const y = (scH - scM.b) - (i / 4) * (scH - scM.t - scM.b);
              return (
                <React.Fragment key={i}>
                  <line className="sg-grid" x1={x} y1={scM.t} x2={x} y2={scH - scM.b} />
                  <line className="sg-grid" x1={scM.l} y1={y} x2={scW - scM.r} y2={y} />
                </React.Fragment>
              );
            })}
            <rect className="sg-quad win" x={scM.l} y={(scH - scM.b) / 2} width={(scW - scM.l - scM.r) / 2} height={(scH - scM.t - scM.b) / 2} />
            <rect className="sg-quad lose" x={(scM.l + scW + scM.r) / 2} y={scM.t} width={(scW - scM.l - scM.r) / 2} height={(scH - scM.t - scM.b) / 2} />
            <line className="sg-axis" x1={scM.l} y1={scH - scM.b} x2={scW - scM.r} y2={scH - scM.b} />
            <line className="sg-axis" x1={scM.l} y1={scM.t} x2={scM.l} y2={scH - scM.b} />
            {scatterDots.map((p, i) => {
              const cx = sx(p.sWall), cy = sy(p.oWall);
              const color = p.oWall < p.sWall ? 'var(--green)' : (p.oWall > p.sWall ? 'var(--red)' : 'var(--text-faint)');
              return (
                <circle key={i} className="sg-dot" cx={cx} cy={cy} r={2.5} fill={color} onClick={() => { const found = cells.find(c => c.task_id === p.task && c.variant === 'ours'); onSelect?.(found || null); }}>
                  <title>{p.task}: stock {p.sWall.toFixed(1)}s · ours {p.oWall.toFixed(1)}s</title>
                </circle>
              );
            })}
            <text className="sg-label" x={scW / 2} y={scH - 4} textAnchor="middle" fontSize={9}>stock wall (s)</text>
            <text className="sg-label" x={scM.l - 4} y={scH / 2} textAnchor="end" transform={`rotate(-90 ${scM.l - 4} ${scH / 2})`} fontSize={9}>ours wall (s)</text>
          </svg>
          <div className="comp-stats">
            <span className="cs-stat">tasks <strong>{scatterDots.length}</strong></span>
            <span className="cs-stat">wins <strong style={{ color: 'var(--green)' }}>{wins}</strong></span>
            <span className="cs-stat">losses <strong style={{ color: 'var(--red)' }}>{losses}</strong></span>
          </div>
        </div>

        {/* Histogram */}
        <div className="comp-card">
          <div className="comp-card-title">Wall Time Distribution</div>
          <svg viewBox={`0 0 ${hW} ${hH}`} className="comp-svg">
            <line className="sg-axis" x1={hM.l} y1={hH - hM.b} x2={hW - hM.r} y2={hH - hM.b} />
            <line className="sg-axis" x1={hM.l} y1={hM.t} x2={hM.l} y2={hH - hM.b} />
            {histBins.sBins.map((sh: number, i: number) => {
              const x = hM.l + (i / histBins.bins) * (hW - hM.l - hM.r);
              const w = (hW - hM.l - hM.r) / histBins.bins - 1;
              const oh = (histBins.oBins[i] / histMax) * (hH - hM.t - hM.b);
              return (
                <React.Fragment key={i}>
                  <rect className="sg-hbar stock" x={x} y={hH - hM.b - (sh / histMax) * (hH - hM.t - hM.b)} width={w / 2} height={(sh / histMax) * (hH - hM.t - hM.b)} />
                  <rect className="sg-hbar ours" x={x + w / 2 + 1} y={hH - hM.b - oh} width={w / 2} height={oh} />
                </React.Fragment>
              );
            })}
          </svg>
        </div>

        {/* Diff */}
        <div className="comp-card">
          <div className="comp-card-title">Win / Loss</div>
          <div className="wl-grid">
            <div className="wl-cell win"><div className="wl-num">{wins}</div><div className="wl-lbl">ours faster</div></div>
            <div className="wl-cell lose"><div className="wl-num">{losses}</div><div className="wl-lbl">ours slower</div></div>
            <div className="wl-cell tie"><div className="wl-num">{scatterDots.length - wins - losses}</div><div className="wl-lbl">tie</div></div>
            <div className="wl-cell delta">
              <div className="wl-num">{
                (() => {
                  const sMean = scatterDots.length ? scatterDots.reduce((a, p) => a + p.sWall, 0) / scatterDots.length : 0;
                  const oMean = scatterDots.length ? scatterDots.reduce((a, p) => a + p.oWall, 0) / scatterDots.length : 0;
                  return scatterDots.length ? ((oMean - sMean) / sMean * 100).toFixed(1) + '%' : '—';
                })()
              }</div>
              <div className="wl-lbl">mean Δ wall</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
