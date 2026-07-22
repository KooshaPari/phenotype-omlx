import React, { useMemo, useState } from 'react';
import { Cell } from '../types';

interface Props {
  cells: Cell[];
}

function percentile(sorted: number[], p: number): number {
  if (!sorted.length) return 0;
  const i = (sorted.length - 1) * p;
  const lo = Math.floor(i);
  const hi = Math.ceil(i);
  if (lo === hi) return sorted[lo];
  return sorted[lo] * (1 - (i - lo)) + sorted[hi] * (i - lo);
}

/** Stock vs ours throughput widgets — make the gap visible; goal is ours dominates. */
export default function Throughput({ cells }: Props) {
  const [cache, setCache] = useState(0.5);
  const [inTok, setInTok] = useState(50_000);
  const [outTok, setOutTok] = useState(500);

  const byVariant = useMemo(() => {
    const map = new Map<string, Cell[]>();
    for (const c of cells) {
      const arr = map.get(c.variant) || [];
      arr.push(c);
      map.set(c.variant, arr);
    }
    return map;
  }, [cells]);

  const simRpm = (tps: number) => {
    const denom = inTok * (1 - cache) + outTok;
    if (denom <= 0 || tps <= 0) return 0;
    return (tps * 60) / denom;
  };

  const agg = (variant: string) => {
    const cs = byVariant.get(variant) || [];
    const tps = cs.map((c) => c.tokens_per_second).filter((x) => x > 0).sort((a, b) => a - b);
    const wall = cs.map((c) => c.wall_clock_s).filter((x) => x > 0).sort((a, b) => a - b);
    const ttft = cs.map((c) => c.first_token_latency_ms).filter((x) => x > 0).sort((a, b) => a - b);
    const meanTps = tps.length ? tps.reduce((a, b) => a + b, 0) / tps.length : 0;
    const meanPass = cs.length
      ? cs.reduce((a, c) => a + c.pass_at_1, 0) / cs.length
      : 0;
    return { meanTps, meanPass, wall, ttft, n: cs.length, rpm: simRpm(meanTps) };
  };

  const stock = agg('stock');
  const ours = agg('ours');
  const oursWinsTps = ours.meanTps > stock.meanTps;
  const oursWinsRpm = ours.rpm > stock.rpm;

  const suites = [...new Set(cells.map((c) => c.suite))].sort();

  return (
    <div className="view-stack" data-testid="throughput-view">
      <div className={`viz-panel ${oursWinsTps ? 'good-banner' : 'warn-banner'}`}>
        <div className="viz-toolbar">
          <span className="viz-title">Throughput · stock vs ours</span>
          <span className="viz-hint">
            goal: ours dominates every dimension · currently{' '}
            {oursWinsTps ? 'ours ahead on tok/s' : 'stock ahead on tok/s'}
          </span>
        </div>
        <div className="thru-cards">
          {(['stock', 'ours'] as const).map((v) => {
            const a = v === 'stock' ? stock : ours;
            return (
              <div key={v} className={`thru-card ${v}`}>
                <h4>{v}</h4>
                <div className="kv"><span className="k">n</span><span className="v">{a.n}</span></div>
                <div className="kv"><span className="k">mean tok/s</span><span className="v">{a.meanTps.toFixed(2)}</span></div>
                <div className="kv"><span className="k">mean pass@1</span><span className="v">{(a.meanPass * 100).toFixed(1)}%</span></div>
                <div className="kv"><span className="k">sim RPM</span><span className="v">{a.rpm.toFixed(3)}</span></div>
                <div className="kv"><span className="k">wall p50</span><span className="v">{percentile(a.wall, 0.5).toFixed(2)}s</span></div>
                <div className="kv"><span className="k">wall p95</span><span className="v">{percentile(a.wall, 0.95).toFixed(2)}s</span></div>
                <div className="kv"><span className="k">wall p99</span><span className="v">{percentile(a.wall, 0.99).toFixed(2)}s</span></div>
                <div className="kv"><span className="k">TTFT p50</span><span className="v">{percentile(a.ttft, 0.5).toFixed(0)}ms</span></div>
                <div className="kv"><span className="k">TTFT p95</span><span className="v">{percentile(a.ttft, 0.95).toFixed(0)}ms</span></div>
              </div>
            );
          })}
        </div>
      </div>

      <div className="viz-panel">
        <div className="viz-toolbar">
          <span className="viz-title">Cache-hit RPM simulator</span>
        </div>
        <p className="formula">
          sim_rpm = (tokens_per_second × 60) / (prompt_tokens × (1 − cache) + output_tokens)
        </p>
        <div className="thru-controls">
          <label>
            cache {(cache * 100).toFixed(0)}%
            <input
              type="range"
              min={0}
              max={100}
              step={25}
              value={cache * 100}
              onChange={(e) => setCache(Number(e.target.value) / 100)}
            />
          </label>
          <label>
            tokens in
            <input type="number" value={inTok} onChange={(e) => setInTok(Number(e.target.value) || 0)} />
          </label>
          <label>
            tokens out
            <input type="number" value={outTok} onChange={(e) => setOutTok(Number(e.target.value) || 0)} />
          </label>
        </div>
        <div className="thru-cards">
          <div className="thru-card stock">stock RPM @ cache: {stock.rpm.toFixed(4)}</div>
          <div className={`thru-card ours ${oursWinsRpm ? 'good' : 'warn'}`}>
            ours RPM @ cache: {ours.rpm.toFixed(4)} {oursWinsRpm ? '✓' : '(behind)'}
          </div>
        </div>
      </div>

      <div className="viz-panel">
        <div className="viz-toolbar">
          <span className="viz-title">Per-suite tok/s · pass@1</span>
        </div>
        <table className="heat-table">
          <thead>
            <tr>
              <th>suite</th>
              <th>stock tok/s</th>
              <th>ours tok/s</th>
              <th>Δ tok/s</th>
              <th>stock p@1</th>
              <th>ours p@1</th>
              <th>GPU peak</th>
            </tr>
          </thead>
          <tbody>
            {suites.map((s) => {
              const sc = cells.filter((c) => c.suite === s && c.variant === 'stock');
              const oc = cells.filter((c) => c.suite === s && c.variant === 'ours');
              const st = sc.length ? sc.reduce((a, c) => a + c.tokens_per_second, 0) / sc.length : 0;
              const ot = oc.length ? oc.reduce((a, c) => a + c.tokens_per_second, 0) / oc.length : 0;
              const sp = sc.length ? sc.reduce((a, c) => a + c.pass_at_1, 0) / sc.length : 0;
              const op = oc.length ? oc.reduce((a, c) => a + c.pass_at_1, 0) / oc.length : 0;
              const gpu = Math.max(0, ...cells.filter((c) => c.suite === s).map((c) => c.peak_gpu_mem_mb || 0));
              return (
                <tr key={s}>
                  <td>{s}</td>
                  <td>{st.toFixed(2)}</td>
                  <td className={ot > st ? 'good' : 'bad'}>{ot.toFixed(2)}</td>
                  <td className={ot - st >= 0 ? 'good' : 'bad'}>{(ot - st).toFixed(2)}</td>
                  <td>{(sp * 100).toFixed(0)}%</td>
                  <td>{(op * 100).toFixed(0)}%</td>
                  <td className={gpu > 20000 ? 'bad' : ''}>{gpu ? gpu.toFixed(0) : '—'}{gpu > 20000 ? ' ⚠' : ''}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
