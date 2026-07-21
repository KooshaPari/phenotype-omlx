import React from 'react';

interface VerdictStripProps {
  summary: { stock: Record<string, number>; ours: Record<string, number> };
  statusText?: string;
  statusLevel?: string;
}

const METRICS = [
  { key: 'pass_at_1',           label: 'Pass@1',     fmt: (v: number) => (v * 100).toFixed(1) + '%',   better: 'up' },
  { key: 'mean_wall_clock_s',   label: 'Wall',        fmt: (v: number) => v.toFixed(2) + 's',          better: 'down' },
  { key: 'mean_partial_credit', label: 'PC',          fmt: (v: number) => v.toFixed(3),                better: 'up' },
  { key: 'mean_format_compliance', label: 'Fmt',      fmt: (v: number) => (v * 100).toFixed(0) + '%',  better: 'up' },
  { key: 'n_hallucinations',    label: 'Halluc',      fmt: (v: number) => String(v),                   better: 'down' },
  { key: 'mean_tokens_per_second', label: 'Tok/s',    fmt: (v: number) => v ? v.toFixed(1) : '—',      better: 'up' },
];

const METRICS_WITH_TOTALS = [
  ...METRICS,
  { key: 'mean_first_token_latency', label: 'TTFT', fmt: (v: number) => v ? (v / 1000).toFixed(2) + 's' : '—', better: 'down' },
  { key: 'mean_cost_usd',       label: 'Cost',       fmt: (v: number) => v ? '$' + v.toFixed(4) : '—',      better: 'down' },
];

function metricValue(side: Record<string, number>, key: string): number {
  if (key === 'mean_tokens_per_second') {
    return side.mean_tokens_per_second || side.mean_decode_speed_tps || side.mean_tokens_read || 0;
  }
  return side[key] ?? 0;
}

export default function VerdictStrip({ summary, statusText, statusLevel }: VerdictStripProps) {
  const s = summary.stock, o = summary.ours;
  return (
    <div className="verdict-strip">
      {METRICS.map(m => {
        const sv = metricValue(s, m.key);
        const ov = metricValue(o, m.key);
        const delta = ov - sv;
        const isBetter = (m.better === 'up' ? delta > 0 : delta < 0);
        const cls = Math.abs(delta) > 0.0001 ? (isBetter ? 'positive' : 'negative') : 'neutral';
        const denom = Math.max(sv, ov, 1);
        return (
          <div key={m.key} className="vc-pair">
            <div className="vc-label">{m.label}</div>
            <div className="vc-bars">
              <div className="vc-bar-row">
                <span className="vc-bar stock" style={{ width: `${(sv / denom) * 100}%` }} />
                <span className="vc-val stock">{m.fmt(sv)}</span>
              </div>
              <div className="vc-bar-row">
                <span className="vc-bar ours" style={{ width: `${(ov / denom) * 100}%` }} />
                <span className="vc-val ours">{m.fmt(ov)}</span>
              </div>
            </div>
            <div className={`vc-delta ${cls}`}>
              Δ {delta >= 0 ? '+' : ''}{m.key === 'pass_at_1' || m.key === 'mean_format_compliance'
                ? (delta * 100).toFixed(1) + 'pp'
                : delta.toFixed(2)}
            </div>
          </div>
        );
      })}
      {statusText && <div className={`vs-status ${statusLevel || ''}`}>{statusText}</div>}
    </div>
  );
}
