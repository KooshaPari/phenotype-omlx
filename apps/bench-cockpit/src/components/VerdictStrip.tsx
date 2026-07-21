import React from 'react';
import {
  summaryHasVerified,
  summaryQualityLabel,
  summaryQualityPass,
} from '../lib/metrics';
import { VariantSummary } from '../types';

interface VerdictStripProps {
  summary: { stock: Record<string, number>; ours: Record<string, number> };
  statusText?: string;
  statusLevel?: string;
  /** When true, Pass@1 is generation-ok / reported — demote label and order. */
  passAt1Untrusted?: boolean;
}

type MetricDef = {
  key: string;
  label: string;
  fmt: (v: number) => string;
  better: 'up' | 'down';
  value: (side: Record<string, number>) => number;
};

const BASE_METRICS: MetricDef[] = [
  { key: 'mean_partial_credit', label: 'PC', fmt: (v) => v.toFixed(3), better: 'up', value: (s) => s.mean_partial_credit ?? 0 },
  { key: 'mean_wall_clock_s', label: 'Wall', fmt: (v) => v.toFixed(2) + 's', better: 'down', value: (s) => s.mean_wall_clock_s ?? 0 },
  {
    key: 'mean_tokens_per_second',
    label: 'Tok/s',
    fmt: (v) => (v ? v.toFixed(1) : '—'),
    better: 'up',
    value: (s) => s.mean_tokens_per_second || s.mean_decode_speed_tps || s.mean_tokens_read || 0,
  },
  {
    key: 'mean_format_compliance',
    label: 'Fmt',
    fmt: (v) => (v * 100).toFixed(0) + '%',
    better: 'up',
    value: (s) => s.mean_format_compliance ?? 0,
  },
  { key: 'n_hallucinations', label: 'Halluc', fmt: (v) => String(v), better: 'down', value: (s) => s.n_hallucinations ?? 0 },
];

function passMetric(untrusted: boolean, stock: Record<string, number>, ours: Record<string, number>): MetricDef {
  const stockV = stock as VariantSummary;
  const oursV = ours as VariantSummary;
  const hasVerified = summaryHasVerified(stockV) || summaryHasVerified(oursV);
  const label = hasVerified ? 'Verified' : summaryQualityLabel(stockV, untrusted);
  return {
    key: hasVerified ? 'verified_pass_at_1' : 'pass_at_1',
    label,
    fmt: (v) => (v * 100).toFixed(1) + '%',
    better: 'up',
    value: (s) => summaryQualityPass(s as VariantSummary),
  };
}

export default function VerdictStrip({
  summary,
  statusText,
  statusLevel,
  passAt1Untrusted = false,
}: VerdictStripProps) {
  const s = summary.stock;
  const o = summary.ours;
  const metrics = [...BASE_METRICS, passMetric(passAt1Untrusted, s, o)];
  const passKey = metrics[metrics.length - 1].key;
  const passUntrusted = passAt1Untrusted && passKey === 'pass_at_1';

  return (
    <div className={`verdict-strip ${passUntrusted ? 'p1-untrusted' : ''}`}>
      {passUntrusted && (
        <div className="vs-banner faint">Pass@1 = gen ok on this run — prefer PC / Wall / Tok/s</div>
      )}
      {metrics.map((m) => {
        const sv = m.value(s);
        const ov = m.value(o);
        const delta = ov - sv;
        const isBetter = m.better === 'up' ? delta > 0 : delta < 0;
        const cls = Math.abs(delta) > 0.0001 ? (isBetter ? 'positive' : 'negative') : 'neutral';
        const denom = Math.max(sv, ov, 1);
        const pctKeys = m.key === 'pass_at_1' || m.key === 'verified_pass_at_1' || m.key === 'mean_format_compliance';
        return (
          <div
            key={m.key}
            className={`vc-pair ${(m.key === 'pass_at_1' || m.key === 'verified_pass_at_1') && passUntrusted ? 'p1-reported' : ''}`}
          >
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
              Δ {delta >= 0 ? '+' : ''}
              {pctKeys ? (delta * 100).toFixed(1) + 'pp' : delta.toFixed(2)}
            </div>
          </div>
        );
      })}
      {statusText && <div className={`vs-status ${statusLevel || ''}`}>{statusText}</div>}
    </div>
  );
}
