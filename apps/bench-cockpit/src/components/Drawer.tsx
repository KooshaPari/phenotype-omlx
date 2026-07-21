import React, { useState } from 'react';
import { Cell } from '../types';
import TraceView from './TraceView';

interface Props {
  cell: Cell | null;
  paired: Cell | null;
  onClose: () => void;
  onAudit?: (c: Cell) => void;
}

export default function Drawer({ cell, paired, onClose, onAudit }: Props) {
  const [showTrace, setShowTrace] = useState(false);
  if (!cell) return null;

  const section = (title: string, rows: [string, React.ReactNode, string?][]) => (
    <div className="ds">
      <h5>{title}</h5>
      {rows.map(([k, v, cls]) => (
        <div className="kv" key={k}>
          <span className="k">{k}</span>
          <span className={`v${cls ? ' ' + cls : ''}`}>{v}</span>
        </div>
      ))}
    </div>
  );

  const compareBlock = (c: Cell, label: string) => (
    <div className="cmp-side">
      <div className="cmp-title">{label} · {c.variant}</div>
      {section('Performance', [
        ['Wall', `${c.wall_clock_s.toFixed(2)}s`, c.wall_clock_s > 30 ? 'bad' : c.wall_clock_s > 10 ? 'warn' : 'good'],
        ['Tok/s', c.tokens_per_second ? c.tokens_per_second.toFixed(1) : '—', (c.tokens_per_second ?? 0) < 5 ? 'bad' : 'good'],
        ['TTFT', c.first_token_latency_ms ? `${(c.first_token_latency_ms/1000).toFixed(2)}s` : '—',
          c.first_token_latency_ms && c.first_token_latency_ms > 5000 ? 'bad' : c.first_token_latency_ms && c.first_token_latency_ms > 1000 ? 'warn' : 'good'],
        ['RSS', c.peak_rss_mb ? `${c.peak_rss_mb.toFixed(0)}M` : '—'],
        ['Joules', c.energy_proxy_joules ? `${c.energy_proxy_joules.toFixed(1)}` : '—', c.energy_proxy_joules && c.energy_proxy_joules > 100 ? 'bad' : 'good'],
      ])}
      {section('Quality', [
        ['P@1', `${(c.pass_at_1 * 100).toFixed(0)}%`, c.pass_at_1 >= 0.8 ? 'good' : c.pass_at_1 >= 0.5 ? 'warn' : 'bad'],
        ['PC', c.partial_credit.toFixed(3), c.partial_credit >= 0.8 ? 'good' : c.partial_credit >= 0.5 ? 'warn' : 'bad'],
        ['Fmt', `${(c.format_compliance_rate * 100).toFixed(0)}%`, c.format_compliance_rate >= 0.9 ? 'good' : 'warn'],
        ['Halluc', c.hallucination_count, c.hallucination_count === 0 ? 'good' : 'bad'],
        ['Judge', c.judge_score?.toFixed(2) ?? '—', c.judge_score && c.judge_score >= 0.8 ? 'good' : 'warn'],
      ])}
    </div>
  );

  return (
    <div className={`drawer ${cell ? 'open' : ''}`}>
      <div className="drawer-h">
        <span className="drawer-title">{cell.task_id} · {cell.variant}</span>
        <div className="drawer-actions">
          <button className="drawer-btn" onClick={() => setShowTrace((v) => !v)}>
            {showTrace ? 'Hide trace' : 'Trace'}
          </button>
          {onAudit && (
            <button className="drawer-btn" onClick={() => onAudit(cell)}>
              View raw
            </button>
          )}
          <button className="drawer-close" onClick={onClose}>✕</button>
        </div>
      </div>
      {paired && (
        <div className="drawer-compare">
          {compareBlock(paired, 'Paired')}
          <div className="cmp-vs">vs</div>
          {compareBlock(cell, 'Selected')}
        </div>
      )}
      <div className="drawer-body">
        {section('Metadata', [
          ['Suite', cell.suite],
          ['Difficulty', cell.difficulty],
          ['Model', cell.model_name || '—'],
          ['Temperature', String(cell.temperature ?? '—')],
          ['Seed', String(cell.seed ?? '—')],
          ['Created', cell.created_at?.slice(0, 19).replace('T', ' ') || '—'],
        ])}
        {section('Cost & Tokens', [
          ['Tokens in', String(cell.total_tokens_in)],
          ['Tokens out', String(cell.total_tokens_out)],
          ['Cost', cell.cost_usd ? `$${cell.cost_usd.toFixed(4)}` : '—'],
        ])}
        {cell.failure_analysis && Object.keys(cell.failure_analysis).length > 0 && (
          section('Failure Analysis', Object.entries(cell.failure_analysis).map(([k, v]) => [k, String(v)] as [string, string]))
        )}
        {cell.semantic && Object.keys(cell.semantic).length > 0 && (
          <div className="ds">
            <h5>Semantic</h5>
            {Object.entries(cell.semantic).map(([k, v]) => (
              <div className="sem-row" key={k}>
                <span className="l">{k}</span>
                <div className="bar-wrap">
                  <div className={`bar ${v < 0.5 ? 'low' : v < 0.8 ? 'mid' : 'high'}`} style={{ width: `${v * 100}%` }} />
                </div>
                <span className="v">{v.toFixed(2)}</span>
              </div>
            ))}
          </div>
        )}
        {cell.reply && <div className="ds"><h5>Reply</h5><pre className="reply-box">{cell.reply.slice(0, 2000)}</pre></div>}
        {cell.prompt && <div className="ds"><h5>Prompt</h5><pre className="reply-box">{cell.prompt.slice(0, 1000)}</pre></div>}
        {showTrace && <TraceView cell={cell} />}
      </div>
    </div>
  );
}
