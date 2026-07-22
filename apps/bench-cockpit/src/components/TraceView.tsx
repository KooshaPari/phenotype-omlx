import React, { useMemo, useState } from 'react';
import { Cell } from '../types';
import { decodeTrace, Span } from '../types/span';
import { isTraceTruncated, resolveTraceRows } from '../lib/assignment';

interface Props {
  cell: Cell;
  /** When true, hide duplicate prompt/reply panes (TaskPage shows them above). */
  chatOnly?: boolean;
}

function roleSide(role?: string): 'user' | 'assistant' | 'system' {
  const r = (role || '').toLowerCase();
  if (r === 'user' || r === 'human') return 'user';
  if (r === 'system' || r === 'tool' || r === 'function') return 'system';
  return 'assistant';
}

function ChatBubble({
  side,
  label,
  children,
  meta,
}: {
  side: 'user' | 'assistant' | 'system';
  label?: string;
  children: React.ReactNode;
  meta?: string;
}) {
  return (
    <div className={`imsg-row imsg-${side}`} data-testid={`imsg-${side}`}>
      {label && <div className="imsg-label">{label}</div>}
      <div className={`imsg-bubble imsg-bubble-${side}`}>{children}</div>
      {meta && <div className="imsg-meta">{meta}</div>}
    </div>
  );
}

/** Multi-turn chat/tool trace — iMessage-style bubbles + span rail. */
export default function TraceView({ cell, chatOnly = false }: Props) {
  const rows = useMemo(() => resolveTraceRows(cell), [cell]);
  const spans = useMemo(() => decodeTrace(rows), [rows]);
  const turns = spans.filter((s): s is Extract<Span, { kind: 'turn' }> => s.kind === 'turn');
  const tools = spans.filter((s): s is Extract<Span, { kind: 'tool' }> => s.kind === 'tool');
  const [active, setActive] = useState(0);
  const truncated = isTraceTruncated(cell, spans.length);
  const hasConversational =
    turns.some((t) => Boolean(t.content?.trim())) ||
    tools.length > 0 ||
    Boolean(cell.prompt?.trim() || cell.reply?.trim());

  return (
    <div className="trace-view" data-testid="trace-view">
      <div className="trace-rail">
        <h5>Turns</h5>
        {turns.length === 0 && (
          <div className="muted">
            {spans.length === 0
              ? 'No transcript spans'
              : `No turn spans — raw/tool only (${spans.length})`}
          </div>
        )}
        {turns.map((t, i) => (
          <button
            key={t.id}
            type="button"
            className={`trace-turn ${i === active ? 'active' : ''}`}
            onClick={() => setActive(i)}
          >
            #{t.turn} {t.role || ''}
          </button>
        ))}
        <h5>Spans</h5>
        <ul className="trace-span-list">
          {spans.map((s) => (
            <li key={s.id} className={`span-${s.kind}`}>
              <code>{s.kind}</code> {s.name || (s.kind === 'tool' ? s.toolName : '') || s.id}
              {s.kind === 'reward' && s.composite != null && (
                <span className="v"> composite={s.composite.toFixed(3)}</span>
              )}
              {s.kind === 'verifier' && (
                <span className="v">
                  {' '}
                  {s.passed ? 'pass' : 'fail'} r={s.reward?.toFixed?.(2) ?? '—'}
                </span>
              )}
              {s.kind === 'tool' && s.ok != null && (
                <span className="v"> {s.ok ? 'ok' : 'fail'}</span>
              )}
            </li>
          ))}
        </ul>
      </div>

      <div className="trace-main">
        {(truncated || (!hasConversational && spans.length === 0)) && (
          <div className="trace-empty" data-testid="trace-empty">
            <strong>Transcript unavailable</strong>
            <p>
              {truncated
                ? 'V5 / export data looks truncated — progress_trace or chat bodies were stripped. Re-run with full IO export or enrich harness dual-read fields.'
                : 'No chat or tool spans on this cell. Prompt/reply may still appear below when present.'}
            </p>
          </div>
        )}

        <div className="imsg-thread" data-testid="imsg-thread">
          {turns.map((t) => {
            const side = roleSide(t.role);
            return (
              <ChatBubble
                key={t.id}
                side={side}
                label={t.role || side}
                meta={t.startMs != null ? `${t.startMs}ms` : undefined}
              >
                {t.content?.trim() || (
                  <span className="muted">
                    (empty turn — body truncated)
                  </span>
                )}
              </ChatBubble>
            );
          })}
          {tools.map((t) => (
            <ChatBubble
              key={t.id}
              side="system"
              label={`tool · ${t.toolName || t.name || t.id}`}
              meta={t.ok == null ? undefined : t.ok ? 'ok' : 'fail'}
            >
              <code>{t.toolName || t.name || 'tool'}</code>
            </ChatBubble>
          ))}
          {turns.length === 0 && spans.length > 0 && !tools.length && (
            <div className="muted imsg-fallback">
              Spans decoded but no chat turns — open span list for reward/verifier/llm events.
            </div>
          )}
        </div>

        {turns[active]?.content && (
          <div className="ds">
            <h5>Active turn</h5>
            <pre className="reply-box">{turns[active].content}</pre>
          </div>
        )}

        {!chatOnly && (
          <>
            <div className="ds">
              <h5>Prompt</h5>
              <pre className="reply-box">{cell.prompt || '—'}</pre>
            </div>
            <div className="ds">
              <h5>Reply</h5>
              <pre className="reply-box">{cell.reply || '—'}</pre>
            </div>
            {cell.expected_answer != null && cell.expected_answer !== '' && (
              <div className="ds">
                <h5>Expected</h5>
                <pre className="reply-box">{String(cell.expected_answer)}</pre>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
