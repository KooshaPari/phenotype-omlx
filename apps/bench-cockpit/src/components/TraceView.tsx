import React, { useMemo, useState } from 'react';
import { Cell } from '../types';
import { decodeTrace, Span } from '../types/span';

interface Props {
  cell: Cell;
}

/** Multi-turn Trace IDE/Chat: timeline + span list + prompt/reply panes. */
export default function TraceView({ cell }: Props) {
  const spans = useMemo(() => decodeTrace(cell.progress_trace), [cell.progress_trace]);
  const turns = spans.filter((s): s is Extract<Span, { kind: 'turn' }> => s.kind === 'turn');
  const [active, setActive] = useState(0);

  return (
    <div className="trace-view" data-testid="trace-view">
      <div className="trace-rail">
        <h5>Turns</h5>
        {turns.length === 0 && <div className="muted">No turn spans — showing raw trace ({spans.length})</div>}
        {turns.map((t, i) => (
          <button
            key={t.id}
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
              <code>{s.kind}</code> {s.name || s.id}
              {s.kind === 'reward' && s.composite != null && (
                <span className="v"> composite={s.composite.toFixed(3)}</span>
              )}
              {s.kind === 'verifier' && (
                <span className="v">
                  {' '}
                  {s.passed ? 'pass' : 'fail'} r={s.reward?.toFixed?.(2) ?? '—'}
                </span>
              )}
            </li>
          ))}
        </ul>
      </div>
      <div className="trace-main">
        <div className="ds">
          <h5>Prompt</h5>
          <pre className="reply-box">{cell.prompt || '—'}</pre>
        </div>
        <div className="ds">
          <h5>Reply</h5>
          <pre className="reply-box">{cell.reply || '—'}</pre>
        </div>
        {cell.expected_answer != null && (
          <div className="ds">
            <h5>Expected</h5>
            <pre className="reply-box">{String(cell.expected_answer)}</pre>
          </div>
        )}
        {turns[active]?.content && (
          <div className="ds">
            <h5>Active turn content</h5>
            <pre className="reply-box">{turns[active].content}</pre>
          </div>
        )}
      </div>
    </div>
  );
}
