import React, { useMemo, useState } from 'react';
import { Cell, HistoryEntry } from '../types';
import {
  effectiveGenOk,
  effectiveVerifiedPass,
} from '../lib/metrics';
import {
  OUTCOME_EXPLAIN,
  OutcomeKey,
  taskAcceptance,
  taskDescription,
  taskTitle,
} from '../lib/assignment';
import TraceView from './TraceView';

interface Props {
  suite: string;
  taskId: string;
  cells: Cell[];
  history: HistoryEntry[];
  initialVariant?: 'stock' | 'ours';
  onBack: () => void;
  onOpenSuite: () => void;
}

function outcomeValue(cell: Cell, key: OutcomeKey): number | null {
  switch (key) {
    case 'pass_at_1':
      return cell.pass_at_1;
    case 'gen_ok':
      return effectiveGenOk(cell);
    case 'verified_pass_at_1':
      return effectiveVerifiedPass(cell);
    case 'partial_credit':
      return cell.partial_credit;
    case 'judge':
      return cell.judge_score == null || Number.isNaN(cell.judge_score)
        ? null
        : cell.judge_score;
    default:
      return null;
  }
}

export default function TaskPage({
  suite,
  taskId,
  cells,
  history,
  initialVariant = 'ours',
  onBack,
  onOpenSuite,
}: Props) {
  const variants = useMemo(() => {
    const set = new Set(
      cells.filter((c) => c.suite === suite && c.task_id === taskId).map((c) => c.variant),
    );
    return (['stock', 'ours'] as const).filter((v) => set.has(v));
  }, [cells, suite, taskId]);

  const [variant, setVariant] = useState<'stock' | 'ours'>(
    variants.includes(initialVariant) ? initialVariant : variants[0] || 'ours',
  );
  const [runIdx, setRunIdx] = useState(0);

  const cell = useMemo(
    () => cells.find((c) => c.suite === suite && c.task_id === taskId && c.variant === variant) || null,
    [cells, suite, taskId, variant],
  );

  const title = taskTitle(cell, taskId);
  const description = taskDescription(cell);
  const acceptance = taskAcceptance(cell);
  const hasPrompt = Boolean(cell?.prompt?.trim());
  const hasReply = Boolean(cell?.reply?.trim());
  const hasExpected = cell?.expected_answer != null && String(cell.expected_answer).trim() !== '';

  const outcomeKeys = Object.keys(OUTCOME_EXPLAIN) as OutcomeKey[];

  return (
    <div className="view-content task-page task-page-canvas" data-testid="task-page-canvas">
      <div className="detail-nav">
        <button type="button" className="gt-btn" onClick={onBack}>← Back</button>
        <button type="button" className="gt-btn" onClick={onOpenSuite}>Suite {suite}</button>
        <span className="faint mono">{suite}</span>
      </div>

      <header className="assignment-hero">
        <h2 className="assignment-title">{title}</h2>
        {title !== taskId && <div className="assignment-id mono faint">{taskId}</div>}
        {description ? (
          <p className="assignment-desc">{description}</p>
        ) : (
          <p className="assignment-desc muted">
            No description on this cell yet — enrich export with{' '}
            <code>description</code> / <code>task_title</code> (dual-read).
          </p>
        )}
        <div className="assignment-meta faint">
          {cell ? (
            <>
              <span>{cell.difficulty || '—'}</span>
              <span>·</span>
              <span>{cell.task_type || 'task'}</span>
              <span>·</span>
              <span>{cell.scoring_method || 'scoring?'}</span>
              {cell.ok != null && (
                <>
                  <span>·</span>
                  <span className={cell.ok ? 'good' : 'bad'}>{cell.ok ? 'ok' : 'fail'}</span>
                </>
              )}
            </>
          ) : (
            <span>No cell loaded</span>
          )}
        </div>
      </header>

      <div className="task-controls">
        <label>
          Variant
          <select value={variant} onChange={(e) => setVariant(e.target.value as 'stock' | 'ours')}>
            {variants.map((v) => (
              <option key={v} value={v}>{v}</option>
            ))}
          </select>
        </label>
        <label>
          Run
          <select value={runIdx} onChange={(e) => setRunIdx(Number(e.target.value))}>
            <option value={0}>live (current)</option>
            {history.map((h, i) => (
              <option key={i} value={i + 1}>
                hist #{i + 1} · {h.cellCount} cells · {new Date(h.receivedAt).toLocaleTimeString()}
              </option>
            ))}
          </select>
        </label>
        {runIdx > 0 && (
          <span className="faint">History stores suite-level summary only — cell body stays live.</span>
        )}
      </div>

      {!cell ? (
        <div className="empty-state">No cell for {suite}/{taskId} · {variant}</div>
      ) : (
        <>
          <section className="assignment-section" data-testid="assignment-acceptance">
            <h3 className="section-title">Acceptance / rubric</h3>
            {acceptance ? (
              <pre className="assignment-rubric">{acceptance}</pre>
            ) : (
              <div className="trace-empty">
                <p>
                  No acceptance criteria or rubric on this export. Dual-read keys:{' '}
                  <code>acceptance</code>, <code>acceptance_criteria</code>, <code>rubric</code>.
                  {cell.scoring_method && (
                    <> Scoring method: <code>{cell.scoring_method}</code>.</>
                  )}
                  {hasExpected && <> Expected answer is shown below.</>}
                </p>
              </div>
            )}
          </section>

          <section className="assignment-section" data-testid="assignment-outcomes">
            <h3 className="section-title">Outcomes</h3>
            <p className="faint assignment-lead">
              How this run scored — and what each metric means on V5 vs verified grading.
            </p>
            <div className="outcome-grid">
              {outcomeKeys.map((key) => {
                const def = OUTCOME_EXPLAIN[key];
                const raw = outcomeValue(cell, key);
                return (
                  <article key={key} className="outcome-card">
                    <div className="outcome-head">
                      <span className="outcome-label mono">{def.label}</span>
                      <span className="outcome-value">{def.fmt(raw)}</span>
                    </div>
                    <p className="outcome-blurb">{def.blurb}</p>
                  </article>
                );
              })}
            </div>
          </section>

          {(hasPrompt || hasReply || hasExpected) && (
            <section className="assignment-section" data-testid="assignment-io">
              <h3 className="section-title">Prompt / reply / expected</h3>
              {hasPrompt && (
                <div className="ds">
                  <h5>Prompt</h5>
                  <pre className="reply-box">{cell.prompt}</pre>
                </div>
              )}
              {hasReply && (
                <div className="ds">
                  <h5>Reply</h5>
                  <pre className="reply-box">{cell.reply}</pre>
                </div>
              )}
              {hasExpected && (
                <div className="ds">
                  <h5>Expected</h5>
                  <pre className="reply-box">{String(cell.expected_answer)}</pre>
                </div>
              )}
            </section>
          )}

          <section className="assignment-section" data-testid="assignment-trace">
            <h3 className="section-title">Chat / tool trace</h3>
            <p className="faint assignment-lead">
              iMessage-style turns and tool calls from <code>progress_trace</code> or{' '}
              <code>chat_trace</code>.
            </p>
            <TraceView cell={cell} chatOnly />
          </section>
        </>
      )}
    </div>
  );
}
