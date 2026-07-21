import React, { useEffect, useMemo, useState } from 'react';
import { Cell } from '../types';
import { stubIRT, wilsonCI } from '../lib/irt';

interface Props {
  cells: Cell[];
  seed?: Cell | null;
}

interface RawPayload {
  suite: string;
  task_id: string;
  variant: string;
  prompt?: string;
  reply?: string;
  expected_answer?: string;
  scoring_method?: string;
  pass_at_1?: number;
  judge_score?: number;
  failure_analysis?: Record<string, unknown>;
  progress_trace?: unknown[];
  error?: string;
}

export default function Audit({ cells, seed }: Props) {
  const [selected, setSelected] = useState<Cell | null>(seed ?? cells[0] ?? null);
  const [raw, setRaw] = useState<RawPayload | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (seed) setSelected(seed);
  }, [seed]);

  useEffect(() => {
    if (!selected) return;
    const path = `/api/cells/${encodeURIComponent(selected.suite)}/${encodeURIComponent(selected.task_id)}/${encodeURIComponent(selected.variant)}/raw`;
    setErr(null);
    fetch(path)
      .then(async (r) => {
        const j = await r.json();
        if (!r.ok) throw new Error(j.error || r.statusText);
        setRaw(j);
      })
      .catch((e) => setErr(String(e.message || e)));
  }, [selected]);

  const suspicious = useMemo(
    () =>
      cells.filter(
        (c) =>
          c.pass_at_1 >= 0.999 &&
          (!c.expected_answer || !c.scoring_method || c.wall_clock_s < 0.05)
      ),
    [cells]
  );

  const ci = useMemo(() => {
    const n = cells.length;
    const ok = cells.filter((c) => c.pass_at_1 >= 0.5).length;
    return wilsonCI(ok, n);
  }, [cells]);

  const irt = useMemo(() => {
    if (!selected) return null;
    const rates = cells
      .filter((c) => c.task_id === selected.task_id)
      .map((c) => c.pass_at_1);
    return stubIRT(selected.task_id, rates);
  }, [cells, selected]);

  return (
    <div className="view-stack" data-testid="audit-view">
      <div className="viz-panel">
        <div className="viz-toolbar">
          <span className="viz-title">Audit · raw cell + Wilson CI</span>
          <span className="viz-hint">
            pass≥0.5 Wilson 95%: [{(ci.low * 100).toFixed(1)}%, {(ci.high * 100).toFixed(1)}%] n={ci.n}
          </span>
        </div>
        {suspicious.length > 0 && (
          <div className="warn-banner">
            {suspicious.length} suspicious 100% cell(s) — empty expected/scoring or sub-50ms wall
          </div>
        )}
        <div className="audit-grid">
          <div className="audit-list">
            {cells.slice(0, 200).map((c) => (
              <button
                key={`${c.suite}-${c.task_id}-${c.variant}`}
                className={`audit-item ${selected === c ? 'active' : ''} ${
                  c.pass_at_1 >= 0.999 && !c.expected_answer ? 'bad' : ''
                }`}
                onClick={() => setSelected(c)}
              >
                {c.suite}/{c.task_id} · {c.variant} · {(c.pass_at_1 * 100).toFixed(0)}%
              </button>
            ))}
          </div>
          <div className="audit-detail">
            {err && <div className="bad">raw fetch: {err}</div>}
            {irt && (
              <div className="ds">
                <h5>IRT stub</h5>
                <div className="kv"><span className="k">difficulty</span><span className="v">{irt.difficulty.toFixed(3)}</span></div>
                <div className="kv"><span className="k">discrimination</span><span className="v">{irt.discrimination.toFixed(3)}</span></div>
                <div className="kv"><span className="k">guess floor</span><span className="v">{irt.guessingFloor.toFixed(2)}</span></div>
                <div className="kv"><span className="k">ceiling</span><span className="v">{irt.ceiling.toFixed(2)}</span></div>
              </div>
            )}
            {raw && (
              <>
                <div className="ds">
                  <h5>Meta</h5>
                  <div className="kv"><span className="k">scoring</span><span className="v">{raw.scoring_method || '—'}</span></div>
                  <div className="kv"><span className="k">expected</span><span className="v">{raw.expected_answer || '—'}</span></div>
                  <div className="kv"><span className="k">pass@1</span><span className="v">{raw.pass_at_1}</span></div>
                  <div className="kv"><span className="k">judge</span><span className="v">{raw.judge_score}</span></div>
                </div>
                <div className="ds"><h5>Prompt</h5><pre className="reply-box">{raw.prompt || '—'}</pre></div>
                <div className="ds"><h5>Reply</h5><pre className="reply-box">{raw.reply || '—'}</pre></div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
