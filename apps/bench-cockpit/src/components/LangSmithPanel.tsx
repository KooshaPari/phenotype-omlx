import React, { useCallback, useEffect, useState } from 'react';

interface LsSession {
  id?: string;
  name?: string;
  description?: string;
  reference_dataset_id?: string | null;
  start_time?: string;
}

interface LsDataset {
  id?: string;
  name?: string;
  description?: string;
  example_count?: number | null;
}

interface LsStatus {
  enabled: boolean;
  error?: string;
  project_name?: string;
  dataset_name?: string;
  sessions?: LsSession[];
  datasets?: LsDataset[];
}

interface LsSetupResult {
  enabled: boolean;
  project_id?: string;
  project_name?: string;
  dataset_id?: string;
  dataset_name?: string;
  experiment_id?: string;
  examples_uploaded?: number;
  runs_posted?: number;
  dashboard_url?: string;
  errors?: string[];
  meta?: Record<string, string>;
}

export default function LangSmithPanel() {
  const [status, setStatus] = useState<LsStatus | null>(null);
  const [setup, setSetup] = useState<LsSetupResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setErr(null);
    try {
      const r = await fetch('/api/langsmith/status');
      const j = await r.json();
      setStatus(j);
      if (!r.ok && j.error) setErr(j.error);
    } catch (e) {
      setErr(String(e));
    }
  }, []);

  const [evalBusy, setEvalBusy] = useState(false);
  const [evalResult, setEvalResult] = useState<unknown>(null);
  const [evaluators, setEvaluators] = useState<{ name?: string; type?: string; id?: string; feedback_keys?: string[] }[]>([]);

  const refreshEvaluators = useCallback(async () => {
    try {
      const r = await fetch('/api/langsmith/evaluators');
      const j = await r.json();
      setEvaluators(j.evaluators || []);
    } catch {
      /* ignore */
    }
  }, []);

  useEffect(() => {
    void refresh();
    void refreshEvaluators();
  }, [refresh, refreshEvaluators]);

  const runEvals = async (action: 'sync' | 'run' | 'all' | 'hosted') => {
    setEvalBusy(true);
    setErr(null);
    try {
      const r = await fetch(`/api/langsmith/evaluators?action=${action}&limit=12`, { method: 'POST' });
      const j = await r.json();
      setEvalResult(j.result || j);
      if (!r.ok) setErr(j.error || 'evaluator run failed');
      await refreshEvaluators();
    } catch (e) {
      setErr(String(e));
    } finally {
      setEvalBusy(false);
    }
  };

  const runSetup = async () => {
    setBusy(true);
    setErr(null);
    setSetup(null);
    try {
      const r = await fetch('/api/langsmith/setup', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ max_cells: 40, seed_runs: true }),
      });
      const j = (await r.json()) as LsSetupResult;
      setSetup(j);
      if (!r.ok && j.errors?.length) setErr(j.errors[0]);
      await refresh();
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  };

  const sessions = status?.sessions ?? [];
  const datasets = status?.datasets ?? [];
  const projectName = status?.project_name || 'bench-cockpit';
  const project = sessions.find((s) => s.name === projectName);
  const projectURL = setup?.dashboard_url
    || (project?.id
      ? `https://smith.langchain.com/projects/p/${project.id}`
      : 'https://smith.langchain.com');

  return (
    <div className="view-stack" data-testid="langsmith-view">
      <div className="viz-panel">
        <div className="viz-toolbar">
          <span className="viz-title">LangSmith · project + dataset + seeded runs</span>
          <span className="viz-hint">
            {status?.enabled ? 'API key loaded' : 'disabled — set LANGSMITH_API_KEY in .env'}
          </span>
        </div>

        {!status?.enabled && (
          <div className="warn-banner">
            LangSmith is off. Add <code>LANGSMITH_API_KEY</code> to{' '}
            <code>apps/bench-cockpit/.env</code> and restart via{' '}
            <code>bash scripts/start-dev.sh</code>.
          </div>
        )}

        {err && <div className="bad">error: {err}</div>}

        <div className="task-controls" style={{ marginBottom: 16 }}>
          <button type="button" className="gt-btn" onClick={() => void refresh()} disabled={busy}>
            Refresh
          </button>
          <button
            type="button"
            className="gt-btn"
            onClick={() => void runSetup()}
            disabled={busy || !status?.enabled}
          >
            {busy ? 'Setting up…' : 'Setup / sync project'}
          </button>
          <button
            type="button"
            className="gt-btn"
            onClick={() => void runEvals('sync')}
            disabled={evalBusy || !status?.enabled}
          >
            Register code evaluators
          </button>
          <button
            type="button"
            className="gt-btn"
            onClick={() => void runEvals('all')}
            disabled={evalBusy || !status?.enabled}
          >
            {evalBusy ? 'Judging…' : 'Run code + Minimax judges'}
          </button>
          <button
            type="button"
            className="gt-btn"
            onClick={() => void runEvals('hosted')}
            disabled={evalBusy || !status?.enabled}
          >
            Sync hosted Minimax judges
          </button>
          <a className="gt-btn" href={projectURL} target="_blank" rel="noreferrer">
            Open dashboard
          </a>
          <a
            className="gt-btn"
            href="https://smith.langchain.com/evaluators"
            target="_blank"
            rel="noreferrer"
          >
            Evaluators UI
          </a>
        </div>

        <div className="warn-banner" style={{ marginBottom: 12 }}>
          Hosted path: Model config (UI) + Hub prompts <code>bench-correctness</code> /
          <code>bench-hallucination</code> / <code>bench-code-checker</code> + LLM evaluators /
          run rules via <b>Sync hosted Minimax judges</b>. Offline Minimax still posts feedback
          onto traces. Harbor: <code>bash scripts/evals/harbor_langsmith_smoke.sh</code> or{' '}
          <code>bash scripts/evals/run_via_harbor.sh --policy --langsmith</code> (Apple Container).
        </div>

        <div className="ds" style={{ marginBottom: 16 }} data-testid="harbor-kpi-props">
          <h5>Harbor → LangSmith KPI props (named)</h5>
          <p className="faint" style={{ marginTop: 0 }}>
            Dataset <code>omlx-harbor-tasks</code> · experiments{' '}
            <code>omlx-harbor-{'{hello|policy|niah|turbo}'}</code>. SSOT:{' '}
            <code>config/langsmith_harbor_kpis.json</code>
          </p>
          <ul className="mono" style={{ margin: 0, paddingLeft: 18, fontSize: 12 }}>
            <li>
              feedback <b>reward</b> ← verifier <code>/logs/verifier/reward.txt</code> (0|1)
            </li>
            <li>
              feedback <b>harbor_error</b> ← trial exception (when present)
            </li>
            <li>
              outputs <b>rewards</b>, <b>tokens.input/output</b>, <b>task_name</b>, <b>trial_name</b>
            </li>
            <li>
              metadata <b>ls_runner=harbor</b>, <b>harbor_job_id</b>, <b>harbor_job_name</b>
            </li>
          </ul>
        </div>

        {evaluators.length > 0 && (
          <div className="ds" style={{ marginBottom: 16 }}>
            <h5>Workspace evaluators ({evaluators.length})</h5>
            <ul className="mono" style={{ margin: 0, paddingLeft: 18, fontSize: 12 }}>
              {evaluators.map((e) => (
                <li key={e.id || e.name}>
                  {e.name} · {e.type} · keys {(e.feedback_keys || []).join(', ')}
                </li>
              ))}
            </ul>
          </div>
        )}

        {evalResult != null && (
          <div className="ds" style={{ marginBottom: 16 }}>
            <h5>Last evaluator run</h5>
            <pre className="reply-box">{JSON.stringify(evalResult, null, 2)}</pre>
          </div>
        )}

        {setup && (
          <div className="ov-grid" style={{ marginBottom: 16 }}>
            <div className="ov-card">
              <div className="ov-title">Project</div>
              <div className="ov-metric mono">{setup.project_name}</div>
              <div className="faint mono">{setup.project_id}</div>
            </div>
            <div className="ov-card">
              <div className="ov-title">Dataset</div>
              <div className="ov-metric mono">{setup.dataset_name}</div>
              <div className="faint">examples +{setup.examples_uploaded ?? 0}</div>
            </div>
            <div className="ov-card">
              <div className="ov-title">Seeded</div>
              <div className="ov-metric">{setup.runs_posted ?? 0} runs</div>
              <div className="faint mono">{setup.experiment_id || '—'}</div>
            </div>
          </div>
        )}

        <div className="audit-grid">
          <div className="audit-list">
            <h5 style={{ margin: '0 0 8px' }}>Projects / experiments ({sessions.length})</h5>
            {sessions.length === 0 && <div className="faint">None yet — run Setup.</div>}
            {sessions.map((s) => (
              <a
                key={s.id || s.name}
                className="audit-item"
                href={s.id ? `https://smith.langchain.com/projects/p/${s.id}` : projectURL}
                target="_blank"
                rel="noreferrer"
              >
                {s.name}
                {s.reference_dataset_id ? ' · experiment' : ' · project'}
              </a>
            ))}
          </div>
          <div className="audit-detail">
            <h5>Datasets ({datasets.length})</h5>
            {datasets.length === 0 && <div className="faint">None yet — run Setup.</div>}
            {datasets.map((d) => (
              <div key={d.id || d.name} className="ds" style={{ marginBottom: 10 }}>
                <div className="kv">
                  <span className="k">name</span>
                  <span className="v mono">{d.name}</span>
                </div>
                <div className="kv">
                  <span className="k">id</span>
                  <span className="v mono">{d.id}</span>
                </div>
                <div className="kv">
                  <span className="k">examples</span>
                  <span className="v">{d.example_count ?? '—'}</span>
                </div>
                {d.id && (
                  <a
                    href={`https://smith.langchain.com/datasets/${d.id}`}
                    target="_blank"
                    rel="noreferrer"
                  >
                    Open dataset →
                  </a>
                )}
              </div>
            ))}
            {setup?.errors && setup.errors.length > 0 && (
              <div className="ds">
                <h5>Setup notes</h5>
                <pre className="reply-box">{setup.errors.slice(0, 12).join('\n')}</pre>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
