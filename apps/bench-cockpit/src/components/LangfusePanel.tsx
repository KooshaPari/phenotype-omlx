import { useCallback, useEffect, useState } from 'react';

type LfConn = {
  provider?: string;
  adapter?: string;
  baseURL?: string;
  customModels?: string[];
};

type LfEval = {
  name?: string;
  scope?: string;
  modelConfig?: { provider?: string; model?: string };
  evaluationRuleCount?: number;
};

type LfRule = {
  name?: string;
  target?: string;
  enabled?: boolean;
  status?: string;
};

type LfStatus = {
  enabled?: boolean;
  backend?: string;
  base_url?: string;
  health?: { status?: string; version?: string };
  projects?: { data?: { id?: string; name?: string }[] };
  dashboard_url?: string;
  llm_connections?: { data?: LfConn[] };
  evaluators?: { data?: LfEval[] };
  evaluation_rules?: { data?: LfRule[] };
  error?: string;
};

export function LangfusePanel() {
  const [status, setStatus] = useState<LfStatus | null>(null);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<unknown>(null);
  const [err, setErr] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const r = await fetch('/api/langfuse/status');
      const j = (await r.json()) as LfStatus;
      setStatus(j);
    } catch (e) {
      setErr(String(e));
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const runSetup = async () => {
    setBusy(true);
    setErr(null);
    try {
      const r = await fetch('/api/langfuse/setup', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ max_cells: 40 }),
      });
      const j = await r.json();
      setResult(j);
      await refresh();
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  };

  const runAction = async (action: 'sync' | 'judge') => {
    setBusy(true);
    setErr(null);
    try {
      const r = await fetch(`/api/langfuse/evaluators?action=${action}&limit=12`, {
        method: 'POST',
      });
      const j = await r.json();
      setResult(j);
      await refresh();
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  };

  const dash = status?.dashboard_url || status?.base_url || 'https://us.cloud.langfuse.com';
  const projectName = status?.projects?.data?.[0]?.name;
  const conns = status?.llm_connections?.data ?? [];
  const projectEvals = (status?.evaluators?.data ?? []).filter((e) => e.scope === 'project');
  const rules = status?.evaluation_rules?.data ?? [];
  const hasMinimax = conns.some(
    (c) => c.provider === 'Minimax' && (c.customModels ?? []).includes('Minimax-M3'),
  );

  return (
    <div className="view-stack" data-testid="langfuse-view">
      <div className="ds" style={{ marginBottom: 16 }}>
        <h3>Langfuse (primary)</h3>
        <p className="muted" style={{ marginTop: 4 }}>
          Cloud Hobby by default — traces, custom dashboards (bench-cockpit-ops), hosted
          LLM-as-judge, datasets, MCP/CLI/Skill. Self-host only when Hobby caps bite. LangSmith is
          optional legacy.
        </p>
        <div className="kv-grid" style={{ marginTop: 12 }}>
          <div className="kv">
            <span className="k">backend</span>
            <span className="v">{status?.backend ?? '…'}</span>
          </div>
          <div className="kv">
            <span className="k">enabled</span>
            <span className="v">{status?.enabled ? 'yes' : 'no'}</span>
          </div>
          <div className="kv">
            <span className="k">health</span>
            <span className="v">
              {status?.health?.status ?? '—'} {status?.health?.version ?? ''}
            </span>
          </div>
          <div className="kv">
            <span className="k">project</span>
            <span className="v">{projectName ?? '—'}</span>
          </div>
          <div className="kv">
            <span className="k">Minimax LLM</span>
            <span className="v">{hasMinimax ? 'connected' : 'missing'}</span>
          </div>
        </div>

        {!hasMinimax && status?.enabled && (
          <p className="muted" style={{ marginTop: 12 }}>
            Add LLM connection once: Settings → LLM Connections → custom provider{' '}
            <code>Minimax</code>, adapter <code>anthropic</code>, base{' '}
            <code>https://api.minimax.io/anthropic</code>, model <code>Minimax-M3</code>. Then Sync
            hosted judges.
          </p>
        )}

        <div className="row" style={{ gap: 8, marginTop: 12, flexWrap: 'wrap' }}>
          <button type="button" className="gt-btn" onClick={() => void refresh()}>
            Refresh
          </button>
          <button
            type="button"
            className="gt-btn"
            onClick={() => void runAction('sync')}
            disabled={busy || !status?.enabled}
          >
            {busy ? 'Syncing…' : 'Sync hosted Minimax judges'}
          </button>
          <button
            type="button"
            className="gt-btn"
            onClick={() => void runSetup()}
            disabled={busy || !status?.enabled}
          >
            {busy ? 'Seeding…' : 'Seed traces + generations'}
          </button>
          <button
            type="button"
            className="gt-btn"
            onClick={() => void runAction('judge')}
            disabled={busy || !status?.enabled}
          >
            {busy ? 'Judging…' : 'Offline Minimax → scores'}
          </button>
          <a className="gt-btn" href={dash} target="_blank" rel="noreferrer">
            Open Langfuse
          </a>
          <a
            className="gt-btn"
            href={dash}
            target="_blank"
            rel="noreferrer"
            title="Cloud UI → Dashboards → bench-cockpit-ops"
          >
            Dashboards
          </a>
        </div>
        <p className="faint mono" style={{ marginTop: 10 }}>
          bootstrap: scripts/evals/setup_langfuse_cloud.py · MCP:
          scripts/langfuse/print-cursor-mcp-snippet.sh · skill: npx skills add langfuse/skills
          --skill langfuse
        </p>

        {conns.length > 0 && (
          <div style={{ marginTop: 16 }}>
            <h5>LLM connections</h5>
            <ul className="faint mono" style={{ margin: '8px 0 0', paddingLeft: 18 }}>
              {conns.map((c) => (
                <li key={`${c.provider}-${c.baseURL}`}>
                  {c.provider} / {c.adapter} → {c.baseURL} [{(c.customModels ?? []).join(', ')}]
                </li>
              ))}
            </ul>
          </div>
        )}

        {projectEvals.length > 0 && (
          <div style={{ marginTop: 16 }}>
            <h5>Project evaluators ({projectEvals.length})</h5>
            <ul className="faint mono" style={{ margin: '8px 0 0', paddingLeft: 18 }}>
              {projectEvals.map((e) => (
                <li key={e.name}>
                  {e.name} · {e.modelConfig?.provider}/{e.modelConfig?.model} · rules{' '}
                  {e.evaluationRuleCount ?? 0}
                </li>
              ))}
            </ul>
          </div>
        )}

        {rules.length > 0 && (
          <div style={{ marginTop: 16 }}>
            <h5>Evaluation rules ({rules.length})</h5>
            <ul className="faint mono" style={{ margin: '8px 0 0', paddingLeft: 18 }}>
              {rules.map((r) => (
                <li key={r.name}>
                  {r.name} · {r.target} · {r.status ?? (r.enabled ? 'enabled' : 'off')}
                </li>
              ))}
            </ul>
          </div>
        )}

        {err && <pre className="reply-box" style={{ marginTop: 12 }}>{err}</pre>}
        {result != null && (
          <pre className="reply-box" style={{ marginTop: 12 }}>
            {JSON.stringify(result, null, 2)}
          </pre>
        )}
      </div>
    </div>
  );
}
