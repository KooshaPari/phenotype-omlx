import { useCallback, useEffect, useState } from 'react';

type LfStatus = {
  enabled?: boolean;
  backend?: string;
  base_url?: string;
  health?: { status?: string; version?: string };
  projects?: { data?: { id?: string; name?: string }[] };
  dashboard_url?: string;
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

  const runJudges = async () => {
    setBusy(true);
    setErr(null);
    try {
      const r = await fetch('/api/langfuse/evaluators?action=judge&limit=12', { method: 'POST' });
      const j = await r.json();
      setResult(j);
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  };

  const dash = status?.dashboard_url || status?.base_url || 'https://us.cloud.langfuse.com';
  const projectName = status?.projects?.data?.[0]?.name;

  return (
    <div className="view-stack" data-testid="langfuse-view">
      <div className="ds" style={{ marginBottom: 16 }}>
        <h3>Langfuse (primary)</h3>
        <p className="muted" style={{ marginTop: 4 }}>
          OSS observability with full feature control. Prefer self-host (Podman / Apple Container)
          for zero SaaS spend; cloud Hobby works for smoke. LangSmith remains optional legacy.
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
        </div>
        <div className="row" style={{ gap: 8, marginTop: 12, flexWrap: 'wrap' }}>
          <button type="button" className="gt-btn" onClick={() => void refresh()}>
            Refresh
          </button>
          <button
            type="button"
            className="gt-btn"
            onClick={() => void runSetup()}
            disabled={busy || !status?.enabled}
          >
            {busy ? 'Seeding…' : 'Seed traces from V5 cells'}
          </button>
          <button
            type="button"
            className="gt-btn"
            onClick={() => void runJudges()}
            disabled={busy || !status?.enabled}
          >
            {busy ? 'Judging…' : 'Run Minimax judges → Langfuse'}
          </button>
          <a className="gt-btn" href={dash} target="_blank" rel="noreferrer">
            Open Langfuse
          </a>
        </div>
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
