import React, { useEffect, useState } from 'react';

type Fit = {
  fits: boolean;
  vram_estimate_gb: number;
  available_gb: number;
  params: number;
  dtype: string;
  source: string;
  error?: string;
};

export default function VramFit({ modelName }: { modelName?: string }) {
  const [fit, setFit] = useState<Fit | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    const model = (modelName || '').trim();
    if (!model) {
      setFit(null);
      return;
    }
    let cancelled = false;
    (async () => {
      try {
        const r = await fetch(
          `/api/capacity/fit?model=${encodeURIComponent(model)}&dtype=F16`,
        );
        const j = (await r.json()) as Fit & { error?: string; hint?: string };
        if (cancelled) return;
        if (j.error) {
          setErr(j.error);
          setFit(null);
          return;
        }
        setErr(null);
        setFit(j);
      } catch (e) {
        if (!cancelled) setErr(String(e));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [modelName]);

  if (!modelName) return null;
  if (err) {
    return (
      <div className="ds">
        <h5>VRAM fit</h5>
        <div className="kv">
          <span className="k">Status</span>
          <span className="v warn">{err}</span>
        </div>
      </div>
    );
  }
  if (!fit) {
    return (
      <div className="ds">
        <h5>VRAM fit</h5>
        <div className="kv">
          <span className="k">Status</span>
          <span className="v faint">checking…</span>
        </div>
      </div>
    );
  }

  const cls = fit.fits ? 'good' : 'bad';
  return (
    <div className="ds" data-testid="vram-fit">
      <h5>VRAM fit</h5>
      <div className="kv">
        <span className="k">Fits</span>
        <span className={`v ${cls}`}>{fit.fits ? 'yes' : 'no'}</span>
      </div>
      <div className="kv">
        <span className="k">Need</span>
        <span className="v">{fit.vram_estimate_gb.toFixed(2)} GB ({fit.dtype})</span>
      </div>
      <div className="kv">
        <span className="k">Have</span>
        <span className="v">{fit.available_gb.toFixed(1)} GiB</span>
      </div>
      <div className="kv">
        <span className="k">Params</span>
        <span className="v faint">
          {(fit.params / 1e9).toFixed(2)}B · {fit.source}
        </span>
      </div>
    </div>
  );
}
