import React, { useEffect, useMemo, useState } from 'react';
import { Cell } from '../types';

interface Props {
  cells: Cell[];
  /** Suites to auto-expand (e.g. from Overview jump). */
  focusSuites?: string[];
  onSelect?: (c: Cell) => void;
  /** When true, P@1 is treated as reported/synthetic (de-emphasize). */
  passAt1Untrusted?: boolean;
  onOpenSuite?: (suite: string) => void;
  onOpenTask?: (taskId: string, variant: 'stock' | 'ours', suite: string) => void;
}

type TaskPair = {
  taskId: string;
  difficulty: string;
  stock?: Cell;
  ours?: Cell;
};

type SuiteGroup = {
  suite: string;
  cells: Cell[];
  stock: Cell[];
  ours: Cell[];
  tasks: TaskPair[];
};

function perSuite(cells: Cell[]): SuiteGroup[] {
  const m = new Map<string, Cell[]>();
  for (const c of cells) {
    const a = m.get(c.suite) || [];
    a.push(c);
    m.set(c.suite, a);
  }
  return [...m.entries()]
    .sort((a, b) => a[0].localeCompare(b[0]))
    .map(([suite, arr]) => {
      const stock = arr.filter((c) => c.variant === 'stock');
      const ours = arr.filter((c) => c.variant === 'ours');
      const byTask = new Map<string, TaskPair>();
      for (const c of arr) {
        const cur = byTask.get(c.task_id) || {
          taskId: c.task_id,
          difficulty: c.difficulty || '—',
          stock: undefined,
          ours: undefined,
        };
        if (c.variant === 'stock') cur.stock = c;
        else if (c.variant === 'ours') cur.ours = c;
        if (!cur.difficulty || cur.difficulty === '—') {
          cur.difficulty = c.difficulty || '—';
        }
        byTask.set(c.task_id, cur);
      }
      const tasks = [...byTask.values()].sort((a, b) =>
        a.taskId.localeCompare(b.taskId),
      );
      return { suite, cells: arr, stock, ours, tasks };
    });
}

function mean(vals: number[]): number {
  return vals.length ? vals.reduce((s, x) => s + x, 0) / vals.length : 0;
}

function fmtPct(v: number): string {
  return `${(v * 100).toFixed(1)}%`;
}

function fmtDeltaPp(stock: number, ours: number): { text: string; cls: string } {
  const d = (ours - stock) * 100;
  const text = `${d >= 0 ? '+' : ''}${d.toFixed(1)}pp`;
  const cls = d > 0.05 ? 'positive' : d < -0.05 ? 'negative' : 'neutral';
  return { text, cls };
}

function fmtDeltaRaw(stock: number, ours: number, digits = 3): { text: string; cls: string } {
  const d = ours - stock;
  const text = `${d >= 0 ? '+' : ''}${d.toFixed(digits)}`;
  const cls = d > 0.005 ? 'positive' : d < -0.005 ? 'negative' : 'neutral';
  return { text, cls };
}

function MetricCell({
  value,
  fmt,
  tone,
}: {
  value: number;
  fmt: (v: number) => string;
  tone: 'stock' | 'ours';
}) {
  return (
    <td className={`heat-cell ${tone}`}>
      <span className="v">{fmt(value)}</span>
    </td>
  );
}

export default function Suites({
  cells,
  focusSuites = [],
  onSelect,
  passAt1Untrusted = false,
  onOpenSuite,
  onOpenTask,
}: Props) {
  const suites = useMemo(() => perSuite(cells), [cells]);
  const [expanded, setExpanded] = useState<Set<string>>(() => new Set());

  useEffect(() => {
    if (!focusSuites.length) return;
    setExpanded((prev) => {
      let changed = false;
      const next = new Set(prev);
      for (const s of focusSuites) {
        if (!next.has(s)) {
          next.add(s);
          changed = true;
        }
      }
      return changed ? next : prev;
    });
  }, [focusSuites]);

  const toggle = (suite: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(suite)) next.delete(suite);
      else next.add(suite);
      return next;
    });
  };

  const expandAll = () => setExpanded(new Set(suites.map((s) => s.suite)));
  const collapseAll = () => setExpanded(new Set());

  const allPassOne = useMemo(() => {
    if (!cells.length) return false;
    const meanP = cells.reduce((s, c) => s + c.pass_at_1, 0) / cells.length;
    return meanP >= 0.999;
  }, [cells]);
  const showReportedBanner = passAt1Untrusted || allPassOne;

  return (
    <div className="view-content suites-view">
      <div className="suites-toolbar">
        <div className="suites-toolbar-title">
          Suites
          <span className="faint"> · {suites.length} suites · {cells.length} cells</span>
        </div>
        <div className="suites-toolbar-actions">
          <button type="button" className="gt-btn" onClick={expandAll}>
            Expand all
          </button>
          <button type="button" className="gt-btn" onClick={collapseAll}>
            Collapse all
          </button>
        </div>
      </div>

      {showReportedBanner && (
        <div className="warn-banner suites-banner">
          Pass@1 is ~100% on every suite in this file — that is <b>reported/synthetic</b> evidence
          (not live grading). Prefer <b>PC</b> / <b>Wall</b> / format columns; open Calibration for
          lint details.
        </div>
      )}

      <table className="heat-table suites-table">
        <thead>
          <tr>
            <th className="col-expand" />
            <th>Suite / Task</th>
            <th>n</th>
            <th className="subhead">PC stock</th>
            <th className="subhead">PC ours</th>
            <th>Δ PC</th>
            <th className="subhead">Wall s</th>
            <th className="subhead">Wall o</th>
            <th className={`subhead ${showReportedBanner ? 'p1-reported' : ''}`}>P@1 s</th>
            <th className={`subhead ${showReportedBanner ? 'p1-reported' : ''}`}>P@1 o</th>
          </tr>
        </thead>
        <tbody>
          {suites.map((g) => {
            const open = expanded.has(g.suite);
            const sPass = mean(g.stock.map((c) => c.pass_at_1));
            const oPass = mean(g.ours.map((c) => c.pass_at_1));
            const sWall = mean(g.stock.map((c) => c.wall_clock_s).filter((v) => v > 0));
            const oWall = mean(g.ours.map((c) => c.wall_clock_s).filter((v) => v > 0));
            const sPc = mean(g.stock.map((c) => c.partial_credit));
            const oPc = mean(g.ours.map((c) => c.partial_credit));
            const deltaPc = fmtDeltaRaw(sPc, oPc);
            const focused = focusSuites.includes(g.suite);

            return (
              <React.Fragment key={g.suite}>
                <tr
                  className={`suite-row ${open ? 'open' : ''} ${focused ? 'focused' : ''}`}
                  onClick={() => toggle(g.suite)}
                >
                  <td className="col-expand">
                    <span className={`chevron ${open ? 'open' : ''}`} aria-hidden>
                      ▸
                    </span>
                  </td>
                  <td>
                    <b>{g.suite}</b>
                    <span className="faint"> · {g.tasks.length} tasks</span>
                    {onOpenSuite && (
                      <button
                        type="button"
                        className="gt-btn suite-open-btn"
                        onClick={(e) => {
                          e.stopPropagation();
                          onOpenSuite(g.suite);
                        }}
                      >
                        Open
                      </button>
                    )}
                  </td>
                  <td className="mono">{g.cells.length}</td>
                  <MetricCell value={sPc} fmt={(v) => v.toFixed(3)} tone="stock" />
                  <MetricCell value={oPc} fmt={(v) => v.toFixed(3)} tone="ours" />
                  <td>
                    <span className={`vc-delta ${deltaPc.cls}`}>{deltaPc.text}</span>
                  </td>
                  <td className="mono faint">{sWall ? sWall.toFixed(2) : '—'}</td>
                  <td className="mono faint">{oWall ? oWall.toFixed(2) : '—'}</td>
                  <td className={`heat-cell stock ${showReportedBanner ? 'p1-reported' : ''}`}>
                    <span className="v">{fmtPct(sPass)}</span>
                  </td>
                  <td className={`heat-cell ours ${showReportedBanner ? 'p1-reported' : ''}`}>
                    <span className="v">{fmtPct(oPass)}</span>
                  </td>
                </tr>

                {open &&
                  g.tasks.map((t) => {
                    const stock = t.stock;
                    const ours = t.ours;
                    const sp = stock?.pass_at_1 ?? 0;
                    const op = ours?.pass_at_1 ?? 0;
                    const spc = stock?.partial_credit ?? 0;
                    const opc = ours?.partial_credit ?? 0;
                    const dPc = fmtDeltaRaw(
                      stock ? spc : opc,
                      ours ? opc : spc,
                    );
                    const pick = ours ?? stock;
                    return (
                      <tr
                        key={`${g.suite}:${t.taskId}`}
                        className="task-row"
                        onClick={(e) => {
                          e.stopPropagation();
                          if (onOpenTask) {
                            onOpenTask(t.taskId, ours ? 'ours' : 'stock', g.suite);
                          } else if (pick && onSelect) {
                            onSelect(pick);
                          }
                        }}
                      >
                        <td className="col-expand" />
                        <td className="task-cell">
                          <span className="task-id mono">{t.taskId}</span>
                          <span className="badge">{t.difficulty}</span>
                        </td>
                        <td className="mono faint">
                          {(stock ? 1 : 0) + (ours ? 1 : 0)}
                        </td>
                        <td className="heat-cell stock">
                          <span className="v">{stock ? spc.toFixed(3) : '—'}</span>
                        </td>
                        <td className="heat-cell ours">
                          <span className="v">{ours ? opc.toFixed(3) : '—'}</span>
                        </td>
                        <td>
                          {stock && ours ? (
                            <span className={`vc-delta ${dPc.cls}`}>{dPc.text}</span>
                          ) : (
                            <span className="faint">—</span>
                          )}
                        </td>
                        <td className="mono faint">
                          {stock ? stock.wall_clock_s.toFixed(2) : '—'}
                        </td>
                        <td className="mono faint">
                          {ours ? ours.wall_clock_s.toFixed(2) : '—'}
                        </td>
                        <td className={`heat-cell stock ${showReportedBanner ? 'p1-reported' : ''}`}>
                          <span className="v">{stock ? fmtPct(sp) : '—'}</span>
                        </td>
                        <td className={`heat-cell ours ${showReportedBanner ? 'p1-reported' : ''}`}>
                          <span className="v">{ours ? fmtPct(op) : '—'}</span>
                        </td>
                      </tr>
                    );
                  })}
              </React.Fragment>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
