import React, { useEffect, useCallback, useMemo, useState, useRef } from 'react';
import { useBenchState } from './state/useBenchState';
import SummaryBar from './components/SummaryBar';
import VerdictStrip from './components/VerdictStrip';
import Overview from './components/Overview';
import Suites from './components/Suites';
import CellsTable from './components/CellsTable';
import Comparison from './components/Comparison';
import Failures from './components/Failures';
import Calibration from './components/Calibration';
import Viz from './components/Viz';
import Throughput from './components/Throughput';
import RLVRPanel from './components/RLVRPanel';
import Audit from './components/Audit';
import Drawer from './components/Drawer';
import { Cell, ViewType, Insight } from './types';

// --- Toast System ---
interface Toast {
  id: string;
  message: string;
  type: 'info' | 'success' | 'error';
}

function ToastContainer({ toasts, onDismiss }: { toasts: Toast[]; onDismiss: (id: string) => void }) {
  return (
    <div className="toast-container">
      {toasts.map(t => (
        <div key={t.id} className={`toast toast-${t.type}`} onClick={() => onDismiss(t.id)}>
          {t.message}
        </div>
      ))}
    </div>
  );
}

// --- Command Palette ---
interface PaletteProps {
  isOpen: boolean;
  views: ViewType[];
  actions: { label: string; key: string; action: () => void }[];
  cells: Cell[];
  onSelect: (item: any) => void;
  onClose: () => void;
}

function CommandPalette({ isOpen, views, actions, cells, onSelect, onClose }: PaletteProps) {
  const [query, setQuery] = useState('');
  const [selectedIdx, setSelectedIdx] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (isOpen) {
      setQuery('');
      setSelectedIdx(0);
      setTimeout(() => inputRef.current?.focus(), 50);
    }
  }, [isOpen]);

  const items = useMemo(() => {
    const q = query.toLowerCase();
    const viewItems = views.map(v => ({ type: 'view', id: v, label: v.charAt(0).toUpperCase() + v.slice(1) }));
    const actionItems = actions.map(a => ({ type: 'action', id: a.key, label: a.label, action: a.action }));
    const cellItems = cells.slice(0, 50).map(c => ({ type: 'cell', id: `${c.task_id}-${c.variant}`, label: `${c.task_id} · ${c.variant}`, data: c }));

    return [...viewItems, ...actionItems, ...cellItems].filter(i => 
      i.label.toLowerCase().includes(q)
    );
  }, [query, views, actions, cells]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setSelectedIdx(i => (i + 1) % items.length);
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setSelectedIdx(i => (i - 1 + items.length) % items.length);
    } else if (e.key === 'Enter') {
      if (items[selectedIdx]) onSelect(items[selectedIdx]);
    } else if (e.key === 'Escape') {
      onClose();
    }
  };

  if (!isOpen) return null;

  return (
    <div className="palette-overlay" onClick={onClose}>
      <div className="palette-modal" onClick={e => e.stopPropagation()}>
        <input
          ref={inputRef}
          className="palette-input"
          placeholder="Type a command or search..."
          value={query}
          onChange={e => { setQuery(e.target.value); setSelectedIdx(0); }}
          onKeyDown={handleKeyDown}
        />
        <div className="palette-results">
          {items.map((item, idx) => (
            <div
              key={item.id}
              className={`palette-item ${idx === selectedIdx ? 'selected' : ''}`}
              onClick={() => onSelect(item)}
              onMouseEnter={() => setSelectedIdx(idx)}
            >
              <span className="pi-icon">{item.type === 'view' ? '◈' : item.type === 'action' ? '⚡' : '☰'}</span>
              <span className="pi-label">{item.label}</span>
              {item.type === 'cell' && <span className="pi-meta">{(item as any).data.suite}</span>}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const { state, dispatch, connectWS, filteredCells, cellStatus, history, diff, insights, trends, pairedCell } = useBenchState();
  const [selected, setSelected] = useState<Cell | null>(null);
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [toasts, setToasts] = useState<Toast[]>([]);

  // --- Toasts ---
  const addToast = useCallback((message: string, type: Toast['type'] = 'info') => {
    const id = Date.now().toString();
    setToasts(prev => [...prev, { id, message, type }]);
    setTimeout(() => setToasts(prev => prev.filter(t => t.id !== id)), 4000);
  }, []);

  // --- Connect WS ---
  useEffect(() => { connectWS(); }, [connectWS]);
  
  // --- Toasts for Connection Status ---
  useEffect(() => {
    if (state.payload) {
      addToast('Data stream received', 'success');
    } else {
      addToast('Connecting...', 'info');
    }
  }, [state.payload, addToast]);

  const allCells = state.payload?.data?.cells ?? [];
  const summary = state.payload?.data?.summary ?? null;

  const cells = useMemo(() => {
    if (state.payload) return filteredCells();
    return [];
  }, [state.payload, filteredCells]);

  // --- Selection Logic ---
  const handleSelect = useCallback((c: Cell | null) => {
    setSelected(c);
  }, []);

  const onJumpToSuite = useCallback((suite: string) => {
    dispatch({ type: 'SET_VIEW', view: 'cells' });
    dispatch({ type: 'SET_FILTER', filterKey: 'suite', value: suite });
  }, [dispatch]);

  const statusLevel = state.payload ? 'connected' : 'error';
  const statusText = state.payload ? 'LIVE' : 'DISCONNECTED';





  // --- Keyboard Shortcuts ---
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      // Cmd+K / Ctrl+K
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        setPaletteOpen(p => !p);
        return;
      }
      // Esc
      if (e.key === 'Escape') {
        if (paletteOpen) setPaletteOpen(false);
        else if (selected) handleSelect(null);
        return;
      }

      // Only if not in an input
      if ((e.target as HTMLElement).tagName === 'INPUT') return;

      const viewMap: Record<string, ViewType> = {
        '1': 'overview', '2': 'suites', '3': 'cells', '4': 'comparison',
        '5': 'failures', '6': 'calibration', '7': 'viz', '8': 'throughput',
        '9': 'rlvr', '0': 'audit',
      };
      if (viewMap[e.key]) {
        dispatch({ type: 'SET_VIEW', view: viewMap[e.key] });
      }

      // j/k scroll failures
      if (state.view === 'failures' && (e.key === 'j' || e.key === 'k')) {
        // Assuming the failure table has an ID
        const table = document.querySelector('.fail-table-wrap');
        if (table) {
          table.scrollBy({ top: e.key === 'j' ? 40 : -40, behavior: 'smooth' });
        }
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [paletteOpen, selected, state.view, dispatch, handleSelect, setPaletteOpen]);

  // --- Export Actions ---
  const onExportMd = useCallback(() => {
    addToast('Markdown export started', 'info');
  }, [addToast]);

  const onExportJson = useCallback(() => {
    addToast('JSON export started', 'info');
  }, [addToast]);

  const onCopySummary = useCallback(() => {
    if (!state.payload) return;
    navigator.clipboard.writeText(JSON.stringify(state.payload.data.summary, null, 2));
    addToast('Summary copied to clipboard', 'success');
  }, [state.payload, addToast]);

  const onInsightAction = useCallback((insight: Insight) => {
    if (insight.jumpTo) {
      // Parse optional query from jumpTo like 'cells?suite=X'
      const [viewPart, qs] = insight.jumpTo.split('?');
      dispatch({ type: 'SET_VIEW', view: viewPart as ViewType });
      if (qs) {
        const params = new URLSearchParams(qs);
        for (const [k, v] of params) {
          if (k === 'suite' || k === 'difficulty') {
            dispatch({ type: 'SET_FILTER', filterKey: k, value: v });
          }
        }
      }
    }
  }, [dispatch]);

  const renderContent = () => {
    if (!state.payload) {
      return <div className="empty-state">● Awaiting WebSocket data stream…</div>;
    }
    switch (state.view) {
      case 'overview':
        return <Overview cells={allCells} summary={summary!} onJumpToSuite={onJumpToSuite} />;
      case 'suites':
        return <Suites cells={allCells} />;
      case 'cells':
        return <CellsTable cells={cells} state={state} onSelect={handleSelect} onSort={(k) => dispatch({ type: 'SORT', key: k })} onGroup={(g) => dispatch({ type: 'GROUP', group: g })} />;
      case 'comparison':
        return <Comparison cells={allCells} onSelect={handleSelect} />;
      case 'failures':
        return <Failures cells={allCells} failMode={state.failMode} onFailMode={(m) => dispatch({ type: 'FAIL_MODE', mode: m })} onSelect={handleSelect} />;
      case 'calibration':
        return (
          <Calibration
            cells={allCells}
            warnings={state.payload?.warnings}
            lintRunTs={state.payload?.lintRunTs}
          />
        );
      case 'viz':
        return <Viz cells={allCells} onSelect={handleSelect} />;
      case 'throughput':
        return <Throughput cells={allCells} />;
      case 'rlvr':
        return <RLVRPanel cells={allCells} />;
      case 'audit':
        return <Audit cells={allCells} seed={selected} />;
    }
  };

  const paletteActions = [
    { label: 'Export Markdown', key: 'export-md', action: onExportMd },
    { label: 'Export JSON', key: 'export-json', action: onExportJson },
    { label: 'Copy Summary', key: 'copy-sum', action: onCopySummary },
    { label: 'Reconnect WebSocket', key: 'reconnect', action: connectWS },
  ];

  return (
    <div className="app-layout">
      <aside className="sidebar">
        <SummaryBar
          state={state}
          cells={allCells.length}
          filteredCount={cells.length}
          onChangeView={(v) => dispatch({ type: 'SET_VIEW', view: v })}
          onSearch={(s) => dispatch({ type: 'SET_SEARCH', search: s })}
          onReconnect={connectWS}
          onExportMd={onExportMd}
          onExportJson={onExportJson}
          onCopySummary={onCopySummary}
          onOpenPalette={() => setPaletteOpen(true)}
          wsStatus={!!state.payload}
          statusText={statusText}
          statusLevel={statusLevel}
        />
      </aside>

      <main className="main-panel">
        <div className="top-fixed">
          <VerdictStrip 
            summary={summary ? { stock: summary.by_variant.stock, ours: summary.by_variant.ours } : { stock: {}, ours: {} }} 
            statusText={statusText} 
            statusLevel={statusLevel} 
          />
          
          {(insights.length > 0 || (state.payload?.warnings?.length ?? 0) > 0) && (
            <div className="insights-strip">
              {state.payload?.warnings?.filter(w => w.severity === 'error').slice(0, 1).map((w, i) => (
                <button key={`lint-err-${i}`} className="insight-pill insight-bad" onClick={() => dispatch({ type: 'SET_VIEW', view: 'calibration' })}>
                  LINT: {w.code} ({w.cells?.length ?? 0} cell(s))
                </button>
              ))}
              {state.payload?.warnings?.filter(w => w.severity === 'warning').slice(0, 1).map((w, i) => (
                <button key={`lint-warn-${i}`} className="insight-pill insight-warn" onClick={() => dispatch({ type: 'SET_VIEW', view: 'calibration' })}>
                  LINT: {w.code} ({w.cells?.length ?? 0} cell(s))
                </button>
              ))}
              {insights.filter(i => !state.dismissedInsights.has(i.kind)).map(i => (
                <button key={i.kind} className={`insight-pill insight-${i.level}`} onClick={() => onInsightAction(i)}>
                  {i.text}
                </button>
              ))}
            </div>
          )}

          {diff && (diff.added > 0 || diff.removed > 0 || diff.changed > 0) && (
            <div className="diff-bar">
              {diff.added > 0 && <span className="diff-add">+{diff.added}</span>}
              {diff.removed > 0 && <span className="diff-rem">-{diff.removed}</span>}
              {diff.changed > 0 && <span className="diff-met">{diff.changed} changed</span>}
            </div>
          )}
        </div>

        <div className="view-scroll">
          {renderContent()}
        </div>
      </main>

      <Drawer
        cell={selected}
        paired={pairedCell}
        onClose={() => setSelected(null)}
        onAudit={(c) => {
          setSelected(c);
          dispatch({ type: 'SET_VIEW', view: 'audit' });
        }}
      />
      
      <CommandPalette 
        isOpen={paletteOpen}
        views={['overview', 'suites', 'cells', 'comparison', 'failures', 'calibration', 'viz', 'throughput', 'rlvr', 'audit']}
        actions={paletteActions}
        cells={cells}
        onSelect={(item) => {
          if (item.type === 'view') dispatch({ type: 'SET_VIEW', view: item.id });
          else if (item.type === 'action') item.action();
          else if (item.type === 'cell') { handleSelect(item.data); }
          setPaletteOpen(false);
        }}
        onClose={() => setPaletteOpen(false)}
      />

      <ToastContainer toasts={toasts} onDismiss={(id) => setToasts(prev => prev.filter(t => t.id !== id))} />
    </div>
  );
}
