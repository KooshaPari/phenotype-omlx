import React, { useState } from 'react';
import { ViewType } from '../types';
import { BenchState } from '../state/useBenchState';

interface SummaryBarProps {
  state: BenchState;
  cells: number;
  filteredCount: number;
  onChangeView: (v: ViewType) => void;
  onSearch: (s: string) => void;
  onReconnect: () => void;
  onExportMd: () => void;
  onExportJson: () => void;
  onCopySummary: () => void;
  onOpenPalette: () => void;
  wsStatus: boolean;
  statusText: string;
  statusLevel: string;
}

const VIEWS: { id: ViewType; label: string; icon: string; key: string }[] = [
  { id: 'overview', label: 'Overview', icon: '◈', key: '1' },
  { id: 'suites', label: 'Suites', icon: '▦', key: '2' },
  { id: 'cells', label: 'Cells', icon: '☰', key: '3' },
  { id: 'comparison', label: 'Compare', icon: '⇔', key: '4' },
  { id: 'failures', label: 'Fails', icon: '⚠', key: '5' },
  { id: 'calibration', label: 'Calib', icon: '⌀', key: '6' },
  { id: 'viz', label: 'Viz', icon: '▦', key: '7' },
  { id: 'throughput', label: 'Thru', icon: '↗', key: '8' },
  { id: 'rlvr', label: 'RLVR', icon: '∑', key: '9' },
  { id: 'audit', label: 'Audit', icon: '⌕', key: '0' },
];

export default function SummaryBar({ 
  state, 
  cells, 
  filteredCount, 
  onChangeView, 
  onSearch, 
  onReconnect,
  onExportMd,
  onExportJson,
  onCopySummary,
  onOpenPalette,
  wsStatus,
  statusText,
  statusLevel
}: SummaryBarProps) {
  const model = state.payload?.data?.summary?.meta?.model || '—';

  return (
    <div className="sidebar-content">
      <div className="sb-header">
        <div className="sb-brand">
          <span className="sb-logo">⚡</span>
          <span className="sb-app-title">Bench</span>
        </div>
      </div>

      <div className="sb-section">
        <div className="sb-pill model-pill">{model}</div>
        <div className="sb-stats">
          <span>{cells} total</span>
          {filteredCount !== cells && <span className="faint"> · {filteredCount} shown</span>}
        </div>
      </div>

      <div className="sb-connection">
        <span className={`status-dot ${wsStatus ? 'connected' : 'disconnected'}`} />
        <span className="status-text" data-level={statusLevel}>{statusText}</span>
      </div>

      <div className="sb-search-wrapper">
        <input
          className="sb-search"
          type="text"
          placeholder="Search... ⌘K"
          value={state.search}
          onChange={e => onSearch(e.target.value)}
          onClick={onOpenPalette}
          readOnly
        />
      </div>

      <nav className="sb-nav">
        {VIEWS.map(v => (
          <button
            key={v.id}
            className={`sb-nav-item ${state.view === v.id ? 'active' : ''}`}
            onClick={() => onChangeView(v.id)}
          >
            <span className="nav-icon">{v.icon}</span>
            <span className="nav-label">{v.label}</span>
            <span className="nav-key">{v.key}</span>
          </button>
        ))}
      </nav>

      <div className="sb-actions">
        <button className="sb-action-btn" onClick={onExportMd}>Export MD</button>
        <button className="sb-action-btn" onClick={onExportJson}>Export JSON</button>
        <button className="sb-action-btn" onClick={onCopySummary}>Copy Summary</button>
        <button className="sb-action-btn reconnect" onClick={onReconnect}>↻ Reconnect</button>
      </div>
    </div>
  );
}
