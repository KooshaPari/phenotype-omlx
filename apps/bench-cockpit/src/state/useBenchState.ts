import { useReducer, useCallback, useRef, useEffect, useMemo } from 'react';
import type {
  BenchPayload,
  Cell,
  CellFilters,
  DiffInfo,
  FailMode,
  GroupBy,
  HistoryEntry,
  Insight,
  SortDir,
  SummaryData,
  TrendPoint,
  ViewType,
} from '../types';

/* ─── Cell status helper ─────────────────────────────────────────── */

export function cellStatus(c: Cell): 'ok' | 'fail' | 'timeout' {
  if (c.wall_clock_s >= 59 && (c.tokens_per_second || 0) === 0) return 'timeout';
  if (!c.ok || c.partial_credit < 0.5) return 'fail';
  return 'ok';
}

/* ─── State shape ────────────────────────────────────────────────── */

export interface BenchState {
  payload: BenchPayload | null;
  view: ViewType;
  search: string;
  sortKey: string;
  sortDir: SortDir;
  groupBy: GroupBy;
  failMode: FailMode;
  scatterScale: number;
  selectedCell: Cell | null;
  filters: CellFilters;
  dismissedInsights: Set<string>;
}

/* ─── Actions ────────────────────────────────────────────────────── */

type Action =
  | { type: 'SET_PAYLOAD'; payload: BenchPayload }
  | { type: 'SET_VIEW'; view: ViewType }
  | { type: 'SET_SEARCH'; search: string }
  | { type: 'SORT'; key: string }
  | { type: 'GROUP'; group: GroupBy }
  | { type: 'FAIL_MODE'; mode: FailMode }
  | { type: 'SELECT_CELL'; cell: Cell | null }
  | { type: 'SET_FILTER'; filterKey: keyof CellFilters; value: string }
  | { type: 'SET_SCALE'; scale: number }
  | { type: 'DISMISS_INSIGHT'; kind: string };

/* ─── Initial state (read from URL hash where possible) ──────────── */

function parseHash(): Partial<BenchState> {
  if (typeof window === 'undefined') return {};
  const h = window.location.hash.slice(1);
  if (!h) return {};
  try {
    const p = new URLSearchParams(h);
    const out: Partial<BenchState> = {};
    const view = p.get('view') as ViewType | null;
    if (view) out.view = view;
    const q = p.get('q');
    if (q) out.search = q;
    const sort = p.get('sort');
    if (sort) out.sortKey = sort;
    const dir = p.get('dir');
    if (dir) out.sortDir = Number(dir) as SortDir;
    const group = p.get('group') as GroupBy | null;
    if (group) out.groupBy = group;
    const fail = p.get('fail') as FailMode | null;
    if (fail) out.failMode = fail;
    const scale = p.get('scale');
    if (scale) out.scatterScale = Number(scale);
    return out;
  } catch {
    return {};
  }
}

const h = parseHash();

const initialVariantSet = new Set<string>(['stock', 'ours']);

const initialState: BenchState = {
  payload: null,
  view: h.view ?? 'overview',
  search: h.search ?? '',
  sortKey: h.sortKey ?? 'wall_clock_s',
  sortDir: (h.sortDir as SortDir) ?? -1,
  groupBy: h.groupBy ?? 'none',
  failMode: h.failMode ?? 'all',
  scatterScale: h.scatterScale ?? 1,
  selectedCell: null,
  filters: {
    suite: new Set<string>(),
    difficulty: new Set<string>(),
    variant: new Set(initialVariantSet),
  },
  dismissedInsights: new Set<string>(),
};

/* ─── Reducer ────────────────────────────────────────────────────── */

function reducer(state: BenchState, action: Action): BenchState {
  switch (action.type) {
    case 'SET_PAYLOAD':
      return { ...state, payload: action.payload };

    case 'SET_VIEW':
      return { ...state, view: action.view };

    case 'SET_SEARCH':
      return { ...state, search: action.search };

    case 'SORT': {
      const dir: SortDir = state.sortKey === action.key ? (-(state.sortDir) as SortDir) : -1;
      return { ...state, sortKey: action.key, sortDir: dir };
    }

    case 'GROUP':
      return { ...state, groupBy: action.group };

    case 'FAIL_MODE':
      return { ...state, failMode: action.mode };

    case 'SELECT_CELL':
      return { ...state, selectedCell: action.cell };

    case 'SET_FILTER': {
      const prev = state.filters[action.filterKey];
      const next = new Set(prev);
      if (next.has(action.value)) next.delete(action.value);
      else next.add(action.value);
      return { ...state, filters: { ...state.filters, [action.filterKey]: next } };
    }

    case 'SET_SCALE':
      return { ...state, scatterScale: action.scale };

    case 'DISMISS_INSIGHT': {
      const ds = new Set(state.dismissedInsights);
      ds.add(action.kind);
      return { ...state, dismissedInsights: ds };
    }

    default:
      return state;
  }
}

/* ─── Hook ───────────────────────────────────────────────────────── */

export function useBenchState() {
  const [state, dispatch] = useReducer(reducer, initialState);
  const wsRef = useRef<WebSocket | null>(null);
  const attemptsRef = useRef(0);
  const prevCellsRef = useRef<Map<string, Cell>>(new Map());

  /* ── History ring buffer (last 30 pushes) ──────────────────────── */
  const historyRef = useRef<HistoryEntry[]>([]);

  const history: HistoryEntry[] = historyRef.current;

  /* ── Push new payload → update history ─────────────────────────── */
  useEffect(() => {
    if (!state.payload) return;
    const { summary, cells } = state.payload.data;
    const entry: HistoryEntry = {
      receivedAt: new Date().toISOString(),
      summary,
      cellCount: cells.length,
    };
    historyRef.current = [...historyRef.current.slice(-29), entry];
  }, [state.payload]);

  /* ── Diff: compare current cells with previous push ────────────── */
  const diff: DiffInfo = useMemo(() => {
    const cells = state.payload?.data?.cells ?? [];
    const prevMap = prevCellsRef.current;
    const curMap = new Map(cells.map(c => [`${c.task_id}:${c.variant}`, c]));

    let added = 0;
    let removed = 0;
    let changed = 0;

    for (const [key, cur] of curMap) {
      const prev = prevMap.get(key);
      if (!prev) {
        added++;
      } else if (prev.wall_clock_s !== cur.wall_clock_s || prev.pass_at_1 !== cur.pass_at_1 || prev.ok !== cur.ok || prev.partial_credit !== cur.partial_credit) {
        changed++;
      }
    }
    for (const key of prevMap.keys()) {
      if (!curMap.has(key)) removed++;
    }

    // Update prev ref for next render cycle
    prevCellsRef.current = curMap;

    return {
      added,
      removed,
      changed,
      ts: state.payload?.serverTs ?? '',
    };
  }, [state.payload]);

  /* ── Insights: auto-detect from current data ───────────────────── */
  const insights: Insight[] = useMemo(() => {
    if (!state.payload) return [];
    const { summary, cells } = state.payload.data;
    const { stock, ours } = summary.by_variant;
    const results: Insight[] = [];

    // --- wins-on-hard: ours beats stock on hard+ difficulty
    const hardOurs = cells.filter(c => c.variant === 'ours' && (c.difficulty === 'hard' || c.difficulty === 'ultra'));
    const hardStock = cells.filter(c => c.variant === 'stock' && (c.difficulty === 'hard' || c.difficulty === 'ultra'));
    const oursPass = hardOurs.filter(c => c.pass_at_1 >= 1).length;
    const stockPass = hardStock.filter(c => c.pass_at_1 >= 1).length;
    if (hardOurs.length > 0 && oursPass > stockPass && stockPass < hardStock.length * 0.5) {
      results.push({
        kind: 'wins-on-hard',
        level: 'good',
        text: `Ours passes ${oursPass}/${hardOurs.length} hard tasks vs stock's ${stockPass}/${hardStock.length}.`,
        jumpTo: 'cells',
      });
    }

    // --- losses-on-ultra: stock beats ours on ultra
    const ultraOurs = cells.filter(c => c.variant === 'ours' && c.difficulty === 'ultra');
    const ultraStock = cells.filter(c => c.variant === 'stock' && c.difficulty === 'ultra');
    const uoPass = ultraOurs.filter(c => c.pass_at_1 >= 1).length;
    const usPass = ultraStock.filter(c => c.pass_at_1 >= 1).length;
    if (ultraOurs.length > 0 && usPass > uoPass) {
      results.push({
        kind: 'losses-on-ultra',
        level: 'warn',
        text: `Stock beats ours on ultra: ${usPass}/${ultraStock.length} vs ${uoPass}/${ultraOurs.length}.`,
        jumpTo: 'failures',
      });
    }

    // --- variance-ratio: if one variant's wall-clock std is 2x the other
    const stockWalls = hardStock.map(c => c.wall_clock_s);
    const oursWalls = hardOurs.map(c => c.wall_clock_s);
    const variance = (arr: number[]) => {
      if (arr.length < 2) return 0;
      const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
      return arr.reduce((s, v) => s + (v - mean) ** 2, 0) / (arr.length - 1);
    };
    const sVar = variance(stockWalls);
    const oVar = variance(oursWalls);
    if (sVar > 0 && oVar > 0) {
      const ratio = Math.max(sVar / oVar, oVar / sVar);
      if (ratio >= 4) {
        const moreVaried = sVar > oVar ? 'stock' : 'ours';
        results.push({
          kind: 'variance-ratio',
          level: 'bad',
          text: `${moreVaried} variant has ${(ratio).toFixed(1)}x wall-clock variance on hard tasks — unstable performance.`,
        });
      }
    }

    // --- suite-sweep: ours wins every suite
    const suites = [...new Set(cells.map(c => c.suite))];
    if (suites.length >= 2) {
      const oursWinsAll = suites.every(s => {
        const oAvg = cells.filter(c => c.variant === 'ours' && c.suite === s).reduce((a, c) => a + c.pass_at_1, 0) / Math.max(1, cells.filter(c => c.variant === 'ours' && c.suite === s).length);
        const sAvg = cells.filter(c => c.variant === 'stock' && c.suite === s).reduce((a, c) => a + c.pass_at_1, 0) / Math.max(1, cells.filter(c => c.variant === 'stock' && c.suite === s).length);
        return oAvg > sAvg;
      });
      if (oursWinsAll) {
        results.push({
          kind: 'suite-sweep',
          level: 'good',
          text: `Ours leads on all ${suites.length} suites.`,
        });
      }
    }

    return results;
  }, [state.payload]);

  /* ── Trends: sparkline data from history ring buffer ────────────── */
  const trends = useMemo(() => {
    const entries = historyRef.current;
    if (entries.length < 2) return null;
    const metrics: Record<string, TrendPoint[]> = {};

    const pushPoint = (metric: string, stockVal: number, oursVal: number, ts: string) => {
      if (!metrics[metric]) metrics[metric] = [];
      metrics[metric].push({ stock: stockVal, ours: oursVal, ts });
    };

    for (const entry of entries) {
      const ts = entry.receivedAt;
      const s = entry.summary.by_variant.stock;
      const o = entry.summary.by_variant.ours;
      pushPoint('pass_at_1', s.pass_at_1, o.pass_at_1, ts);
      pushPoint('wall_clock', s.mean_wall_clock_s, o.mean_wall_clock_s, ts);
      pushPoint('partial_credit', s.mean_partial_credit, o.mean_partial_credit, ts);
      pushPoint('cost', s.mean_cost_usd, o.mean_cost_usd, ts);
      pushPoint('hallucinations', s.n_hallucinations, o.n_hallucinations, ts);
      pushPoint('tokens_read', s.mean_tokens_read, o.mean_tokens_read, ts);
      pushPoint('first_token_ms', s.mean_first_token_ms, o.mean_first_token_ms, ts);
    }

    return metrics;
  }, [history.length]);

  /* ── Paired cell lookup ────────────────────────────────────────── */
  const pairedCell: Cell | null = useMemo(() => {
    const sel = state.selectedCell;
    if (!sel || !state.payload) return null;
    return (
      state.payload.data.cells.find(
        c => c.task_id === sel.task_id && c.variant !== sel.variant,
      ) ?? null
    );
  }, [state.selectedCell, state.payload]);

  /* ── Filtered cells ────────────────────────────────────────────── */
  const filteredCells = useCallback((): Cell[] => {
    if (!state.payload) return [];
    const { filters, search } = state;
    const q = search.toLowerCase();
    return state.payload.data.cells.filter(c => {
      if (!filters.variant.has(c.variant)) return false;
      if (filters.suite.size && !filters.suite.has(c.suite)) return false;
      if (filters.difficulty.size && !filters.difficulty.has(c.difficulty)) return false;
      if (q) {
        const hay = [
          c.task_id, c.suite, c.difficulty, c.task_type,
          c.reply ?? '', c.prompt ?? '', c.error_message ?? '',
        ].join(' ').toLowerCase();
        if (!hay.includes(q)) return false;
      }
      return true;
    });
  }, [state]);

  /* ── WebSocket connection with exponential backoff ─────────────── */
  const connectWS = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    // Use /api/ws — NOT /ws — so Vite HMR on :5173 does not steal the socket.
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${proto}//${window.location.host}/api/ws`;

    let ws: WebSocket;
    try {
      ws = new WebSocket(url);
    } catch {
      scheduleReconnect();
      return;
    }

    ws.onopen = () => {
      attemptsRef.current = 0;
    };

    ws.onmessage = (ev) => {
      try {
        dispatch({ type: 'SET_PAYLOAD', payload: JSON.parse(ev.data) });
      } catch {
        /* ignore malformed frames */
      }
    };

    ws.onerror = () => {
      ws.close();
    };

    ws.onclose = () => {
      scheduleReconnect();
    };

    wsRef.current = ws;
  }, []);

  /** HTTP bootstrap so the UI is usable even if WS proxy flaps. */
  const fetchState = useCallback(async () => {
    try {
      const r = await fetch('/api/state');
      if (!r.ok) return;
      const payload = await r.json();
      dispatch({ type: 'SET_PAYLOAD', payload });
    } catch {
      /* ignore; WS / next poll will retry */
    }
  }, []);

  useEffect(() => {
    void fetchState();
    const id = window.setInterval(() => {
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
        void fetchState();
      }
    }, 5000);
    return () => window.clearInterval(id);
  }, [fetchState]);

  const scheduleReconnect = useCallback(() => {
    const delay = Math.min(8000, 500 * Math.pow(1.6, attemptsRef.current++));
    setTimeout(connectWS, delay);
  }, [connectWS]);

  /* ── URL hash persistence ──────────────────────────────────────── */
  useEffect(() => {
    const p = new URLSearchParams();
    p.set('view', state.view);
    if (state.search) p.set('q', state.search);
    if (state.sortKey !== 'wall_clock_s') p.set('sort', state.sortKey);
    if (state.sortDir !== -1) p.set('dir', String(state.sortDir));
    if (state.groupBy !== 'none') p.set('group', state.groupBy);
    if (state.failMode !== 'all') p.set('fail', state.failMode);
    if (state.scatterScale !== 1) p.set('scale', String(state.scatterScale));
    window.location.hash = p.toString();
  }, [state.view, state.search, state.sortKey, state.sortDir, state.groupBy, state.failMode, state.scatterScale]);

  // Listen for external hash changes
  useEffect(() => {
    const handleHashChange = () => {
      const newHash = parseHash();
      // Only dispatch if hash is different from current state to avoid loops
      if (newHash.view && newHash.view !== state.view) dispatch({ type: 'SET_VIEW', view: newHash.view });
      if (newHash.search !== undefined && newHash.search !== state.search) dispatch({ type: 'SET_SEARCH', search: newHash.search });
    };
    window.addEventListener('hashchange', handleHashChange);
    return () => window.removeEventListener('hashchange', handleHashChange);
  }, [state.view, state.search, dispatch]);

  /* ── Public API ────────────────────────────────────────────────── */
  return {
    state,
    dispatch,
    connectWS,
    filteredCells,
    cellStatus,
    history,
    diff,
    insights,
    trends,
    pairedCell,
  } as const;
}
