import type { ComponentType, CSSProperties } from 'react';
import ReactECharts from 'echarts-for-react';

/** echarts-for-react types lag React 19 — cast once for cockpit charts. */
export const EChart = ReactECharts as unknown as ComponentType<{
  option: Record<string, unknown>;
  style?: CSSProperties;
  opts?: { renderer?: string };
  onEvents?: Record<string, (p: { value?: unknown; data?: unknown; name?: string }) => void>;
}>;
