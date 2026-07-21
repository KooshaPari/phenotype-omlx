import type { ComponentType, CSSProperties } from 'react';
import { useEffect, useRef } from 'react';
import ReactECharts from 'echarts-for-react';

/** echarts-for-react types lag React 19 — cast once for cockpit charts. */
const ReactEChartsAny = ReactECharts as unknown as ComponentType<{
  option: Record<string, unknown>;
  style?: CSSProperties;
  opts?: { renderer?: string };
  onEvents?: Record<string, (p: { value?: unknown; data?: unknown; name?: string }) => void>;
  onChartReady?: (inst: { resize: () => void }) => void;
}>;

type Props = {
  option: Record<string, unknown>;
  style?: CSSProperties;
  opts?: { renderer?: string };
  onEvents?: Record<string, (p: { value?: unknown; data?: unknown; name?: string }) => void>;
};

/**
 * Thin wrapper: force resize after mount / option change so lazy Viz panels
 * do not paint a zero-size (all-black) canvas.
 */
export function EChart({ option, style, opts, onEvents }: Props) {
  const instRef = useRef<{ resize: () => void } | null>(null);

  useEffect(() => {
    const id = window.requestAnimationFrame(() => {
      instRef.current?.resize();
    });
    return () => window.cancelAnimationFrame(id);
  }, [option]);

  return (
    <ReactEChartsAny
      option={option}
      style={style}
      opts={opts ?? { renderer: 'canvas' }}
      onEvents={onEvents}
      onChartReady={(inst) => {
        instRef.current = inst;
        inst.resize();
      }}
    />
  );
}
