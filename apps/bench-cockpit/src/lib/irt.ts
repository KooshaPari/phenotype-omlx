/** Wilson score interval + stub IRT helpers for audit/calibration views. */

export interface WilsonCI {
  n: number;
  p: number;
  low: number;
  high: number;
}

/** Wilson score 95% CI for a binomial proportion. */
export function wilsonCI(successes: number, n: number, z = 1.96): WilsonCI {
  if (n <= 0) return { n: 0, p: 0, low: 0, high: 0 };
  const p = successes / n;
  const z2 = z * z;
  const denom = 1 + z2 / n;
  const centre = p + z2 / (2 * n);
  const margin = z * Math.sqrt((p * (1 - p) + z2 / (4 * n)) / n);
  return {
    n,
    p,
    low: Math.max(0, (centre - margin) / denom),
    high: Math.min(1, (centre + margin) / denom),
  };
}

export interface IRTStub {
  itemId: string;
  difficulty: number;
  discrimination: number;
  guessingFloor: number;
  ceiling: number;
}

/** Placeholder IRT from per-variant pass rates (PSN-IRT stub until enough models). */
export function stubIRT(itemId: string, passRates: number[]): IRTStub {
  const mean = passRates.length
    ? passRates.reduce((a, b) => a + b, 0) / passRates.length
    : 0.5;
  const variance =
    passRates.length > 1
      ? passRates.reduce((a, p) => a + (p - mean) ** 2, 0) / (passRates.length - 1)
      : 0;
  return {
    itemId,
    difficulty: 1 - mean,
    discrimination: Math.min(3, Math.sqrt(variance) * 4 + 0.5),
    guessingFloor: 0.1,
    ceiling: Math.min(1, mean + 0.15),
  };
}
