import { describe, expect, test } from 'bun:test';
import { resolveRlvr } from './rlvr';

describe('resolveRlvr', () => {
  test('fromHarness wins when rlvr_* fields present (source === harness)', () => {
    const r = resolveRlvr({
      pass_at_1: 0,
      format_compliance_rate: 0,
      partial_credit: 0,
      wall_clock_s: 99,
      tokens_per_second: 1,
      rlvr_composite: 0.88,
      rlvr_l0: 1,
      rlvr_l1: 0.9,
      rlvr_l2: 0.8,
      rlvr_l3: 0.7,
      rlvr_passed: true,
      rlvr_verifiable: true,
      rlvr_tournament_delta: 0.15,
      rlvr_reward_breakdown: { json: 1, tests: 0.9 },
    });
    expect(r.source).toBe('harness');
    expect(r.composite).toBe(0.88);
    expect(r.l0).toBe(1);
    expect(r.l1).toBe(0.9);
    expect(r.l2).toBe(0.8);
    expect(r.l3).toBe(0.7);
    expect(r.passed).toBe(true);
    expect(r.verifiable).toBe(true);
    expect(r.tournamentDelta).toBe(0.15);
    expect(r.breakdown.json).toBe(1);
  });

  test('falls back to derived when no harness or trace RLVR', () => {
    const r = resolveRlvr({
      pass_at_1: 1,
      format_compliance_rate: 1,
      partial_credit: 1,
    });
    expect(r.source).toBe('derived');
  });
});
