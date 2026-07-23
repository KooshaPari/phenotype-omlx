import { describe, expect, test } from 'bun:test';
import {
  deriveRlvr,
  formatRlvrScore,
  resolveRlvr,
  unavailableRlvr,
} from './rlvr';

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
    expect(r.authoritative).toBe(true);
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

  test('default: unavailable when no harness or trace RLVR (no silent derived)', () => {
    const r = resolveRlvr({
      pass_at_1: 1,
      format_compliance_rate: 1,
      partial_credit: 1,
      intent_preservation_rate: 1,
      judge_score: 1,
      hallucination_count: 0,
    });
    expect(r.source).toBe('unavailable');
    expect(r.authoritative).toBe(false);
    expect(r.verifiable).toBe(false);
    expect(r.passed).toBe(false);
    expect(Number.isNaN(r.composite)).toBe(true);
    expect(Number.isNaN(r.l0)).toBe(true);
    expect(Number.isNaN(r.l1)).toBe(true);
    expect(Object.keys(r.breakdown)).toHaveLength(0);
  });

  test('allowDerived opt-in synthesizes non-authoritative derived scores', () => {
    const r = resolveRlvr(
      {
        pass_at_1: 1,
        format_compliance_rate: 1,
        partial_credit: 1,
      },
      { allowDerived: true },
    );
    expect(r.source).toBe('derived');
    expect(r.authoritative).toBe(false);
    expect(r.verifiable).toBe(false);
    expect(Number.isFinite(r.composite)).toBe(true);
  });

  test('stub perfect quality metrics must not look like harness 100% without opt-in', () => {
    const silent = resolveRlvr({
      pass_at_1: 1,
      format_compliance_rate: 1,
      partial_credit: 1,
      intent_preservation_rate: 1,
      tool_call_success_rate: 1,
      judge_score: 1,
      hallucination_count: 0,
    });
    expect(silent.source).not.toBe('derived');
    expect(silent.source).not.toBe('harness');
    expect(silent.composite).not.toBe(1);
    expect(Number.isNaN(silent.composite)).toBe(true);

    const opted = resolveRlvr(
      {
        pass_at_1: 1,
        format_compliance_rate: 1,
        partial_credit: 1,
      },
      { allowDerived: true },
    );
    expect(opted.source).toBe('derived');
    expect(opted.authoritative).toBe(false);
  });

  test('harness still preferred over allowDerived', () => {
    const r = resolveRlvr(
      {
        rlvr_composite: 0.42,
        rlvr_l0: 0.4,
        format_compliance_rate: 1,
        pass_at_1: 1,
      },
      { allowDerived: true },
    );
    expect(r.source).toBe('harness');
    expect(r.composite).toBe(0.42);
  });

  test('trace reward preferred over derived/unavailable', () => {
    const r = resolveRlvr({
      pass_at_1: 1,
      progress_trace: [{ kind: 'reward', composite: 0.77, l0: 0.8, l1: 0.7, l2: 0.6, l3: 0.5 }],
    });
    expect(r.source).toBe('trace');
    expect(r.authoritative).toBe(true);
    expect(r.composite).toBe(0.77);
  });
});

describe('deriveRlvr / unavailableRlvr / formatRlvrScore', () => {
  test('deriveRlvr remains available for explicit debugging', () => {
    const r = deriveRlvr({
      pass_at_1: 1,
      format_compliance_rate: 1,
      partial_credit: 0.5,
    });
    expect(r.source).toBe('derived');
    expect(r.authoritative).toBe(false);
  });

  test('unavailableRlvr is NaN and non-authoritative', () => {
    const r = unavailableRlvr();
    expect(r.source).toBe('unavailable');
    expect(r.authoritative).toBe(false);
    expect(Number.isNaN(r.composite)).toBe(true);
  });

  test('formatRlvrScore renders em-dash for non-finite', () => {
    expect(formatRlvrScore(Number.NaN)).toBe('—');
    expect(formatRlvrScore(0.88)).toBe('0.880');
    expect(formatRlvrScore(0.88, 2)).toBe('0.88');
  });
});
