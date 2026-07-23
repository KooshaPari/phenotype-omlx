package main

import "encoding/json"

// applyRlvrFromAP dual-reads rlvr_* (and PascalCase aliases) from
// EvaluationReport additionalProperties into a Cell.
func applyRlvrFromAP(c *Cell, ap map[string]interface{}) {
	if c == nil || ap == nil {
		return
	}
	c.RlvrComposite = floatOrFirst(ap, 0, "rlvr_composite", "RLVRReward", "rlvr_reward")
	c.RlvrReward = floatOrFirst(ap, 0, "rlvr_reward", "RLVRReward", "rlvr_composite")
	c.RlvrL0 = floatOrFirst(ap, 0, "rlvr_l0")
	c.RlvrL1 = floatOrFirst(ap, 0, "rlvr_l1")
	c.RlvrL2 = floatOrFirst(ap, 0, "rlvr_l2")
	c.RlvrL3 = floatOrFirst(ap, 0, "rlvr_l3")
	c.RlvrTournamentDelta = floatOrFirst(ap, 0, "rlvr_tournament_delta", "RLVRTournamentDelta")
	if bd := floatMapOr(ap, "rlvr_reward_breakdown", "RLVRRewardBreakdown"); bd != nil {
		c.RlvrRewardBreakdown = bd
	}
	if v, ok := boolPresent(ap, "rlvr_passed", "RLVRPassed"); ok {
		c.RlvrPassed = v
	}
	if v, ok := boolPresent(ap, "rlvr_verifiable", "RLVRVerifiable"); ok {
		c.RlvrVerifiable = v
	}
	normalizeRlvrDualRead(c)
}

// ensureRlvrFromRaw fills rlvr_* from aliases when primary JSON keys are absent.
func ensureRlvrFromRaw(c *Cell, raw map[string]json.RawMessage) {
	if c == nil || raw == nil {
		return
	}
	ap := map[string]interface{}{}
	for _, key := range []string{
		"rlvr_composite", "rlvr_reward", "RLVRReward",
		"rlvr_l0", "rlvr_l1", "rlvr_l2", "rlvr_l3",
		"rlvr_reward_breakdown", "RLVRRewardBreakdown",
		"rlvr_passed", "RLVRPassed",
		"rlvr_verifiable", "RLVRVerifiable",
		"rlvr_tournament_delta", "RLVRTournamentDelta",
	} {
		if v, ok := raw[key]; ok {
			var any interface{}
			if err := json.Unmarshal(v, &any); err == nil {
				ap[key] = any
			}
		}
	}
	if len(ap) == 0 {
		return
	}
	// Prefer already-unmarshaled primary fields; fill gaps from aliases.
	if c.RlvrComposite == 0 {
		c.RlvrComposite = floatOrFirst(ap, 0, "rlvr_composite", "RLVRReward", "rlvr_reward")
	}
	if c.RlvrReward == 0 {
		c.RlvrReward = floatOrFirst(ap, 0, "rlvr_reward", "RLVRReward", "rlvr_composite")
	}
	if c.RlvrL0 == 0 {
		c.RlvrL0 = floatOrFirst(ap, 0, "rlvr_l0")
	}
	if c.RlvrL1 == 0 {
		c.RlvrL1 = floatOrFirst(ap, 0, "rlvr_l1")
	}
	if c.RlvrL2 == 0 {
		c.RlvrL2 = floatOrFirst(ap, 0, "rlvr_l2")
	}
	if c.RlvrL3 == 0 {
		c.RlvrL3 = floatOrFirst(ap, 0, "rlvr_l3")
	}
	if c.RlvrTournamentDelta == 0 {
		c.RlvrTournamentDelta = floatOrFirst(ap, 0, "rlvr_tournament_delta", "RLVRTournamentDelta")
	}
	if c.RlvrRewardBreakdown == nil {
		if bd := floatMapOr(ap, "rlvr_reward_breakdown", "RLVRRewardBreakdown"); bd != nil {
			c.RlvrRewardBreakdown = bd
		}
	}
	if _, ok := raw["rlvr_passed"]; !ok {
		if v, ok := boolPresent(ap, "RLVRPassed"); ok {
			c.RlvrPassed = v
		}
	}
	if _, ok := raw["rlvr_verifiable"]; !ok {
		if v, ok := boolPresent(ap, "RLVRVerifiable"); ok {
			c.RlvrVerifiable = v
		}
	}
	normalizeRlvrDualRead(c)
}

// normalizeRlvrDualRead mirrors composite↔reward and L0–L3↔breakdown.
func normalizeRlvrDualRead(c *Cell) {
	if c == nil {
		return
	}
	if c.RlvrComposite == 0 && c.RlvrReward != 0 {
		c.RlvrComposite = c.RlvrReward
	}
	if c.RlvrReward == 0 && c.RlvrComposite != 0 {
		c.RlvrReward = c.RlvrComposite
	}
	if c.RlvrRewardBreakdown != nil {
		if c.RlvrL0 == 0 {
			c.RlvrL0 = c.RlvrRewardBreakdown["l0"]
		}
		if c.RlvrL1 == 0 {
			c.RlvrL1 = c.RlvrRewardBreakdown["l1"]
		}
		if c.RlvrL2 == 0 {
			c.RlvrL2 = c.RlvrRewardBreakdown["l2"]
		}
		if c.RlvrL3 == 0 {
			c.RlvrL3 = c.RlvrRewardBreakdown["l3"]
		}
	}
}

func floatOrFirst(m map[string]interface{}, def float64, keys ...string) float64 {
	for _, key := range keys {
		if _, ok := m[key]; ok {
			return floatOr(m, key, def)
		}
	}
	return def
}

func floatMapOr(m map[string]interface{}, keys ...string) map[string]float64 {
	for _, key := range keys {
		v, ok := m[key]
		if !ok || v == nil {
			continue
		}
		switch t := v.(type) {
		case map[string]float64:
			return t
		case map[string]interface{}:
			out := make(map[string]float64, len(t))
			for k, raw := range t {
				switch n := raw.(type) {
				case float64:
					out[k] = n
				case int:
					out[k] = float64(n)
				case json.Number:
					f, err := n.Float64()
					if err == nil {
						out[k] = f
					}
				}
			}
			if len(out) > 0 {
				return out
			}
		}
	}
	return nil
}

func boolPresent(m map[string]interface{}, keys ...string) (bool, bool) {
	for _, key := range keys {
		v, ok := m[key]
		if !ok || v == nil {
			continue
		}
		switch t := v.(type) {
		case bool:
			return t, true
		case float64:
			return t != 0, true
		case int:
			return t != 0, true
		case string:
			if t == "true" || t == "1" {
				return true, true
			}
			if t == "false" || t == "0" {
				return false, true
			}
		}
	}
	return false, false
}
