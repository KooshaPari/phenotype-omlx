package main

import (
	"encoding/json"
	"testing"
)

func TestCellUnmarshalJSON_RlvrPassthroughRoundTrip(t *testing.T) {
	raw := []byte(`{
		"suite":"s","task_id":"t","variant":"ours",
		"pass_at_1":1.0,
		"rlvr_composite":0.82,
		"rlvr_l0":1.0,"rlvr_l1":0.8,"rlvr_l2":0.7,"rlvr_l3":0.6,
		"rlvr_reward":0.82,
		"rlvr_reward_breakdown":{"l0":1.0,"l1":0.8,"l2":0.7,"l3":0.6,"json":1.0},
		"rlvr_passed":true,
		"rlvr_verifiable":true,
		"rlvr_tournament_delta":0.12
	}`)
	var c Cell
	if err := json.Unmarshal(raw, &c); err != nil {
		t.Fatal(err)
	}
	if c.RlvrComposite != 0.82 || c.RlvrReward != 0.82 {
		t.Fatalf("composite=%v reward=%v", c.RlvrComposite, c.RlvrReward)
	}
	if c.RlvrL0 != 1 || c.RlvrL1 != 0.8 || c.RlvrL2 != 0.7 || c.RlvrL3 != 0.6 {
		t.Fatalf("layers l0=%v l1=%v l2=%v l3=%v", c.RlvrL0, c.RlvrL1, c.RlvrL2, c.RlvrL3)
	}
	if !c.RlvrPassed || !c.RlvrVerifiable || c.RlvrTournamentDelta != 0.12 {
		t.Fatalf("passed=%v verifiable=%v delta=%v", c.RlvrPassed, c.RlvrVerifiable, c.RlvrTournamentDelta)
	}
	if c.RlvrRewardBreakdown["json"] != 1 {
		t.Fatalf("breakdown=%v", c.RlvrRewardBreakdown)
	}

	out, err := json.Marshal(c)
	if err != nil {
		t.Fatal(err)
	}
	var again Cell
	if err := json.Unmarshal(out, &again); err != nil {
		t.Fatal(err)
	}
	if again.RlvrComposite != 0.82 || again.RlvrL1 != 0.8 || !again.RlvrVerifiable {
		t.Fatalf("round-trip lost fields: %#v", again)
	}
	if again.RlvrRewardBreakdown["l3"] != 0.6 {
		t.Fatalf("round-trip breakdown=%v", again.RlvrRewardBreakdown)
	}
}

func TestCellUnmarshalJSON_RlvrAliasDualRead(t *testing.T) {
	raw := []byte(`{
		"suite":"s","task_id":"t","variant":"stock",
		"RLVRReward":0.55,
		"RLVRRewardBreakdown":{"l0":0.9,"l1":0.5,"l2":0.4,"l3":0.3},
		"RLVRPassed":true,
		"RLVRVerifiable":true,
		"RLVRTournamentDelta":-0.05
	}`)
	var c Cell
	if err := json.Unmarshal(raw, &c); err != nil {
		t.Fatal(err)
	}
	if c.RlvrComposite != 0.55 || c.RlvrReward != 0.55 {
		t.Fatalf("composite/reward from RLVRReward: %v / %v", c.RlvrComposite, c.RlvrReward)
	}
	if c.RlvrL0 != 0.9 || c.RlvrL3 != 0.3 {
		t.Fatalf("layers from breakdown: l0=%v l3=%v", c.RlvrL0, c.RlvrL3)
	}
	if !c.RlvrPassed || !c.RlvrVerifiable || c.RlvrTournamentDelta != -0.05 {
		t.Fatalf("alias bools/delta: passed=%v verifiable=%v delta=%v",
			c.RlvrPassed, c.RlvrVerifiable, c.RlvrTournamentDelta)
	}
}

func TestCellUnmarshalJSON_HarnessMixedBreakdown(t *testing.T) {
	// Real pheno-harness dry-run cell shape: numeric L0–L3 plus string metadata
	// inside rlvr_reward_breakdown. Must not fail loud on import.
	raw := []byte(`{
		"suite":"mmlu-pro","task_id":"mmlu-pro-task-000","variant":"baseline_mlx",
		"pass_at_1":0.0,
		"rlvr_composite":0.5531,
		"rlvr_l0":0.6667,"rlvr_l1":0.4125,"rlvr_l2":0.0,"rlvr_l3":1.0,
		"rlvr_reward":0.5531,
		"rlvr_reward_breakdown":{
			"l0":0.6667,"l1":0.4125,"l2":0.0,"l3":1.0,
			"soft_partial":0.35,"process_bonus":0.65,
			"heuristic_reward":0.5,
			"heuristic_reason":"expected 'ok' not found in synthetic answer",
			"verifier_total":0.25,
			"motion":"forward"
		},
		"rlvr_passed":false,
		"rlvr_verifiable":true,
		"rlvr_tournament_delta":0.0
	}`)
	var c Cell
	if err := json.Unmarshal(raw, &c); err != nil {
		t.Fatalf("harness mixed breakdown must unmarshal: %v", err)
	}
	if c.RlvrComposite != 0.5531 || c.RlvrReward != 0.5531 {
		t.Fatalf("composite=%v reward=%v", c.RlvrComposite, c.RlvrReward)
	}
	if c.RlvrL0 != 0.6667 || c.RlvrL1 != 0.4125 || c.RlvrL3 != 1.0 {
		t.Fatalf("layers l0=%v l1=%v l2=%v l3=%v", c.RlvrL0, c.RlvrL1, c.RlvrL2, c.RlvrL3)
	}
	if c.RlvrRewardBreakdown["l0"] != 0.6667 || c.RlvrRewardBreakdown["soft_partial"] != 0.35 {
		t.Fatalf("numeric breakdown=%v", c.RlvrRewardBreakdown)
	}
	if _, ok := c.RlvrRewardBreakdown["heuristic_reason"]; ok {
		t.Fatalf("string metadata must be dropped from float map, got %v", c.RlvrRewardBreakdown)
	}
	if _, ok := c.RlvrRewardBreakdown["motion"]; ok {
		t.Fatalf("string motion must be dropped from float map, got %v", c.RlvrRewardBreakdown)
	}
	if !c.RlvrVerifiable {
		t.Fatal("expected rlvr_verifiable=true")
	}
	if c.RlvrPassed {
		t.Fatal("expected rlvr_passed=false")
	}
}

func TestCellFromTask_RlvrFromAdditionalProperties(t *testing.T) {
	ap := map[string]interface{}{
		"pass_at_1":       1.0,
		"rlvr_composite":  0.91,
		"rlvr_l0":         1.0,
		"rlvr_l1":         0.95,
		"rlvr_l2":         0.9,
		"rlvr_l3":         0.8,
		"rlvr_passed":     true,
		"rlvr_verifiable": true,
		"rlvr_reward_breakdown": map[string]interface{}{
			"json": 1.0, "tests": 0.95,
		},
	}
	tr := evalTaskResult{TaskID: "t1", Status: "ok", Judge: "deterministic", RawScore: 1}
	cell := cellFromTask("suite", "ours", evalReport{}, tr, ap)
	if cell.RlvrComposite != 0.91 || cell.RlvrReward != 0.91 {
		t.Fatalf("composite=%v reward=%v", cell.RlvrComposite, cell.RlvrReward)
	}
	if cell.RlvrL1 != 0.95 || !cell.RlvrPassed || !cell.RlvrVerifiable {
		t.Fatalf("l1=%v passed=%v verifiable=%v", cell.RlvrL1, cell.RlvrPassed, cell.RlvrVerifiable)
	}
	if cell.RlvrRewardBreakdown["json"] != 1 {
		t.Fatalf("breakdown=%v", cell.RlvrRewardBreakdown)
	}
}
