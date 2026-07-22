package main

import (
	"encoding/json"
)

// cellGenOk returns generation-ok score (dual-read: gen_ok → pass_at_1).
func cellGenOk(c Cell) float64 {
	if c.dualReadGenOkSet {
		return c.GenOk
	}
	if c.GenOk != 0 {
		return c.GenOk
	}
	return c.PassAt1
}

// cellVerifiedPass returns verified pass when >0 or evidence is live-verified.
func cellVerifiedPass(c Cell) (float64, bool) {
	if c.VerifiedPassAt1 > 0 {
		return c.VerifiedPassAt1, true
	}
	if isVerifiedEvidence(c) && c.dualReadVerifiedSet {
		return c.VerifiedPassAt1, true
	}
	return 0, false
}

func isVerifiedEvidence(c Cell) bool {
	if c.Metadata == nil {
		return false
	}
	label := c.Metadata["evidence_label"]
	return label == "live_verified" || label == "verified"
}

// cellQualityPass prefers verified pass for aggregates when meaningful.
func cellQualityPass(c Cell) float64 {
	if v, ok := cellVerifiedPass(c); ok {
		return v
	}
	return cellGenOk(c)
}

// UnmarshalJSON dual-reads gen_ok from pass_at_1 when the new field is absent,
// and assignment/transcript aliases (nested assignment, chat_trace, reply_full, rubric).
func (c *Cell) UnmarshalJSON(data []byte) error {
	type cellWire Cell
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	var wire cellWire
	if err := json.Unmarshal(data, &wire); err != nil {
		return err
	}
	*c = Cell(wire)
	if _, ok := raw["gen_ok"]; !ok {
		c.GenOk = c.PassAt1
	} else {
		c.dualReadGenOkSet = true
	}
	if _, ok := raw["verified_pass_at_1"]; ok {
		c.dualReadVerifiedSet = true
	}
	ensureAssignmentFromRaw(c, raw)
	return nil
}

// UnmarshalJSON dual-reads gen_ok from pass_at_1 when the new field is absent.
func (v *VariantSummary) UnmarshalJSON(data []byte) error {
	type vsWire VariantSummary
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	var wire vsWire
	if err := json.Unmarshal(data, &wire); err != nil {
		return err
	}
	*v = VariantSummary(wire)
	if _, ok := raw["gen_ok"]; !ok {
		v.GenOk = v.PassAt1
	}
	return nil
}

func normalizeDualRead(data *ResultsData) {
	if data == nil {
		return
	}
	for i := range data.Cells {
		c := &data.Cells[i]
		if !c.dualReadGenOkSet {
			c.GenOk = c.PassAt1
		}
		normalizeAssignmentDualRead(c)
	}
	enriched := summarizeByVariant(data.Cells)
	if data.Summary.ByVariant == nil {
		data.Summary.ByVariant = enriched
		return
	}
	for variant, sum := range enriched {
		cur := data.Summary.ByVariant[variant]
		if cur.GenOk == 0 && cur.PassAt1 != 0 {
			cur.GenOk = cur.PassAt1
		}
		cur.GenOk = sum.GenOk
		cur.VerifiedPassAt1 = sum.VerifiedPassAt1
		data.Summary.ByVariant[variant] = cur
	}
}
