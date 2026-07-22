package main

import "encoding/json"

// normalizeAssignmentDualRead fills assignment/transcript aliases on a Cell.
// Call after JSON unmarshal or EvaluationReport flatten.
// Priority: flat fields → nested assignment → cross-aliases (acceptance↔rubric,
// progress_trace↔chat_trace, reply↔reply_full).
func normalizeAssignmentDualRead(c *Cell) {
	if c == nil {
		return
	}
	if c.Acceptance == "" && c.Rubric != "" {
		c.Acceptance = c.Rubric
	}
	if c.Rubric == "" && c.Acceptance != "" {
		c.Rubric = c.Acceptance
	}
	if c.Assignment != nil {
		if c.TaskTitle == "" {
			c.TaskTitle = strFromAny(c.Assignment["title"])
		}
		if c.TaskDescription == "" {
			c.TaskDescription = strFromAny(c.Assignment["description"])
		}
		if c.Acceptance == "" {
			c.Acceptance = firstNonEmpty(
				strFromAny(c.Assignment["acceptance"]),
				strFromAny(c.Assignment["rubric"]),
			)
		}
		if c.Rubric == "" {
			c.Rubric = firstNonEmpty(
				strFromAny(c.Assignment["rubric"]),
				strFromAny(c.Assignment["acceptance"]),
				c.Acceptance,
			)
		}
	}
	if c.Acceptance == "" && c.Rubric != "" {
		c.Acceptance = c.Rubric
	}
	if c.Rubric == "" && c.Acceptance != "" {
		c.Rubric = c.Acceptance
	}
	if c.ReplyFull != "" && (c.Reply == "" || len(c.ReplyFull) > len(c.Reply)) {
		c.Reply = c.ReplyFull
	}
	if c.ReplyFull == "" && c.Reply != "" {
		c.ReplyFull = c.Reply
	}
	if len(c.ProgressTrace) == 0 && len(c.ChatTrace) > 0 {
		c.ProgressTrace = c.ChatTrace
	}
	if len(c.ChatTrace) == 0 && len(c.ProgressTrace) > 0 {
		c.ChatTrace = c.ProgressTrace
	}
}

func strFromAny(v interface{}) string {
	if v == nil {
		return ""
	}
	if s, ok := v.(string); ok {
		return s
	}
	return ""
}

func assignmentFromAP(ap map[string]interface{}) map[string]interface{} {
	if ap == nil {
		return nil
	}
	raw, ok := ap["assignment"]
	if !ok || raw == nil {
		return nil
	}
	if m, ok := raw.(map[string]interface{}); ok {
		return m
	}
	return nil
}

func interfaceSlice(ap map[string]interface{}, key string) []interface{} {
	if ap == nil {
		return nil
	}
	v, ok := ap[key]
	if !ok || v == nil {
		return nil
	}
	if s, ok := v.([]interface{}); ok {
		return s
	}
	return nil
}

// applyAssignmentFromAP dual-reads assignment/transcript fields from
// EvaluationReport additionalProperties into a Cell.
func applyAssignmentFromAP(c *Cell, ap map[string]interface{}) {
	if c == nil || ap == nil {
		return
	}
	c.Assignment = assignmentFromAP(ap)
	c.TaskTitle = strOr(ap, "task_title", "")
	c.TaskDescription = strOr(ap, "task_description", "")
	c.Acceptance = strOr(ap, "acceptance", "")
	c.Rubric = strOr(ap, "rubric", "")
	if p := strOr(ap, "prompt", ""); p != "" {
		c.Prompt = p
	}
	if r := strOr(ap, "reply_full", ""); r != "" {
		c.ReplyFull = r
		c.Reply = r
	} else if r := strOr(ap, "reply", ""); r != "" {
		c.Reply = r
		c.ReplyFull = r
	}
	progress := interfaceSlice(ap, "progress_trace")
	chat := interfaceSlice(ap, "chat_trace")
	if len(progress) > 0 {
		c.ProgressTrace = progress
	} else if len(chat) > 0 {
		c.ProgressTrace = chat
	}
	if len(chat) > 0 {
		c.ChatTrace = chat
	} else {
		c.ChatTrace = c.ProgressTrace
	}
	normalizeAssignmentDualRead(c)
}

// ensureAssignmentRaw detects presence of dual-read keys during UnmarshalJSON.
func ensureAssignmentFromRaw(c *Cell, raw map[string]json.RawMessage) {
	if _, ok := raw["assignment"]; ok && c.Assignment == nil {
		var asg map[string]interface{}
		if err := json.Unmarshal(raw["assignment"], &asg); err == nil {
			c.Assignment = asg
		}
	}
	normalizeAssignmentDualRead(c)
}
