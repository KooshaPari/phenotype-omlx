package main

import (
	"encoding/json"
	"testing"
)

func TestCellUnmarshalJSON_AssignmentNestedDualRead(t *testing.T) {
	raw := []byte(`{
		"suite":"terminal-bench","task_id":"tb-easy-00","variant":"stock",
		"pass_at_1":1.0,
		"assignment":{
			"title":"Echo date",
			"description":"Print the current date and time",
			"acceptance":"Output must include a datetime"
		},
		"chat_trace":[{"kind":"turn","turn":0,"role":"user","content":"echo date"}],
		"reply_full":"Sun Jul 21 18:00:00 PDT 2026"
	}`)
	var c Cell
	if err := json.Unmarshal(raw, &c); err != nil {
		t.Fatal(err)
	}
	if c.TaskTitle != "Echo date" {
		t.Fatalf("TaskTitle=%q", c.TaskTitle)
	}
	if c.TaskDescription != "Print the current date and time" {
		t.Fatalf("TaskDescription=%q", c.TaskDescription)
	}
	if c.Acceptance != "Output must include a datetime" {
		t.Fatalf("Acceptance=%q", c.Acceptance)
	}
	if c.Rubric != c.Acceptance {
		t.Fatalf("Rubric=%q want Acceptance dual-read", c.Rubric)
	}
	if c.Reply != "Sun Jul 21 18:00:00 PDT 2026" {
		t.Fatalf("Reply=%q want reply_full", c.Reply)
	}
	if len(c.ProgressTrace) != 1 {
		t.Fatalf("ProgressTrace len=%d want chat_trace dual-read", len(c.ProgressTrace))
	}
}

func TestCellUnmarshalJSON_FlatAssignmentOverridesNested(t *testing.T) {
	raw := []byte(`{
		"suite":"s","task_id":"t","variant":"ours",
		"task_title":"Flat",
		"task_description":"Flat desc",
		"rubric":"Rubric only",
		"prompt":"full prompt text",
		"reply":"short",
		"progress_trace":[{"kind":"llm","model":"smoke"}],
		"assignment":{"title":"Nested","description":"Nested desc","acceptance":"Nested accept"}
	}`)
	var c Cell
	if err := json.Unmarshal(raw, &c); err != nil {
		t.Fatal(err)
	}
	if c.TaskTitle != "Flat" {
		t.Fatalf("TaskTitle=%q want Flat", c.TaskTitle)
	}
	if c.TaskDescription != "Flat desc" {
		t.Fatalf("TaskDescription=%q", c.TaskDescription)
	}
	if c.Acceptance != "Rubric only" {
		t.Fatalf("Acceptance=%q want rubric fallback", c.Acceptance)
	}
	if c.Prompt != "full prompt text" {
		t.Fatalf("Prompt=%q", c.Prompt)
	}
	if len(c.ProgressTrace) != 1 {
		t.Fatalf("ProgressTrace len=%d", len(c.ProgressTrace))
	}
}

func TestCellFromTask_AssignmentFromAdditionalProperties(t *testing.T) {
	ap := map[string]interface{}{
		"pass_at_1":        1.0,
		"task_title":       "AP Title",
		"task_description": "AP Desc",
		"acceptance":       "Must pass tests",
		"prompt":           "implement foo",
		"reply":            "done",
		"progress_trace": []interface{}{
			map[string]interface{}{"kind": "turn", "role": "user", "content": "implement foo"},
		},
	}
	tr := evalTaskResult{TaskID: "t1", Status: "ok", Judge: "deterministic", RawScore: 1}
	cell := cellFromTask("suite", "stock", evalReport{}, tr, ap)
	if cell.TaskTitle != "AP Title" || cell.Acceptance != "Must pass tests" {
		t.Fatalf("title=%q acceptance=%q", cell.TaskTitle, cell.Acceptance)
	}
	if cell.Prompt != "implement foo" || cell.Reply != "done" {
		t.Fatalf("prompt=%q reply=%q", cell.Prompt, cell.Reply)
	}
	if cell.Rubric != "Must pass tests" {
		t.Fatalf("Rubric=%q", cell.Rubric)
	}
	if len(cell.ProgressTrace) != 1 || len(cell.ChatTrace) != 1 {
		t.Fatalf("trace lens progress=%d chat=%d", len(cell.ProgressTrace), len(cell.ChatTrace))
	}
}

func TestCellFromTask_ChatTraceFallback(t *testing.T) {
	ap := map[string]interface{}{
		"chat_trace": []interface{}{
			map[string]interface{}{"kind": "tool", "tool_name": "bash", "ok": true},
		},
		"assignment": map[string]interface{}{
			"title":       "Nested",
			"description": "Desc",
			"rubric":      "R",
		},
		"reply_full": "full reply body",
	}
	tr := evalTaskResult{TaskID: "t2", Status: "ok"}
	cell := cellFromTask("s", "ours", evalReport{}, tr, ap)
	if cell.TaskTitle != "Nested" || cell.Acceptance != "R" {
		t.Fatalf("title=%q acceptance=%q", cell.TaskTitle, cell.Acceptance)
	}
	if cell.Reply != "full reply body" {
		t.Fatalf("Reply=%q", cell.Reply)
	}
	if len(cell.ProgressTrace) != 1 {
		t.Fatalf("expected chat_trace → progress_trace, got %d", len(cell.ProgressTrace))
	}
}
