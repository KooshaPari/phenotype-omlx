package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestCellsFromMatrixJSON_FR_SUITE_001(t *testing.T) {
	raw := []byte(`{
	  "model": "accounts/fireworks/models/minimax-m3",
	  "suites": [{
	    "suite": "hle",
	    "n": 2,
	    "passed": 1,
	    "pass_at_1": 0.5,
	    "wall_clock_s_total": 10,
	    "task_results": [
	      {"task_id": "t0", "ok": true, "wall_clock_s": 4, "reply_preview": "a"},
	      {"task_id": "t1", "ok": false, "wall_clock_s": 6, "reply_preview": "b"}
	    ]
	  }]
	}`)
	cells, variant, err := cellsFromMatrixJSON(raw, "")
	if err != nil {
		t.Fatal(err)
	}
	if variant != "minimax-m3" {
		t.Fatalf("variant=%q", variant)
	}
	if len(cells) != 2 {
		t.Fatalf("cells=%d", len(cells))
	}
	if cells[0].Suite != "hle" || cells[0].Variant != "minimax-m3" {
		t.Fatalf("cell0=%+v", cells[0])
	}
	if cells[0].PassAt1 != 1 || cells[1].PassAt1 != 0 {
		t.Fatalf("pass values %v %v", cells[0].PassAt1, cells[1].PassAt1)
	}
}

func TestMergeResultsAndCoverage_FR_SUITE_002(t *testing.T) {
	base := &ResultsData{Cells: []Cell{
		{Suite: "arc-agi-2", TaskID: "a", Variant: "stock", PassAt1: 1, GenOk: 1},
		{Suite: "arc-agi-2", TaskID: "a", Variant: "ours", PassAt1: 0, GenOk: 0},
	}}
	extra := &ResultsData{Cells: []Cell{
		{Suite: "hle", TaskID: "h0", Variant: "minimax-m3", PassAt1: 1, GenOk: 1},
		{Suite: "arc-agi-2", TaskID: "a", Variant: "stock", PassAt1: 0, GenOk: 0}, // duplicate skip
	}}
	merged := mergeResults(base, extra)
	if len(merged.Cells) != 3 {
		t.Fatalf("merged cells=%d", len(merged.Cells))
	}
	cov := buildSuiteCoverage(merged.Cells)
	byName := map[string]suiteCoverageRow{}
	for _, r := range cov {
		byName[r.Suite] = r
	}
	if !byName["arc-agi-2"].HasStock || !byName["arc-agi-2"].HasOurs {
		t.Fatalf("arc coverage %+v", byName["arc-agi-2"])
	}
	if !byName["hle"].Present || byName["hle"].HasStock {
		t.Fatalf("hle coverage %+v", byName["hle"])
	}
	if _, ok := byName["ycbench"]; ok {
		t.Fatal("ycbench was deferred out of KnownSuiteCatalog")
	}
	for _, d := range DeferredSuites {
		if d == "ycbench" {
			return
		}
	}
	t.Fatal("DeferredSuites should list ycbench")
}

func TestLoadResultsFileMatrix_FR_SUITE_003(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "matrix.json")
	raw := map[string]any{
		"model": "minimax-m3",
		"suites": []map[string]any{
			{
				"suite": "pinchbench", "n": 1, "passed": 0, "pass_at_1": 0,
				"wall_clock_s_total": 1,
				"task_results": []map[string]any{
					{"task_id": "p0", "ok": false, "wall_clock_s": 1, "reply_preview": "x"},
				},
			},
		},
	}
	b, _ := json.Marshal(raw)
	if err := os.WriteFile(p, b, 0o644); err != nil {
		t.Fatal(err)
	}
	data, err := loadResultsFile(p)
	if err != nil {
		t.Fatal(err)
	}
	if len(data.Cells) != 1 || data.Cells[0].Suite != "pinchbench" {
		t.Fatalf("%+v", data.Cells)
	}
}
