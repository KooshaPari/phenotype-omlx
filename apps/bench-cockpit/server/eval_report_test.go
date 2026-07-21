package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestResultsFromEvaluationReport_MiniCombined(t *testing.T) {
	path := filepath.Join("..", "fixtures", "eval_report_mini.json")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("fixture: %v", err)
	}
	if !looksLikeEvaluationReport(raw) {
		t.Fatal("mini fixture should look like EvaluationReport")
	}
	data, err := resultsFromEvaluationReport(raw)
	if err != nil {
		t.Fatalf("convert: %v", err)
	}
	if len(data.Cells) != 2 {
		t.Fatalf("cells=%d want 2", len(data.Cells))
	}
	if data.Summary.ByVariant["stock"].NCells != 1 || data.Summary.ByVariant["ours"].NCells != 1 {
		t.Fatalf("by_variant=%v", data.Summary.ByVariant)
	}
	if data.Cells[0].Variant != "stock" || data.Cells[1].Variant != "ours" {
		t.Fatalf("variants %q %q", data.Cells[0].Variant, data.Cells[1].Variant)
	}
	if data.Cells[0].TokensPerSecond != 40 || data.Cells[1].TokensPerSecond != 70 {
		t.Fatalf("tps %v %v", data.Cells[0].TokensPerSecond, data.Cells[1].TokensPerSecond)
	}
	warns := lintCells(data.Cells)
	found := false
	for _, w := range warns {
		if w.Code == "synthetic_100pct" {
			found = true
		}
	}
	if !found {
		t.Fatalf("expected synthetic_100pct, got %#v", warns)
	}
}

func TestLooksLikeEvaluationReport_SmokeFixture(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("..", "fixtures", "smoke_results.json"))
	if err != nil {
		t.Fatalf("fixture: %v", err)
	}
	if looksLikeEvaluationReport(raw) {
		t.Fatal("smoke fixture must not look like EvaluationReport")
	}
}

func TestResultsFromEvaluationReport_V5LiveOptional(t *testing.T) {
	if os.Getenv("BENCH_V5_LIVE") != "1" {
		t.Skip("set BENCH_V5_LIVE=1 to exercise full V5 artifact")
	}
	path := "/Users/kooshapari/CodeProjects/Phenotype/pheno-harness/bench/results/stock-vs-ours/run-v5-qwen35-08b-contract.json"
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	data, err := resultsFromEvaluationReport(raw)
	if err != nil {
		t.Fatalf("convert: %v", err)
	}
	if len(data.Cells) != 500 {
		t.Fatalf("cells=%d want 500", len(data.Cells))
	}
}
