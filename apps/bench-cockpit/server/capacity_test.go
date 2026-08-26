package main

import "testing"

func TestVramEstimateMistralApprox(t *testing.T) {
	b := vramEstimate(7_240_000_000, "F16")
	gb := float64(b) / 1e9
	if gb < 14.4 || gb > 14.6 {
		t.Fatalf("got %v GB", gb)
	}
}

func TestParamsFromModel(t *testing.T) {
	p, src := paramsFromModelName("qwen35-08b")
	if p != 800_000_000 || src == "" {
		t.Fatalf("got %d %s", p, src)
	}
	if !modelFitsIn(p, 24*1024*1024*1024, "F16") {
		t.Fatal("0.8B should fit 24GB F16")
	}
}
