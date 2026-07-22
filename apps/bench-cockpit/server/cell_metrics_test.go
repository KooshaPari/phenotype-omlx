package main

import (
	"encoding/json"
	"testing"
)

func TestCellUnmarshalJSON_GenOkFallback(t *testing.T) {
	raw := []byte(`{"suite":"s","task_id":"t","variant":"stock","pass_at_1":0.42}`)
	var c Cell
	if err := json.Unmarshal(raw, &c); err != nil {
		t.Fatal(err)
	}
	if c.GenOk != 0.42 {
		t.Fatalf("GenOk=%v want 0.42 from pass_at_1 fallback", c.GenOk)
	}
	if c.PassAt1 != 0.42 {
		t.Fatalf("PassAt1=%v want 0.42", c.PassAt1)
	}
}

func TestCellUnmarshalJSON_ExplicitGenOk(t *testing.T) {
	raw := []byte(`{"suite":"s","task_id":"t","variant":"stock","pass_at_1":0.9,"gen_ok":0.1}`)
	var c Cell
	if err := json.Unmarshal(raw, &c); err != nil {
		t.Fatal(err)
	}
	if c.GenOk != 0.1 {
		t.Fatalf("GenOk=%v want explicit 0.1", c.GenOk)
	}
}

func TestCellUnmarshalJSON_VerifiedPass(t *testing.T) {
	raw := []byte(`{"suite":"s","task_id":"t","variant":"ours","pass_at_1":1,"gen_ok":1,"verified_pass_at_1":0.75}`)
	var c Cell
	if err := json.Unmarshal(raw, &c); err != nil {
		t.Fatal(err)
	}
	if c.VerifiedPassAt1 != 0.75 {
		t.Fatalf("VerifiedPassAt1=%v want 0.75", c.VerifiedPassAt1)
	}
	if v, ok := cellVerifiedPass(c); !ok || v != 0.75 {
		t.Fatalf("cellVerifiedPass=%v ok=%v", v, ok)
	}
}

func TestLoadData_SmokeFixtureDualRead(t *testing.T) {
	path := "../fixtures/smoke_results.json"
	old := dataPath
	dataPath = path
	t.Cleanup(func() { dataPath = old })

	data, err := loadData()
	if err != nil {
		t.Fatalf("loadData: %v", err)
	}
	if len(data.Cells) == 0 {
		t.Fatal("expected cells")
	}
	for i, c := range data.Cells {
		if c.GenOk != c.PassAt1 {
			t.Fatalf("cell[%d] GenOk=%v PassAt1=%v want equal on legacy fixture", i, c.GenOk, c.PassAt1)
		}
	}
	stock := data.Summary.ByVariant["stock"]
	if stock.GenOk != stock.PassAt1 {
		t.Fatalf("summary stock GenOk=%v PassAt1=%v", stock.GenOk, stock.PassAt1)
	}
}
