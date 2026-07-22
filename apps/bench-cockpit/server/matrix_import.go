package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// KnownSuiteCatalog is the expected bench surface (V5 stock-vs-ours + extended matrix).
// Suites absent from loaded cells show as gaps in /api/state suite_coverage.
var KnownSuiteCatalog = []string{
	"aider-polyglot",
	"aime",
	"arc-agi-2",
	"bfcl",
	"browsercomp",
	"deep-swe",
	"gpqa-diamond",
	"hle",
	"ifeval",
	"kernelbench",
	"livecodebench",
	"mmlu-pro",
	"mt-bench",
	"osworld",
	"perplexity",
	"pinchbench",
	"startup-bench",
	"swe-bench",
	"swe-bench-pro",
	"swe-bench-verified",
	"terminal-bench",
	"vending-bench",
	// Not implemented in pheno-harness yet — kept for coverage visibility.
	"ycbench",
}

type suiteCoverageRow struct {
	Suite        string         `json:"suite"`
	Present      bool           `json:"present"`
	Variants     map[string]int `json:"variants"`
	NCells       int            `json:"n_cells"`
	HasStock     bool           `json:"has_stock"`
	HasOurs      bool           `json:"has_ours"`
	ExperimentArms []string     `json:"experiment_arms"`
}

type matrixFile struct {
	Model   string `json:"model"`
	Suites  []struct {
		Suite       string  `json:"suite"`
		N           int     `json:"n"`
		Passed      int     `json:"passed"`
		PassAt1     float64 `json:"pass_at_1"`
		WallClockS  float64 `json:"wall_clock_s_total"`
		TaskResults []struct {
			TaskID       string  `json:"task_id"`
			OK           bool    `json:"ok"`
			WallClockS   float64 `json:"wall_clock_s"`
			ReplyPreview string  `json:"reply_preview"`
			Error        any     `json:"error"`
		} `json:"task_results"`
	} `json:"suites"`
}

func variantFromModel(model string) string {
	m := strings.ToLower(model)
	switch {
	case strings.Contains(m, "minimax"):
		return "minimax-m3"
	case strings.Contains(m, "qwen"):
		return "qwen-stock"
	default:
		slug := strings.Map(func(r rune) rune {
			if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') || r == '-' {
				return r
			}
			return '-'
		}, m)
		slug = strings.Trim(slug, "-")
		if slug == "" {
			return "external"
		}
		if len(slug) > 32 {
			slug = slug[:32]
		}
		return slug
	}
}

func cellsFromMatrixJSON(raw []byte, variantOverride string) ([]Cell, string, error) {
	var mf matrixFile
	if err := json.Unmarshal(raw, &mf); err != nil {
		return nil, "", err
	}
	if len(mf.Suites) == 0 {
		return nil, "", fmt.Errorf("matrix has no suites")
	}
	variant := variantOverride
	if variant == "" {
		variant = variantFromModel(mf.Model)
	}
	out := make([]Cell, 0, 64)
	for _, s := range mf.Suites {
		if len(s.TaskResults) == 0 {
			// Aggregate-only row → one synthetic cell so the suite appears.
			pass := s.PassAt1
			out = append(out, Cell{
				Suite:           s.Suite,
				TaskID:          "aggregate",
				Difficulty:      "medium",
				Variant:         variant,
				OK:              s.Passed > 0,
				WallClockS:      s.WallClockS,
				PassAt1:         pass,
				GenOk:           pass,
				PartialCredit:   pass,
				FormatCompliance: 1,
				ScoringMethod:   "matrix_aggregate",
				ModelName:       mf.Model,
				Reply:           fmt.Sprintf("matrix aggregate n=%d pass@1=%.3f", s.N, pass),
				Metadata: map[string]string{
					"source": "matrix.json",
					"n":      fmt.Sprintf("%d", s.N),
				},
			})
			continue
		}
		for _, tr := range s.TaskResults {
			pass := 0.0
			if tr.OK {
				pass = 1.0
			}
			out = append(out, Cell{
				Suite:            s.Suite,
				TaskID:           tr.TaskID,
				Difficulty:       "medium",
				Variant:          variant,
				OK:               tr.OK,
				WallClockS:       tr.WallClockS,
				PassAt1:          pass,
				GenOk:            pass,
				PartialCredit:    pass,
				FormatCompliance: 1,
				ScoringMethod:    "matrix_task",
				ModelName:        mf.Model,
				Reply:            tr.ReplyPreview,
				Metadata: map[string]string{
					"source": "matrix.json",
				},
			})
		}
	}
	return out, variant, nil
}

func looksLikeMatrix(raw []byte) bool {
	var probe map[string]json.RawMessage
	if err := json.Unmarshal(raw, &probe); err != nil {
		return false
	}
	_, hasSuites := probe["suites"]
	_, hasCells := probe["cells"]
	return hasSuites && !hasCells
}

func loadResultsFile(path string) (*ResultsData, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", path, err)
	}
	if looksLikeMatrix(raw) {
		cells, variant, err := cellsFromMatrixJSON(raw, "")
		if err != nil {
			return nil, fmt.Errorf("matrix %s: %w", path, err)
		}
		data := &ResultsData{Cells: cells}
		data.Summary.Meta.Model = variant
		data.Summary.Meta.Variants = []string{variant}
		data.Summary.Meta.NCells = len(cells)
		data.Summary.ByVariant = summarizeByVariant(cells)
		normalizeDualRead(data)
		return data, nil
	}
	if looksLikeEvaluationReport(raw) {
		data, err := resultsFromEvaluationReport(raw)
		if err != nil {
			return nil, err
		}
		normalizeDualRead(data)
		return data, nil
	}
	var data ResultsData
	if err := json.Unmarshal(raw, &data); err != nil {
		return nil, fmt.Errorf("unmarshal %s: %w", path, err)
	}
	if len(data.Cells) == 0 {
		return nil, fmt.Errorf("%s has 0 cells", path)
	}
	enrichVariantThroughput(&data)
	normalizeDualRead(&data)
	return &data, nil
}

func mergeResults(base *ResultsData, extras ...*ResultsData) *ResultsData {
	if base == nil {
		base = &ResultsData{Summary: Summary{ByVariant: map[string]VariantSummary{}}}
	}
	seen := make(map[string]struct{}, len(base.Cells))
	for _, c := range base.Cells {
		seen[c.Suite+"\x00"+c.TaskID+"\x00"+c.Variant] = struct{}{}
	}
	for _, ex := range extras {
		if ex == nil {
			continue
		}
		for _, c := range ex.Cells {
			key := c.Suite + "\x00" + c.TaskID + "\x00" + c.Variant
			if _, ok := seen[key]; ok {
				continue
			}
			seen[key] = struct{}{}
			base.Cells = append(base.Cells, c)
		}
	}
	base.Summary.ByVariant = summarizeByVariant(base.Cells)
	suites := map[string]struct{}{}
	variants := map[string]struct{}{}
	for _, c := range base.Cells {
		suites[c.Suite] = struct{}{}
		variants[c.Variant] = struct{}{}
	}
	base.Summary.Meta.NCells = len(base.Cells)
	base.Summary.Meta.NSuites = len(suites)
	vl := make([]string, 0, len(variants))
	for v := range variants {
		vl = append(vl, v)
	}
	base.Summary.Meta.Variants = vl
	// Model pill = ablation peers only (stock/ours). Aux arms (minimax judge/eval,
	// distillers, …) must not appear as peer model names joined with '+'.
	ablation := make([]string, 0, 2)
	for _, v := range []string{"stock", "ours"} {
		if _, ok := variants[v]; ok {
			ablation = append(ablation, v)
		}
	}
	if len(ablation) > 0 {
		base.Summary.Meta.Model = strings.Join(ablation, "+")
	} else if base.Summary.Meta.Model == "" && len(vl) > 0 {
		base.Summary.Meta.Model = strings.Join(vl, "+")
	}
	enrichVariantThroughput(base)
	return base
}

func parseExtraPaths(extraFlag string) []string {
	if strings.TrimSpace(extraFlag) == "" {
		if env := strings.TrimSpace(os.Getenv("BENCH_EXTRA_DATA")); env != "" {
			extraFlag = env
		}
	}
	if strings.TrimSpace(extraFlag) == "" {
		return nil
	}
	parts := strings.Split(extraFlag, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		if !filepath.IsAbs(p) {
			if abs, err := filepath.Abs(p); err == nil {
				p = abs
			}
		}
		if _, err := os.Stat(p); err == nil {
			out = append(out, p)
		}
	}
	return out
}

func buildSuiteCoverage(cells []Cell) []suiteCoverageRow {
	bySuite := map[string]map[string]int{}
	for _, c := range cells {
		if bySuite[c.Suite] == nil {
			bySuite[c.Suite] = map[string]int{}
		}
		bySuite[c.Suite][c.Variant]++
	}
	// Union catalog + present
	order := append([]string{}, KnownSuiteCatalog...)
	for s := range bySuite {
		found := false
		for _, k := range KnownSuiteCatalog {
			if k == s {
				found = true
				break
			}
		}
		if !found {
			order = append(order, s)
		}
	}
	rows := make([]suiteCoverageRow, 0, len(order))
	for _, suite := range order {
		variants := bySuite[suite]
		if variants == nil {
			variants = map[string]int{}
		}
		n := 0
		arms := make([]string, 0)
		for v, c := range variants {
			n += c
			if v != "stock" && v != "ours" {
				arms = append(arms, v)
			}
		}
		rows = append(rows, suiteCoverageRow{
			Suite:          suite,
			Present:        n > 0,
			Variants:       variants,
			NCells:         n,
			HasStock:       variants["stock"] > 0,
			HasOurs:        variants["ours"] > 0,
			ExperimentArms: arms,
		})
	}
	return rows
}
