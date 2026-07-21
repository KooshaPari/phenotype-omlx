package main

import (
	"encoding/json"
	"fmt"
	"math"
)

// EvaluationReport (pheno-harness contract_version 0.1) shapes — subset
// needed to flatten suites/task_results into cockpit Cell rows.

type evalReport struct {
	ContractVersion string          `json:"contract_version"`
	ArtifactKind    string          `json:"artifact_kind"`
	Run             evalRun         `json:"run"`
	Suites          []evalSuite     `json:"suites"`
	Totals          evalTotals      `json:"totals"`
	Matrix          json.RawMessage `json:"matrix"`
}

type evalRun struct {
	RunID         string `json:"run_id"`
	Variant       string `json:"variant"`
	Model         string `json:"model"`
	EvidenceLabel string `json:"evidence_label"`
	ExecutedBy    string `json:"executed_by"`
	Command       string `json:"command"`
}

type evalTotals struct {
	Cells         int     `json:"cells"`
	Passed        int     `json:"passed"`
	PassAt1       float64 `json:"pass_at_1"`
	EvidenceLabel string  `json:"evidence_label"`
}

type evalSuite struct {
	Suite        string          `json:"suite"`
	N            int             `json:"n"`
	Passed       int             `json:"passed"`
	PassAt1      float64         `json:"pass_at_1"`
	EvidenceLabel string         `json:"evidence_label"`
	TaskResults  []evalTaskResult `json:"task_results"`
}

type evalTaskResult struct {
	TaskID              string                 `json:"task_id"`
	Status              string                 `json:"status"`
	Judge               string                 `json:"judge"`
	WallClockS          float64                `json:"wall_clock_s"`
	TokensIn            int                    `json:"tokens_in"`
	TokensOut           int                    `json:"tokens_out"`
	FirstTokenLatencyS  float64                `json:"first_token_latency_s"`
	EnergyJoules        float64                `json:"energy_joules"`
	RawScore            float64                `json:"raw_score"`
	FailureReason       []string               `json:"failure_reason"`
	AdditionalProperties map[string]interface{} `json:"additionalProperties"`
}

func looksLikeEvaluationReport(raw []byte) bool {
	var probe struct {
		ContractVersion string `json:"contract_version"`
		Suites          []json.RawMessage `json:"suites"`
	}
	if err := json.Unmarshal(raw, &probe); err != nil {
		return false
	}
	return probe.ContractVersion != "" && len(probe.Suites) > 0
}

func resultsFromEvaluationReport(raw []byte) (*ResultsData, error) {
	var rep evalReport
	if err := json.Unmarshal(raw, &rep); err != nil {
		return nil, fmt.Errorf("eval report unmarshal: %w", err)
	}
	if len(rep.Suites) == 0 {
		return nil, fmt.Errorf("eval report has no suites")
	}

	suitePass := map[string]int{}
	cells := make([]Cell, 0, rep.Totals.Cells)
	difficulty := map[string]int{}
	variantsSeen := map[string]struct{}{}

	for _, suite := range rep.Suites {
		variant := resolveSuiteVariant(rep.Run.Variant, suite.Suite, suitePass[suite.Suite])
		suitePass[suite.Suite]++
		variantsSeen[variant] = struct{}{}

		for _, tr := range suite.TaskResults {
			ap := tr.AdditionalProperties
			if ap == nil {
				ap = map[string]interface{}{}
			}
			cell := cellFromTask(suite.Suite, variant, rep, tr, ap)
			diff := strOr(ap, "difficulty", "unknown")
			difficulty[diff]++
			cells = append(cells, cell)
		}
	}

	if len(cells) == 0 {
		return nil, fmt.Errorf("eval report produced 0 cells")
	}

	byVariant := summarizeByVariant(cells)
	variants := make([]string, 0, len(variantsSeen))
	for v := range variantsSeen {
		variants = append(variants, v)
	}
	// Stable-ish order: stock then ours then others
	variants = orderVariants(variants)

	model := rep.Run.Model
	if model == "" || model == "unknown" {
		if m := firstModelSlug(cells); m != "" {
			model = m
		}
	}

	nSuites := len(suitePass)
	nTasks := 0
	if nSuites > 0 {
		nTasks = len(cells) / nSuites
		if len(byVariant) > 1 {
			nTasks = len(cells) / (nSuites * len(byVariant))
		}
		// Prefer suite.N when uniform
		if rep.Suites[0].N > 0 {
			nTasks = rep.Suites[0].N
		}
	}

	return &ResultsData{
		Summary: Summary{
			Meta: Meta{
				Model:          model,
				MLXURL:         "direct",
				NSuites:        nSuites,
				NTasks:         nTasks,
				Variants:       variants,
				NCells:         len(cells),
				Difficulty:     difficulty,
			},
			ByVariant: byVariant,
		},
		Cells: cells,
	}, nil
}

// resolveSuiteVariant: first occurrence of a suite name uses run.variant;
// a second block with the same suite name (combined stock+ours contract)
// flips to the other A/B variant.
func resolveSuiteVariant(runVariant, suite string, priorCount int) string {
	base := runVariant
	if base == "" {
		base = "stock"
	}
	if priorCount == 0 {
		return base
	}
	if base == "stock" {
		return "ours"
	}
	if base == "ours" {
		return "stock"
	}
	return fmt.Sprintf("%s_%d", base, priorCount+1)
}

func cellFromTask(suite, variant string, rep evalReport, tr evalTaskResult, ap map[string]interface{}) Cell {
	passAt1 := floatOr(ap, "pass_at_1", tr.RawScore)
	ok := tr.Status == "ok" || passAt1 >= 0.999
	meta := map[string]string{
		"evidence_label": coalesce(rep.Run.EvidenceLabel, "reported"),
		"judge":          tr.Judge,
		"contract":       "evaluation_report_v0.1",
	}
	if syn, ok := ap["synthetic"].(bool); ok && syn {
		meta["synthetic"] = "true"
	}
	if slug, ok := ap["model_slug"].(string); ok && slug != "" {
		meta["model_slug"] = slug
	}
	if id, ok := ap["model_id"].(string); ok && id != "" {
		meta["model_id"] = id
	}

	errMsg := ""
	errCode := ""
	if !ok {
		errCode = "task_not_ok"
		if len(tr.FailureReason) > 0 {
			errMsg = tr.FailureReason[0]
		}
	}

	ftMs := tr.FirstTokenLatencyS * 1000.0
	if v, ok := ap["ttft_ms"].(float64); ok {
		ftMs = v
	}

	failAnalysis, _ := ap["failure_analysis"].(map[string]interface{})
	progress, _ := ap["progress_trace"].([]interface{})

	return Cell{
		Suite:               suite,
		TaskID:              tr.TaskID,
		Difficulty:          strOr(ap, "difficulty", "unknown"),
		Variant:             variant,
		OK:                  ok,
		WallClockS:          tr.WallClockS,
		TokensPerSecond:     floatOr(ap, "tokens_per_second", 0),
		FirstTokenLatencyMS: ftMs,
		PeakRSSMB:           floatOr(ap, "peak_rss_mb", 0),
		PeakGPUMemMB:        floatOr(ap, "peak_gpu_mem_mb", 0),
		EnergyProxyJoules:   tr.EnergyJoules,
		PassAt1:             passAt1,
		PartialCredit:       floatOr(ap, "partial_credit", passAt1),
		JudgeScore:          floatOr(ap, "judge_score", 0),
		IntentPreservation:  floatOr(ap, "intent_preservation_rate", 0),
		HallucinationCount:  intOr(ap, "hallucination_count", 0),
		ToolCallSuccess:     floatOr(ap, "tool_call_success_rate", 0),
		RetryCount:          intOr(ap, "retry_count", 0),
		FormatCompliance:    floatOr(ap, "format_compliance_rate", 0),
		Reply:               "",
		Prompt:              "",
		FailureAnalysis:     failAnalysis,
		ProgressTrace:       progress,
		ModelName:           strOr(ap, "model_id", rep.Run.Model),
		TotalTokensIn:       tr.TokensIn,
		TotalTokensOut:      tr.TokensOut,
		CostUSD:             floatOr(ap, "cost_per_1k_tokens_usd", 0),
		ErrorMessage:        errMsg,
		ErrorCode:           errCode,
		ExpectedAnswer:      strOr(ap, "expected_answer", ""),
		ScoringMethod:       strOr(ap, "scoring_method", tr.Judge),
		Metadata:            meta,
	}
}

func summarizeByVariant(cells []Cell) map[string]VariantSummary {
	type acc struct {
		n, ok, hall int
		pass, wall, partial, format, intent float64
	}
	m := map[string]*acc{}
	for _, c := range cells {
		a := m[c.Variant]
		if a == nil {
			a = &acc{}
			m[c.Variant] = a
		}
		a.n++
		if c.OK {
			a.ok++
		}
		a.pass += c.PassAt1
		a.wall += c.WallClockS
		a.partial += c.PartialCredit
		a.format += c.FormatCompliance
		a.intent += c.IntentPreservation
		a.hall += c.HallucinationCount
	}
	out := make(map[string]VariantSummary, len(m))
	for v, a := range m {
		n := float64(a.n)
		if n == 0 {
			continue
		}
		out[v] = VariantSummary{
			NCells:               a.n,
			PassAt1:              a.pass / n,
			MeanWallClockS:       a.wall / n,
			MeanPartialCredit:    a.partial / n,
			MeanFormatCompliance: a.format / n,
			MeanIntentPres:       a.intent / n,
			NHallucinations:      a.hall,
			OkCount:              a.ok,
		}
	}
	return out
}

func orderVariants(in []string) []string {
	pref := []string{"stock", "ours"}
	out := make([]string, 0, len(in))
	seen := map[string]bool{}
	for _, p := range pref {
		for _, v := range in {
			if v == p && !seen[v] {
				out = append(out, v)
				seen[v] = true
			}
		}
	}
	for _, v := range in {
		if !seen[v] {
			out = append(out, v)
			seen[v] = true
		}
	}
	return out
}

func firstModelSlug(cells []Cell) string {
	for _, c := range cells {
		if s := c.Metadata["model_slug"]; s != "" {
			return s
		}
		if s := c.Metadata["model_id"]; s != "" {
			return s
		}
	}
	return ""
}

func floatOr(m map[string]interface{}, key string, def float64) float64 {
	v, ok := m[key]
	if !ok || v == nil {
		return def
	}
	switch t := v.(type) {
	case float64:
		if math.IsNaN(t) || math.IsInf(t, 0) {
			return def
		}
		return t
	case int:
		return float64(t)
	case json.Number:
		f, err := t.Float64()
		if err != nil {
			return def
		}
		return f
	default:
		return def
	}
}

func intOr(m map[string]interface{}, key string, def int) int {
	v, ok := m[key]
	if !ok || v == nil {
		return def
	}
	switch t := v.(type) {
	case float64:
		return int(t)
	case int:
		return t
	default:
		return def
	}
}

func strOr(m map[string]interface{}, key, def string) string {
	v, ok := m[key]
	if !ok || v == nil {
		return def
	}
	if s, ok := v.(string); ok && s != "" {
		return s
	}
	return def
}

func coalesce(a, b string) string {
	if a != "" {
		return a
	}
	return b
}
