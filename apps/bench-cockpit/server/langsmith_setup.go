package main

import (
	"bytes"
	"crypto/rand"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
)

const (
	defaultLSProject = "bench-cockpit"
	defaultLSDataset = "bench-cockpit-v5-cells"
)

type lsSetupRequest struct {
	ProjectName string `json:"project_name"`
	DatasetName string `json:"dataset_name"`
	MaxCells    int    `json:"max_cells"`
	// SeedRuns defaults to true when omitted.
	SeedRuns *bool `json:"seed_runs"`
}

type lsSetupResult struct {
	Enabled      bool              `json:"enabled"`
	ProjectID    string            `json:"project_id,omitempty"`
	ProjectName  string            `json:"project_name,omitempty"`
	DatasetID    string            `json:"dataset_id,omitempty"`
	DatasetName  string            `json:"dataset_name,omitempty"`
	ExperimentID string            `json:"experiment_id,omitempty"`
	Examples     int               `json:"examples_uploaded"`
	Runs         int               `json:"runs_posted"`
	DashboardURL string            `json:"dashboard_url,omitempty"`
	Errors       []string          `json:"errors,omitempty"`
	Meta         map[string]string `json:"meta,omitempty"`
}

func lsNewID() string {
	b := make([]byte, 16)
	_, _ = rand.Read(b)
	b[6] = (b[6] & 0x0f) | 0x40
	b[8] = (b[8] & 0x3f) | 0x80
	return fmt.Sprintf("%x-%x-%x-%x-%x", b[0:4], b[4:6], b[6:8], b[8:10], b[10:])
}

func lsProjectName() string {
	if v := strings.TrimSpace(os.Getenv("LANGSMITH_PROJECT")); v != "" {
		return v
	}
	return defaultLSProject
}

func lsDatasetName() string {
	if v := strings.TrimSpace(os.Getenv("LANGSMITH_DATASET")); v != "" {
		return v
	}
	return defaultLSDataset
}

func lsJSON(method, path string, body any) (int, []byte, error) {
	var rdr io.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			return 0, nil, err
		}
		rdr = bytes.NewReader(b)
	}
	resp, err := langsmithProxy(method, path, rdr)
	if err != nil {
		return 0, nil, err
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	return resp.StatusCode, raw, err
}

func lsFindSessionByName(name string) (map[string]any, error) {
	code, raw, err := lsJSON(http.MethodGet, "/sessions?limit=100", nil)
	if err != nil {
		return nil, err
	}
	if code >= 300 {
		return nil, fmt.Errorf("list sessions %d: %s", code, truncate(string(raw), 200))
	}
	var list []map[string]any
	if err := json.Unmarshal(raw, &list); err != nil {
		return nil, err
	}
	for _, s := range list {
		if fmt.Sprint(s["name"]) == name {
			return s, nil
		}
	}
	return nil, nil
}

func lsFindDatasetByName(name string) (map[string]any, error) {
	code, raw, err := lsJSON(http.MethodGet, "/datasets?limit=100", nil)
	if err != nil {
		return nil, err
	}
	if code >= 300 {
		return nil, fmt.Errorf("list datasets %d: %s", code, truncate(string(raw), 200))
	}
	var list []map[string]any
	if err := json.Unmarshal(raw, &list); err != nil {
		return nil, err
	}
	for _, d := range list {
		if fmt.Sprint(d["name"]) == name {
			return d, nil
		}
	}
	return nil, nil
}

func lsEnsureProject(name, description string) (id string, created bool, err error) {
	if existing, err := lsFindSessionByName(name); err != nil {
		return "", false, err
	} else if existing != nil {
		return fmt.Sprint(existing["id"]), false, nil
	}
	code, raw, err := lsJSON(http.MethodPost, "/sessions?upsert=true", map[string]any{
		"name":        name,
		"description": description,
	})
	if err != nil {
		return "", false, err
	}
	if code >= 300 {
		return "", false, fmt.Errorf("create project %d: %s", code, truncate(string(raw), 240))
	}
	var out map[string]any
	if err := json.Unmarshal(raw, &out); err != nil {
		return "", false, err
	}
	return fmt.Sprint(out["id"]), true, nil
}

func lsEnsureDataset(name, description string) (id string, created bool, err error) {
	if existing, err := lsFindDatasetByName(name); err != nil {
		return "", false, err
	} else if existing != nil {
		return fmt.Sprint(existing["id"]), false, nil
	}
	code, raw, err := lsJSON(http.MethodPost, "/datasets", map[string]any{
		"name":        name,
		"description": description,
		"data_type":   "kv",
	})
	if err != nil {
		return "", false, err
	}
	if code >= 300 {
		return "", false, fmt.Errorf("create dataset %d: %s", code, truncate(string(raw), 240))
	}
	var out map[string]any
	if err := json.Unmarshal(raw, &out); err != nil {
		return "", false, err
	}
	return fmt.Sprint(out["id"]), true, nil
}

func pickCellsForLS(cells []Cell, max int) []Cell {
	if max <= 0 {
		max = 40
	}
	seen := map[string]int{}
	out := make([]Cell, 0, max)
	for _, c := range cells {
		key := c.Suite + "|" + c.Variant
		if seen[key] >= 2 {
			continue
		}
		seen[key]++
		out = append(out, c)
		if len(out) >= max {
			break
		}
	}
	return out
}

func lsUploadExamples(datasetID string, cells []Cell) (ids []string, errs []string) {
	for _, c := range cells {
		body := map[string]any{
			"dataset_id": datasetID,
			"inputs": map[string]any{
				"suite":      c.Suite,
				"task_id":    c.TaskID,
				"variant":    c.Variant,
				"difficulty": c.Difficulty,
				"prompt":     truncate(c.Prompt, 2000),
			},
			"outputs": map[string]any{
				"pass_at_1":                c.PassAt1,
				"partial_credit":           c.PartialCredit,
				"judge_score":              c.JudgeScore,
				"wall_clock_s":             c.WallClockS,
				"tokens_per_second":        c.TokensPerSecond,
				"reply":                    truncate(c.Reply, 2000),
				"format_compliance_rate":   c.FormatCompliance,
				"intent_preservation_rate": c.IntentPreservation,
			},
			"metadata": map[string]any{
				"model_name": c.ModelName,
				"source":     "bench-cockpit",
			},
		}
		code, raw, err := lsJSON(http.MethodPost, "/examples", body)
		if err != nil {
			errs = append(errs, err.Error())
			continue
		}
		if code >= 300 {
			errs = append(errs, fmt.Sprintf("example %s: %d %s", c.TaskID, code, truncate(string(raw), 120)))
			continue
		}
		var out map[string]any
		if err := json.Unmarshal(raw, &out); err == nil {
			if id := fmt.Sprint(out["id"]); id != "" && id != "<nil>" {
				ids = append(ids, id)
			}
		}
	}
	return ids, errs
}

func lsPostRun(sessionID, sessionName, exampleID string, c Cell) error {
	now := time.Now().UTC().Format(time.RFC3339Nano)
	body := map[string]any{
		"id":       lsNewID(),
		"name":     c.Suite + "/" + c.TaskID,
		"run_type": "chain",
		"inputs": map[string]any{
			"suite":      c.Suite,
			"task_id":    c.TaskID,
			"variant":    c.Variant,
			"difficulty": c.Difficulty,
		},
		"outputs": map[string]any{
			"pass_at_1":         c.PassAt1,
			"partial_credit":    c.PartialCredit,
			"tokens_per_second": c.TokensPerSecond,
			"wall_clock_s":      c.WallClockS,
			"reply":             truncate(c.Reply, 1500),
		},
		"extra": map[string]any{
			"metadata": map[string]any{
				"variant": c.Variant,
				"source":  "bench-cockpit",
			},
		},
		"start_time": now,
		"end_time":   now,
		"session_id": sessionID,
	}
	if sessionName != "" {
		body["session_name"] = sessionName
	}
	if exampleID != "" {
		body["reference_example_id"] = exampleID
	}
	code, raw, err := lsJSON(http.MethodPost, "/runs", body)
	if err != nil {
		return err
	}
	if code != 200 && code != 201 && code != 202 {
		return fmt.Errorf("run %d: %s", code, truncate(string(raw), 160))
	}
	return nil
}

func runLangSmithSetup(req lsSetupRequest, cells []Cell) lsSetupResult {
	out := lsSetupResult{
		Enabled: langsmithEnabled(),
		Meta:    map[string]string{},
	}
	if !out.Enabled {
		out.Errors = append(out.Errors, "LANGSMITH_API_KEY not set")
		return out
	}

	project := req.ProjectName
	if project == "" {
		project = lsProjectName()
	}
	dataset := req.DatasetName
	if dataset == "" {
		dataset = lsDatasetName()
	}
	maxCells := req.MaxCells
	if maxCells <= 0 {
		maxCells = 40
	}
	doSeed := true
	if req.SeedRuns != nil {
		doSeed = *req.SeedRuns
	}

	out.ProjectName = project
	out.DatasetName = dataset

	pid, createdP, err := lsEnsureProject(project, "Phenotype omlx bench-cockpit stock-vs-ours traces")
	if err != nil {
		out.Errors = append(out.Errors, err.Error())
		return out
	}
	out.ProjectID = pid
	out.Meta["project_created"] = fmt.Sprintf("%v", createdP)

	did, createdD, err := lsEnsureDataset(dataset, "Bench cockpit cell dataset (inputs=task, outputs=metrics)")
	if err != nil {
		out.Errors = append(out.Errors, err.Error())
		return out
	}
	out.DatasetID = did
	out.Meta["dataset_created"] = fmt.Sprintf("%v", createdD)

	sample := pickCellsForLS(cells, maxCells)
	exIDs, exErrs := lsUploadExamples(did, sample)
	out.Examples = len(exIDs)
	out.Errors = append(out.Errors, exErrs...)

	if doSeed && len(sample) > 0 {
		expName := fmt.Sprintf("bench-seed-%s", time.Now().UTC().Format("20060102-150405"))
		code, raw, err := lsJSON(http.MethodPost, "/sessions?upsert=true", map[string]any{
			"name":                 expName,
			"description":          "Seeded stock-vs-ours runs from bench-cockpit",
			"reference_dataset_id": did,
			"start_time":           time.Now().UTC().Format(time.RFC3339Nano),
			"extra":                map[string]any{"metadata": map[string]any{"source": "bench-cockpit", "kind": "seed"}},
		})
		if err != nil {
			out.Errors = append(out.Errors, err.Error())
		} else if code >= 300 {
			out.Errors = append(out.Errors, fmt.Sprintf("create experiment %d: %s", code, truncate(string(raw), 160)))
		} else {
			var exp map[string]any
			_ = json.Unmarshal(raw, &exp)
			out.ExperimentID = fmt.Sprint(exp["id"])
		}

		runs := 0
		for i, c := range sample {
			if err := lsPostRun(pid, project, "", c); err != nil {
				out.Errors = append(out.Errors, err.Error())
			} else {
				runs++
			}
			if out.ExperimentID != "" && i < len(exIDs) {
				if err := lsPostRun(out.ExperimentID, "", exIDs[i], c); err != nil {
					out.Errors = append(out.Errors, err.Error())
				} else {
					runs++
				}
			}
		}
		out.Runs = runs
		if out.ExperimentID != "" {
			_, _, _ = lsJSON(http.MethodPatch, "/sessions/"+out.ExperimentID, map[string]any{
				"end_time": time.Now().UTC().Format(time.RFC3339Nano),
			})
		}
	}

	out.DashboardURL = "https://smith.langchain.com/projects/p/" + out.ProjectID
	return out
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n]
}

func langsmithStatusHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	if !langsmithEnabled() {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"enabled": false,
			"error":   "langsmith_disabled",
		})
		return
	}
	code, raw, err := lsJSON(http.MethodGet, "/sessions?limit=50", nil)
	if err != nil {
		http.Error(w, fmt.Sprintf(`{"error":%q}`, err.Error()), http.StatusBadGateway)
		return
	}
	var sessions any
	_ = json.Unmarshal(raw, &sessions)
	dcode, draw, _ := lsJSON(http.MethodGet, "/datasets?limit=50", nil)
	var datasets any
	if dcode < 300 {
		_ = json.Unmarshal(draw, &datasets)
	}
	_ = json.NewEncoder(w).Encode(map[string]any{
		"enabled":       true,
		"project_name":  lsProjectName(),
		"dataset_name":  lsDatasetName(),
		"sessions":      sessions,
		"datasets":      datasets,
		"sessions_http": code,
	})
}

func langsmithSetupHandler(w http.ResponseWriter, r *http.Request) {
	if !langsmithEnabled() {
		langsmithDisabled(w)
		return
	}
	if r.Method != http.MethodPost {
		http.Error(w, `{"error":"POST required"}`, http.StatusMethodNotAllowed)
		return
	}
	var req lsSetupRequest
	if r.Body != nil {
		_ = json.NewDecoder(r.Body).Decode(&req)
	}
	if req.MaxCells == 0 {
		req.MaxCells = 40
	}
	if req.SeedRuns == nil {
		t := true
		req.SeedRuns = &t
	}

	env, err := buildEnvelope()
	cells := []Cell{}
	if err == nil && env.Data != nil {
		cells = env.Data.Cells
	}
	result := runLangSmithSetup(req, cells)
	w.Header().Set("Content-Type", "application/json")
	if len(result.Errors) > 0 && result.ProjectID == "" {
		w.WriteHeader(http.StatusBadGateway)
	}
	_ = json.NewEncoder(w).Encode(result)
}
