package main

import (
	"bytes"
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

func langfuseEnabled() bool {
	return strings.TrimSpace(os.Getenv("LANGFUSE_PUBLIC_KEY")) != "" &&
		strings.TrimSpace(os.Getenv("LANGFUSE_SECRET_KEY")) != ""
}

func langfuseBase() string {
	for _, k := range []string{"LANGFUSE_BASE_URL", "LANGFUSE_HOST"} {
		if v := strings.TrimSpace(os.Getenv(k)); v != "" {
			return strings.TrimRight(v, "/")
		}
	}
	return "https://cloud.langfuse.com"
}

func observabilityBackend() string {
	v := strings.ToLower(strings.TrimSpace(os.Getenv("OBSERVABILITY_BACKEND")))
	if v == "" {
		if langfuseEnabled() {
			return "langfuse"
		}
		if langsmithEnabled() {
			return "langsmith"
		}
		return "none"
	}
	return v
}

func langfuseAuthHeader() string {
	pub := strings.TrimSpace(os.Getenv("LANGFUSE_PUBLIC_KEY"))
	sec := strings.TrimSpace(os.Getenv("LANGFUSE_SECRET_KEY"))
	tok := base64.StdEncoding.EncodeToString([]byte(pub + ":" + sec))
	return "Basic " + tok
}

func langfuseDisabled(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusServiceUnavailable)
	_ = json.NewEncoder(w).Encode(map[string]string{"error": "langfuse_disabled"})
}

func langfuseDo(method, path string, body any) (int, []byte, error) {
	var rdr io.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			return 0, nil, err
		}
		rdr = bytes.NewReader(b)
	}
	req, err := http.NewRequest(method, langfuseBase()+path, rdr)
	if err != nil {
		return 0, nil, err
	}
	req.Header.Set("Authorization", langfuseAuthHeader())
	req.Header.Set("Content-Type", "application/json")
	client := &http.Client{Timeout: 60 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return 0, nil, err
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	return resp.StatusCode, raw, err
}

func newUUID() string {
	b := make([]byte, 16)
	_, _ = rand.Read(b)
	b[6] = (b[6] & 0x0f) | 0x40
	b[8] = (b[8] & 0x3f) | 0x80
	return fmt.Sprintf("%x-%x-%x-%x-%x", b[0:4], b[4:6], b[6:8], b[8:10], b[10:])
}

func langfuseStatusHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	out := map[string]any{
		"enabled":  langfuseEnabled(),
		"backend":  observabilityBackend(),
		"base_url": langfuseBase(),
	}
	if !langfuseEnabled() {
		_ = json.NewEncoder(w).Encode(out)
		return
	}
	code, raw, err := langfuseDo(http.MethodGet, "/api/public/health", nil)
	if err != nil {
		out["error"] = err.Error()
		w.WriteHeader(http.StatusBadGateway)
		_ = json.NewEncoder(w).Encode(out)
		return
	}
	out["health_status"] = code
	var health any
	_ = json.Unmarshal(raw, &health)
	out["health"] = health

	code, raw, err = langfuseDo(http.MethodGet, "/api/public/projects", nil)
	if err == nil && code < 300 {
		var projects any
		_ = json.Unmarshal(raw, &projects)
		out["projects"] = projects
	}
	out["dashboard_url"] = langfuseBase()
	_ = json.NewEncoder(w).Encode(out)
}

type lfSetupRequest struct {
	MaxCells int `json:"max_cells"`
}

// langfuseSetupHandler seeds current bench cells as Langfuse traces (ingestion API).
func langfuseSetupHandler(w http.ResponseWriter, r *http.Request) {
	if !langfuseEnabled() {
		langfuseDisabled(w)
		return
	}
	if r.Method != http.MethodPost {
		http.Error(w, `{"error":"POST required"}`, http.StatusMethodNotAllowed)
		return
	}
	var req lfSetupRequest
	_ = json.NewDecoder(r.Body).Decode(&req)
	if req.MaxCells <= 0 {
		req.MaxCells = 40
	}

	env, err := buildEnvelope()
	cells := []Cell{}
	if err == nil && env.Data != nil {
		cells = env.Data.Cells
	}

	if len(cells) > req.MaxCells {
		cells = cells[:req.MaxCells]
	}

	now := time.Now().UTC().Format("2006-01-02T15:04:05.000Z")
	batch := make([]map[string]any, 0, len(cells)*2)
	traceIDs := make([]string, 0, len(cells))

	for _, c := range cells {
		tid := newUUID()
		traceIDs = append(traceIDs, tid)
		genOK := c.GenOk
		if genOK == 0 && c.PassAt1 != 0 {
			genOK = c.PassAt1
		}
		batch = append(batch, map[string]any{
			"id":        newUUID(),
			"type":      "trace-create",
			"timestamp": now,
			"body": map[string]any{
				"id":   tid,
				"name": fmt.Sprintf("%s/%s/%s", c.Suite, c.TaskID, c.Variant),
				"tags": []string{"bench-cockpit", c.Suite, c.Variant},
				"metadata": map[string]any{
					"suite":              c.Suite,
					"task_id":            c.TaskID,
					"variant":            c.Variant,
					"gen_ok":             genOK,
					"verified_pass_at_1": c.VerifiedPassAt1,
					"pass_at_1":          c.PassAt1,
					"partial_credit":     c.PartialCredit,
					"wall_clock_s":       c.WallClockS,
					"tokens_per_second":  c.TokensPerSecond,
					"scoring_method":     c.ScoringMethod,
					"source":             "bench-cockpit",
				},
				"input": map[string]any{
					"prompt":  truncate(c.Prompt, 2000),
					"suite":   c.Suite,
					"task_id": c.TaskID,
					"variant": c.Variant,
				},
				"output": map[string]any{
					"reply":             truncate(c.Reply, 2000),
					"ok":                c.OK,
					"gen_ok":            genOK,
					"partial_credit":    c.PartialCredit,
					"pass_at_1":         c.PassAt1,
					"wall_clock_s":      c.WallClockS,
					"tokens_per_second": c.TokensPerSecond,
				},
			},
		})
		batch = append(batch, map[string]any{
			"id":        newUUID(),
			"type":      "score-create",
			"timestamp": now,
			"body": map[string]any{
				"id":       newUUID(),
				"traceId":  tid,
				"name":     "gen_ok",
				"value":    genOK,
				"dataType": "NUMERIC",
				"comment":  "generation success (not verified pass@1)",
			},
		})
		if c.PartialCredit > 0 {
			batch = append(batch, map[string]any{
				"id":        newUUID(),
				"type":      "score-create",
				"timestamp": now,
				"body": map[string]any{
					"id":       newUUID(),
					"traceId":  tid,
					"name":     "partial_credit",
					"value":    c.PartialCredit,
					"dataType": "NUMERIC",
				},
			})
		}
	}

	code, raw, err := langfuseDo(http.MethodPost, "/api/public/ingestion", map[string]any{"batch": batch})
	w.Header().Set("Content-Type", "application/json")
	if err != nil {
		http.Error(w, fmt.Sprintf(`{"error":%q}`, err.Error()), http.StatusBadGateway)
		return
	}
	var parsed any
	_ = json.Unmarshal(raw, &parsed)
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"status_code":    code,
		"cells_seeded":   len(cells),
		"events":         len(batch),
		"trace_ids":      traceIDs,
		"ingestion":      parsed,
		"dashboard_url":  langfuseBase(),
		"backend":        "langfuse",
	})
}

func langfuseTracesHandler(w http.ResponseWriter, r *http.Request) {
	if !langfuseEnabled() {
		langfuseDisabled(w)
		return
	}
	limit := r.URL.Query().Get("limit")
	if limit == "" {
		limit = "50"
	}
	code, raw, err := langfuseDo(http.MethodGet, "/api/public/traces?limit="+limit, nil)
	if err != nil {
		http.Error(w, fmt.Sprintf(`{"error":%q}`, err.Error()), http.StatusBadGateway)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_, _ = w.Write(raw)
}

func langfuseEvaluatorsHandler(w http.ResponseWriter, r *http.Request) {
	if !langfuseEnabled() {
		langfuseDisabled(w)
		return
	}
	if r.Method != http.MethodPost {
		http.Error(w, `{"error":"POST required"}`, http.StatusMethodNotAllowed)
		return
	}
	action := r.URL.Query().Get("action")
	if action == "" {
		action = "judge"
	}
	limit := r.URL.Query().Get("limit")
	if limit == "" {
		limit = "12"
	}
	script := filepath.Join(evalsRoot(), "scripts", "evals", "run_langfuse_evaluators.py")
	cmd := exec.Command(evalsPython(), script, action, "--limit", limit)
	cmd.Env = os.Environ()
	cmd.Dir = evalsRoot()
	out, err := cmd.CombinedOutput()
	result := map[string]any{"stdout": string(out), "action": action}
	rawOut := strings.TrimSpace(string(out))
	var parsed any
	for i := len(strings.Split(rawOut, "\n")) - 1; i >= 0; i-- {
		line := strings.TrimSpace(strings.Split(rawOut, "\n")[i])
		if strings.HasPrefix(line, "{") && json.Unmarshal([]byte(line), &parsed) == nil {
			result["result"] = parsed
			break
		}
	}
	w.Header().Set("Content-Type", "application/json")
	if err != nil {
		result["error"] = err.Error()
		w.WriteHeader(http.StatusBadGateway)
	}
	_ = json.NewEncoder(w).Encode(result)
}
