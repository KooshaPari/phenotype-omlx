package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

func portageBin() string {
	if v := strings.TrimSpace(os.Getenv("PORTAGE_BIN")); v != "" {
		return v
	}
	return "uv run harbor"
}

// portageRoot returns the portage-TEMP checkout. Required — no hardcoded worktrees.
func portageRoot() (string, error) {
	v := strings.TrimSpace(os.Getenv("PORTAGE_ROOT"))
	if v == "" {
		return "", fmt.Errorf(
			"PORTAGE_ROOT required (portage-TEMP / Harbor checkout); " +
				"see docs/guides/EVAL_PORTAGE_LANGSMITH.md",
		)
	}
	info, err := os.Stat(v)
	if err != nil || !info.IsDir() {
		return "", fmt.Errorf("PORTAGE_ROOT is not a directory: %s", v)
	}
	return v, nil
}

func portageJobsDir() string {
	if v := strings.TrimSpace(os.Getenv("PORTAGE_JOBS_DIR")); v != "" {
		return v
	}
	return filepath.Join(os.TempDir(), "bench-cockpit-portage-jobs")
}

// portageRunHandler accepts a JSON job stub, writes YAML, shells to harbor.
// Degrades to 503 when the binary is missing; 400 when PORTAGE_ROOT unset.
func portageRunHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, `{"error":"POST required"}`, http.StatusMethodNotAllowed)
		return
	}
	root, err := portageRoot()
	if err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_ = json.NewEncoder(w).Encode(map[string]string{
			"error":  "portage_root_required",
			"detail": err.Error(),
		})
		return
	}
	var stub map[string]interface{}
	if err := json.NewDecoder(r.Body).Decode(&stub); err != nil {
		http.Error(w, fmt.Sprintf(`{"error":%q}`, err.Error()), http.StatusBadRequest)
		return
	}
	runID := fmt.Sprintf("run-%d", time.Now().UnixNano())
	dir := filepath.Join(portageJobsDir(), runID)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		http.Error(w, fmt.Sprintf(`{"error":%q}`, err.Error()), http.StatusInternalServerError)
		return
	}
	jobPath := filepath.Join(dir, "job.json")
	raw, _ := json.MarshalIndent(stub, "", "  ")
	if err := os.WriteFile(jobPath, raw, 0o644); err != nil {
		http.Error(w, fmt.Sprintf(`{"error":%q}`, err.Error()), http.StatusInternalServerError)
		return
	}

	bin := portageBin()
	parts := strings.Fields(bin)
	ctx, cancel := context.WithTimeout(r.Context(), 2*time.Minute)
	defer cancel()
	args := append(parts[1:], "run", "-c", jobPath, "-o", dir)
	cmd := exec.CommandContext(ctx, parts[0], args...)
	cmd.Dir = root
	out, err := cmd.CombinedOutput()
	_ = os.WriteFile(filepath.Join(dir, "stdout.log"), out, 0o644)
	if err != nil {
		// Missing binary → clean 503
		if ee, ok := err.(*exec.Error); ok && ee.Err == exec.ErrNotFound {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusServiceUnavailable)
			_ = json.NewEncoder(w).Encode(map[string]string{
				"error":   "portage_unavailable",
				"detail":  fmt.Sprintf("binary not found: %s", parts[0]),
				"run_id":  runID,
				"job_dir": dir,
			})
			return
		}
		// Config/schema errors are expected with stub payloads — return 422
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusUnprocessableEntity)
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"error":   "portage_run_rejected",
			"detail":  err.Error(),
			"stdout":  string(out),
			"run_id":  runID,
			"job_dir": dir,
			"hint":    "pass a valid Harbor JobConfig JSON/YAML via -c; stub payloads are for wiring only",
		})
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]string{
		"run_id":  runID,
		"job_dir": dir,
		"status":  "started",
	})
}

func portageRunStatusHandler(w http.ResponseWriter, r *http.Request) {
	id := strings.TrimPrefix(r.URL.Path, "/api/eval/runs/")
	if id == "" || strings.Contains(id, "/") || strings.Contains(id, "..") {
		http.Error(w, `{"error":"run_id required"}`, http.StatusBadRequest)
		return
	}
	resultPath := filepath.Join(portageJobsDir(), id, "result.json")
	raw, err := os.ReadFile(resultPath)
	if err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusNotFound)
		_ = json.NewEncoder(w).Encode(map[string]string{
			"error":  "result_not_found",
			"run_id": id,
			"hint":   "harbor viewer WS relay to PORTAGE_VIEWER_URL (default http://127.0.0.1:8123) is not yet wired",
		})
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write(raw)
}
