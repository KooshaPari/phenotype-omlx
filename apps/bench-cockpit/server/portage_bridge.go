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
				"see docs/guides/EVAL_PORTAGE_LANGFUSE.md",
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

func defaultHelloWorldTask(root string) string {
	return filepath.Join(root, "examples", "tasks", "hello-world")
}

func harborEnvironment() (string, error) {
	environment := strings.TrimSpace(os.Getenv("HARBOR_ENV"))
	if environment == "" {
		return "apple-container", nil
	}
	if environment != "apple-container" {
		return "", fmt.Errorf(
			"HARBOR_ENV=%q is forbidden on this host; use apple-container. "+
				"If Podman is installed elsewhere, Harbor's override is "+
				"`-e docker --ek container_runtime=podman`",
			environment,
		)
	}
	return environment, nil
}

func portageCommandEnvironment() []string {
	path := "/usr/local/bin"
	if current := os.Getenv("PATH"); current != "" {
		path += string(os.PathListSeparator) + current
	}
	env := make([]string, 0, len(os.Environ())+1)
	for _, entry := range os.Environ() {
		if !strings.HasPrefix(entry, "PATH=") {
			env = append(env, entry)
		}
	}
	return append(env, "PATH="+path)
}

func normalizeJobEnvironment(job map[string]interface{}) error {
	raw, ok := job["environment"]
	if !ok {
		job["environment"] = map[string]interface{}{"type": "apple-container"}
		return nil
	}
	environment, ok := raw.(map[string]interface{})
	if !ok {
		return fmt.Errorf("job environment must be an object")
	}
	rawType, ok := environment["type"]
	if !ok {
		environment["type"] = "apple-container"
		return nil
	}
	environmentType, ok := rawType.(string)
	if !ok {
		return fmt.Errorf("job environment type must be a string")
	}
	environmentType = strings.TrimSpace(environmentType)
	if environmentType == "" {
		environment["type"] = "apple-container"
		return nil
	}
	if environmentType == "docker" {
		return fmt.Errorf(
			"docker environment is forbidden on this host; use apple-container. " +
				"If Podman is installed elsewhere, Harbor's override is " +
				"`-e docker --ek container_runtime=podman`",
		)
	}
	return nil
}

// portageRunHandler starts a Harbor job.
// Body modes:
//  1. {"mode":"hello_world"} or empty → oracle hello-world smoke
//  2. {"mode":"path","task_path":"...","agent":"oracle"} — always attaches --plugin langfuse
//  3. full JobConfig JSON with tasks[] (written to -c)
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
	if err := json.NewDecoder(r.Body).Decode(&stub); err != nil && err.Error() != "EOF" {
		http.Error(w, fmt.Sprintf(`{"error":%q}`, err.Error()), http.StatusBadRequest)
		return
	}
	if stub == nil {
		stub = map[string]interface{}{}
	}

	runID := fmt.Sprintf("run-%d", time.Now().UnixNano())
	dir := filepath.Join(portageJobsDir(), runID)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		http.Error(w, fmt.Sprintf(`{"error":%q}`, err.Error()), http.StatusInternalServerError)
		return
	}

	bin := portageBin()
	parts := strings.Fields(bin)
	if len(parts) == 0 {
		http.Error(w, `{"error":"PORTAGE_BIN empty"}`, http.StatusInternalServerError)
		return
	}

	envName, err := harborEnvironment()
	if err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_ = json.NewEncoder(w).Encode(map[string]string{
			"error":  "docker_forbidden",
			"detail": err.Error(),
		})
		return
	}

	timeout := 15 * time.Minute
	ctx, cancel := context.WithTimeout(r.Context(), timeout)
	defer cancel()

	var args []string
	mode, _ := stub["mode"].(string)
	_, hasTasks := stub["tasks"]
	useConfig := hasTasks && mode != "path" && mode != "hello_world"

	if useConfig {
		if err := normalizeJobEnvironment(stub); err != nil {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadRequest)
			_ = json.NewEncoder(w).Encode(map[string]string{
				"error":  "docker_forbidden",
				"detail": err.Error(),
			})
			return
		}
		jobPath := filepath.Join(dir, "job.json")
		raw, _ := json.MarshalIndent(stub, "", "  ")
		if err := os.WriteFile(jobPath, raw, 0o644); err != nil {
			http.Error(w, fmt.Sprintf(`{"error":%q}`, err.Error()), http.StatusInternalServerError)
			return
		}
		args = append(parts[1:], "run", "-c", jobPath, "-o", dir, "-y")
	} else {
		taskPath, _ := stub["task_path"].(string)
		if taskPath == "" || mode == "hello_world" || mode == "" {
			taskPath = defaultHelloWorldTask(root)
		}
		agent, _ := stub["agent"].(string)
		if agent == "" {
			agent = "oracle"
		}
		nConc := "1"
		if v, ok := stub["n"].(float64); ok && v >= 1 {
			nConc = fmt.Sprintf("%d", int(v))
		}
		args = append(parts[1:], "run", "-e", envName, "-p", taskPath, "-a", agent, "-n", nConc, "-o", dir, "-y")
		// Canonical observability: harbor-langfuse (LangSmith removed from cockpit path).
		args = append(args, "--plugin", "langfuse")
	}

	cmd := exec.CommandContext(ctx, parts[0], args...)
	cmd.Dir = root
	cmd.Env = portageCommandEnvironment()
	pluginSrc := filepath.Join(root, "packages", "harbor-langfuse", "src")
	if _, err := os.Stat(pluginSrc); err == nil {
		prev := os.Getenv("PYTHONPATH")
		if prev != "" {
			cmd.Env = append(cmd.Env, "PYTHONPATH="+pluginSrc+string(os.PathListSeparator)+prev)
		} else {
			cmd.Env = append(cmd.Env, "PYTHONPATH="+pluginSrc)
		}
	}

	out, err := cmd.CombinedOutput()
	_ = os.WriteFile(filepath.Join(dir, "stdout.log"), out, 0o644)
	_ = os.WriteFile(filepath.Join(dir, "cmdline.txt"), []byte(strings.Join(append([]string{parts[0]}, args...), " ")), 0o644)

	if err != nil {
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
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusUnprocessableEntity)
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"error":   "portage_run_rejected",
			"detail":  err.Error(),
			"stdout":  truncate(string(out), 4000),
			"run_id":  runID,
			"job_dir": dir,
			"hint":    "POST {\"mode\":\"hello_world\"} or {\"mode\":\"path\",\"task_path\":\"...\",\"agent\":\"oracle\"}",
		})
		return
	}

	resultPath, reward := findHarborResult(dir)
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]interface{}{
		"run_id":      runID,
		"job_dir":     dir,
		"status":      "completed",
		"result_path": resultPath,
		"reward":      reward,
	})
}

func findHarborResult(runDir string) (string, any) {
	candidates := []string{filepath.Join(runDir, "result.json")}
	entries, _ := os.ReadDir(runDir)
	for _, e := range entries {
		if e.IsDir() {
			candidates = append(candidates, filepath.Join(runDir, e.Name(), "result.json"))
		}
	}
	for _, p := range candidates {
		raw, err := os.ReadFile(p)
		if err != nil {
			continue
		}
		var parsed any
		_ = json.Unmarshal(raw, &parsed)
		return p, extractReward(parsed)
	}
	return "", nil
}

func extractReward(parsed any) any {
	m, ok := parsed.(map[string]any)
	if !ok {
		return nil
	}
	if stats, ok := m["stats"].(map[string]any); ok {
		if evals, ok := stats["evals"].(map[string]any); ok {
			return evals
		}
		return stats
	}
	if v, ok := m["reward"]; ok {
		return v
	}
	return nil
}

func portageRunStatusHandler(w http.ResponseWriter, r *http.Request) {
	id := strings.TrimPrefix(r.URL.Path, "/api/eval/runs/")
	if id == "" || strings.Contains(id, "/") || strings.Contains(id, "..") {
		http.Error(w, `{"error":"run_id required"}`, http.StatusBadRequest)
		return
	}
	runDir := filepath.Join(portageJobsDir(), id)
	resultPath, reward := findHarborResult(runDir)
	if resultPath == "" {
		stdout, _ := os.ReadFile(filepath.Join(runDir, "stdout.log"))
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusNotFound)
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"error":  "result_not_found",
			"run_id": id,
			"stdout": truncate(string(stdout), 2000),
			"hint":   "wait for harbor job; result.json lives under job_dir/<job_name>/",
		})
		return
	}
	raw, _ := os.ReadFile(resultPath)
	var parsed any
	_ = json.Unmarshal(raw, &parsed)
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]interface{}{
		"run_id":      id,
		"result_path": resultPath,
		"reward":      reward,
		"result":      parsed,
	})
}
