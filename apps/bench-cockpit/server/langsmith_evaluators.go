package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

func evalsPython() string {
	if v := strings.TrimSpace(os.Getenv("EVALS_PYTHON")); v != "" {
		return v
	}
	return "python3"
}

func evalsScriptPath() string {
	// server binary runs from apps/bench-cockpit/server or .run/
	candidates := []string{
		filepath.Join("..", "scripts", "evals", "run_evaluators.py"),
		filepath.Join("scripts", "evals", "run_evaluators.py"),
	}
	if wd, err := os.Getwd(); err == nil {
		for _, c := range candidates {
			p := filepath.Join(wd, c)
			if _, err := os.Stat(p); err == nil {
				return p
			}
		}
		// walk up to apps/bench-cockpit
		p := filepath.Join(wd, "..", "scripts", "evals", "run_evaluators.py")
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	return candidates[0]
}

func runEvalsCmd(args ...string) (map[string]any, error) {
	script := evalsScriptPath()
	cmdArgs := append([]string{script}, args...)
	cmd := exec.Command(evalsPython(), cmdArgs...)
	// Ensure Minimax key available for LLM judges
	cmd.Env = os.Environ()
	if os.Getenv("MINIMAX_API_KEY") == "" {
		if out, err := exec.Command("security", "find-generic-password", "-s", "minimax-coding-plan", "-w").Output(); err == nil {
			cmd.Env = append(cmd.Env, "MINIMAX_API_KEY="+strings.TrimSpace(string(out)))
		}
	}
	cmd.Dir = filepath.Dir(filepath.Dir(script)) // apps/bench-cockpit
	out, err := cmd.CombinedOutput()
	result := map[string]any{
		"stdout": string(out),
	}
	rawOut := strings.TrimSpace(string(out))
	// Prefer last line that is a JSON object; else whole stdout.
	var parsed any
	ok := false
	for i := len(strings.Split(rawOut, "\n")) - 1; i >= 0; i-- {
		line := strings.TrimSpace(strings.Split(rawOut, "\n")[i])
		if strings.HasPrefix(line, "{") && json.Unmarshal([]byte(line), &parsed) == nil {
			ok = true
			break
		}
	}
	if !ok && strings.HasPrefix(rawOut, "{") {
		ok = json.Unmarshal([]byte(rawOut), &parsed) == nil
	}
	if ok {
		result["result"] = parsed
	}
	if err != nil {
		result["error"] = err.Error()
		return result, err
	}
	return result, nil
}

func langsmithEvaluatorsHandler(w http.ResponseWriter, r *http.Request) {
	if !langsmithEnabled() {
		langsmithDisabled(w)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	switch r.Method {
	case http.MethodGet:
		code, raw, err := lsJSON(http.MethodGet, "/v1/platform/evaluators", nil)
		if err != nil {
			http.Error(w, fmt.Sprintf(`{"error":%q}`, err.Error()), http.StatusBadGateway)
			return
		}
		w.WriteHeader(code)
		_, _ = w.Write(raw)
	case http.MethodPost:
		action := r.URL.Query().Get("action")
		if action == "" {
			action = "sync"
		}
		limit := r.URL.Query().Get("limit")
		if limit == "" {
			limit = "20"
		}
		var args []string
		switch action {
		case "sync":
			args = []string{"sync"}
		case "run":
			args = []string{"run", "--limit", limit}
			if r.URL.Query().Get("no_llm") == "1" {
				args = append(args, "--no-llm")
			}
		case "all":
			args = []string{"all", "--limit", limit}
			if r.URL.Query().Get("no_llm") == "1" {
				args = append(args, "--no-llm")
			}
		default:
			http.Error(w, `{"error":"action must be sync|run|all"}`, http.StatusBadRequest)
			return
		}
		start := time.Now()
		result, err := runEvalsCmd(args...)
		result["action"] = action
		result["elapsed_ms"] = time.Since(start).Milliseconds()
		if err != nil {
			w.WriteHeader(http.StatusBadGateway)
		}
		_ = json.NewEncoder(w).Encode(result)
	default:
		http.Error(w, `{"error":"GET or POST"}`, http.StatusMethodNotAllowed)
	}
}
