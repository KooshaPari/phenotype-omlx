package main

import (
	"os"
	"path/filepath"
	"strings"
)

func evalsRoot() string {
	candidates := []string{
		filepath.Join(".."),
		".",
	}
	if wd, err := os.Getwd(); err == nil {
		candidates = append([]string{
			filepath.Join(wd, ".."),
			wd,
			filepath.Join(wd, "..", ".."),
		}, candidates...)
	}
	for _, root := range candidates {
		if _, err := os.Stat(filepath.Join(root, "scripts", "evals", "run_langfuse_evaluators.py")); err == nil {
			return root
		}
	}
	return ".."
}

func evalsPython() string {
	if v := strings.TrimSpace(os.Getenv("EVALS_PYTHON")); v != "" {
		return v
	}
	venvPy := filepath.Join(evalsRoot(), ".venv-evals", "bin", "python")
	if _, err := os.Stat(venvPy); err == nil {
		return venvPy
	}
	return "python3"
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n]
}
