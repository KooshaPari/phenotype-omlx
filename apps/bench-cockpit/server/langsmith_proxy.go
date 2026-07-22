package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
)

func langsmithEnabled() bool {
	return strings.TrimSpace(os.Getenv("LANGSMITH_API_KEY")) != ""
}

func langsmithBase() string {
	if v := strings.TrimSpace(os.Getenv("LANGSMITH_ENDPOINT")); v != "" {
		return strings.TrimRight(v, "/")
	}
	return "https://api.smith.langchain.com"
}

func langsmithDisabled(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusServiceUnavailable)
	_ = json.NewEncoder(w).Encode(map[string]string{"error": "langsmith_disabled"})
}

func langsmithProxy(method, path string, body io.Reader) (*http.Response, error) {
	req, err := http.NewRequest(method, langsmithBase()+path, body)
	if err != nil {
		return nil, err
	}
	req.Header.Set("x-api-key", os.Getenv("LANGSMITH_API_KEY"))
	req.Header.Set("Content-Type", "application/json")
	client := &http.Client{Timeout: 30 * time.Second}
	return client.Do(req)
}

func proxyJSON(w http.ResponseWriter, resp *http.Response, err error) {
	if err != nil {
		http.Error(w, fmt.Sprintf(`{"error":%q}`, err.Error()), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(resp.StatusCode)
	_, _ = io.Copy(w, resp.Body)
}

func langsmithProjectsHandler(w http.ResponseWriter, r *http.Request) {
	if !langsmithEnabled() {
		langsmithDisabled(w)
		return
	}
	resp, err := langsmithProxy(http.MethodGet, "/sessions?limit=50", nil)
	proxyJSON(w, resp, err)
}

func langsmithDatasetHandler(w http.ResponseWriter, r *http.Request) {
	if !langsmithEnabled() {
		langsmithDisabled(w)
		return
	}
	id := strings.TrimPrefix(r.URL.Path, "/api/langsmith/datasets/")
	if id == "" || strings.Contains(id, "/") {
		http.Error(w, `{"error":"dataset id required"}`, http.StatusBadRequest)
		return
	}
	resp, err := langsmithProxy(http.MethodGet, "/datasets/"+id, nil)
	proxyJSON(w, resp, err)
}

func langsmithRunsHandler(w http.ResponseWriter, r *http.Request) {
	if !langsmithEnabled() {
		langsmithDisabled(w)
		return
	}
	project := r.URL.Query().Get("project")
	path := "/runs?limit=50"
	if project != "" {
		path += "&session=" + project
	}
	resp, err := langsmithProxy(http.MethodGet, path, nil)
	proxyJSON(w, resp, err)
}

func langsmithFeedbackHandler(w http.ResponseWriter, r *http.Request) {
	if !langsmithEnabled() {
		langsmithDisabled(w)
		return
	}
	if r.Method != http.MethodPost {
		http.Error(w, `{"error":"POST required"}`, http.StatusMethodNotAllowed)
		return
	}
	resp, err := langsmithProxy(http.MethodPost, "/feedback", r.Body)
	proxyJSON(w, resp, err)
}
