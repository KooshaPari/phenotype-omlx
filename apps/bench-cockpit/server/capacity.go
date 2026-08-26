package main

import (
	"encoding/json"
	"math"
	"net/http"
	"os"
	"strconv"
	"strings"
)

// Mirrors perf-core/pheno-capacity (ADR-006 embed seam) for cockpit cell fit hints.

func dtypeBytes(dtype string) float64 {
	switch strings.ToUpper(strings.TrimSpace(dtype)) {
	case "F32", "FP32":
		return 4
	case "BF16":
		return 2
	case "I8", "INT8":
		return 1
	case "I4", "INT4", "Q4":
		return 0.5
	default: // F16 / FP16
		return 2
	}
}

func vramEstimate(params uint64, dtype string) uint64 {
	return uint64(math.Round(float64(params) * dtypeBytes(dtype)))
}

func modelFitsIn(params, available uint64, dtype string) bool {
	return vramEstimate(params, dtype) <= available
}

// Heuristic param counts from model_name / meta strings (pairing aliases).
func paramsFromModelName(name string) (uint64, string) {
	n := strings.ToLower(name)
	switch {
	case n == "smoke" || strings.Contains(n, "smoke"):
		return 800_000_000, "smoke→0.8B demo"
	case strings.Contains(n, "0.8") || strings.Contains(n, "08b") || strings.Contains(n, "0_8"):
		return 800_000_000, "0.8B heuristic"
	case strings.Contains(n, "1.5b") || strings.Contains(n, "1_5"):
		return 1_500_000_000, "1.5B heuristic"
	case strings.Contains(n, "3b") && !strings.Contains(n, "30") && !strings.Contains(n, "35"):
		return 3_000_000_000, "3B heuristic"
	case strings.Contains(n, "4b"):
		return 4_000_000_000, "4B heuristic"
	case strings.Contains(n, "7b"):
		return 7_000_000_000, "7B heuristic"
	case strings.Contains(n, "8b"):
		return 8_000_000_000, "8B heuristic"
	case strings.Contains(n, "9b"):
		return 9_000_000_000, "9B heuristic"
	case strings.Contains(n, "14b"):
		return 14_000_000_000, "14B heuristic"
	case strings.Contains(n, "27b"):
		return 27_000_000_000, "27B heuristic"
	case strings.Contains(n, "32b"):
		return 32_000_000_000, "32B heuristic"
	case strings.Contains(n, "70b"):
		return 70_000_000_000, "70B heuristic"
	default:
		return 0, "unknown"
	}
}

func defaultAvailableBytes() uint64 {
	if v := strings.TrimSpace(os.Getenv("CAPACITY_AVAILABLE_BYTES")); v != "" {
		if n, err := strconv.ParseUint(v, 10, 64); err == nil && n > 0 {
			return n
		}
	}
	// Desk default: RTX 3090 Ti ~24 GiB
	return 24 * 1024 * 1024 * 1024
}

type capacityFitResponse struct {
	Fits               bool    `json:"fits"`
	VRAMEstimateBytes  uint64  `json:"vram_estimate_bytes"`
	VRAMEstimateGB     float64 `json:"vram_estimate_gb"`
	AvailableBytes     uint64  `json:"available_bytes"`
	AvailableGB        float64 `json:"available_gb"`
	Params             uint64  `json:"params"`
	Dtype              string  `json:"dtype"`
	Source             string  `json:"source"`
	ModelHint          string  `json:"model_hint,omitempty"`
}

func capacityFitHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	q := r.URL.Query()

	dtype := q.Get("dtype")
	if dtype == "" {
		dtype = "F16"
	}

	var params uint64
	var source string
	if p := q.Get("params"); p != "" {
		n, err := strconv.ParseUint(p, 10, 64)
		if err != nil {
			http.Error(w, `{"error":"invalid params"}`, http.StatusBadRequest)
			return
		}
		params = n
		source = "query"
	} else if model := q.Get("model"); model != "" {
		params, source = paramsFromModelName(model)
		if params == 0 {
			_ = json.NewEncoder(w).Encode(map[string]any{
				"error":  "unknown_model",
				"model":  model,
				"hint":   "pass params= explicitly",
			})
			return
		}
	} else {
		http.Error(w, `{"error":"params or model required"}`, http.StatusBadRequest)
		return
	}

	available := defaultAvailableBytes()
	if a := q.Get("available_bytes"); a != "" {
		if n, err := strconv.ParseUint(a, 10, 64); err == nil {
			available = n
		}
	}

	need := vramEstimate(params, dtype)
	out := capacityFitResponse{
		Fits:              modelFitsIn(params, available, dtype),
		VRAMEstimateBytes: need,
		VRAMEstimateGB:    float64(need) / 1e9,
		AvailableBytes:    available,
		AvailableGB:       float64(available) / (1024 * 1024 * 1024),
		Params:            params,
		Dtype:             dtype,
		Source:            source,
		ModelHint:         q.Get("model"),
	}
	_ = json.NewEncoder(w).Encode(out)
}
