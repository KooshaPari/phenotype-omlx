package main

import (
	"encoding/json"
	"net/http"
	"strings"
)

// apiCellRawHandler — GET /api/cells/{suite}/{task_id}/{variant}/raw
func apiCellRawHandler(ring *RingBuffer) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		path := strings.TrimPrefix(r.URL.Path, "/api/cells/")
		parts := strings.Split(path, "/")
		if len(parts) != 4 || parts[3] != "raw" {
			http.Error(w, `{"error":"expected /api/cells/{suite}/{task_id}/{variant}/raw"}`, http.StatusBadRequest)
			return
		}
		suite, taskID, variant := parts[0], parts[1], parts[2]

		env, ok := ring.Latest()
		if !ok || env.Data == nil {
			// fall back to disk
			data, err := loadData()
			if err != nil {
				http.Error(w, `{"error":"no_data"}`, http.StatusServiceUnavailable)
				return
			}
			writeRawCell(w, data.Cells, suite, taskID, variant)
			return
		}
		writeRawCell(w, env.Data.Cells, suite, taskID, variant)
	}
}

func writeRawCell(w http.ResponseWriter, cells []Cell, suite, taskID, variant string) {
	for i := range cells {
		c := cells[i]
		if c.Suite == suite && c.TaskID == taskID && c.Variant == variant {
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]interface{}{
				"suite":               c.Suite,
				"task_id":             c.TaskID,
				"variant":             c.Variant,
				"task_title":          c.TaskTitle,
				"task_description":    c.TaskDescription,
				"acceptance":          c.Acceptance,
				"rubric":              c.Rubric,
				"assignment":          c.Assignment,
				"prompt":              c.Prompt,
				"reply":               c.Reply,
				"reply_full":          firstNonEmpty(c.ReplyFull, c.Reply),
				"expected_answer":     c.ExpectedAnswer,
				"scoring_method":      c.ScoringMethod,
				"pass_at_1":           c.PassAt1,
				"gen_ok":              cellGenOk(c),
				"verified_pass_at_1":  c.VerifiedPassAt1,
				"judge_score":         c.JudgeScore,
				"failure_analysis":  c.FailureAnalysis,
				"progress_trace":    c.ProgressTrace,
				"chat_trace":        c.ChatTrace,
				"wall_clock_s":      c.WallClockS,
				"tokens_per_second": c.TokensPerSecond,
			})
			return
		}
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusNotFound)
	_ = json.NewEncoder(w).Encode(map[string]string{"error": "cell_not_found"})
}
