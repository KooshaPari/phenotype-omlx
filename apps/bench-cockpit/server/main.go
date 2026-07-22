package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/fsnotify/fsnotify"
	"github.com/gorilla/websocket"
)

// ---------------------------------------------------------------------------
// Domain types (unchanged JSON structure)
// ---------------------------------------------------------------------------

type Meta struct {
	Model      string         `json:"model"`
	MLXURL     string         `json:"mlx_url"`
	NSuites    int            `json:"n_suites"`
	NTasks     int            `json:"n_tasks_per_suite"`
	Variants   []string       `json:"variants"`
	NCells     int            `json:"n_cells"`
	Difficulty map[string]int `json:"difficulty_mix"`
}

type VariantSummary struct {
	NCells               int     `json:"n_cells"`
	PassAt1              float64 `json:"pass_at_1"`
	GenOk                float64 `json:"gen_ok"`
	VerifiedPassAt1      float64 `json:"verified_pass_at_1"`
	MeanWallClockS       float64 `json:"mean_wall_clock_s"`
	MeanPartialCredit    float64 `json:"mean_partial_credit"`
	MeanFormatCompliance float64 `json:"mean_format_compliance"`
	MeanIntentPres       float64 `json:"mean_intent_preservation"`
	NHallucinations      int     `json:"n_hallucinations"`
	OkCount              int     `json:"ok_count"`
	MeanTokensPerSecond  float64 `json:"mean_tokens_per_second"`
	MeanTokensRead       float64 `json:"mean_tokens_read"` // harness alias; UI Tok/s prefers mean_tokens_per_second
}

type Summary struct {
	Meta      Meta                      `json:"meta"`
	ByVariant map[string]VariantSummary `json:"by_variant"`
}

type Cell struct {
	Suite               string                 `json:"suite"`
	TaskID              string                 `json:"task_id"`
	Difficulty          string                 `json:"difficulty"`
	Variant             string                 `json:"variant"`
	OK                  bool                   `json:"ok"`
	WallClockS          float64                `json:"wall_clock_s"`
	TokensPerSecond     float64                `json:"tokens_per_second"`
	FirstTokenLatencyMS float64                `json:"first_token_latency_ms"`
	PeakRSSMB           float64                `json:"peak_rss_mb"`
	PeakGPUMemMB        float64                `json:"peak_gpu_mem_mb"`
	EnergyProxyJoules   float64                `json:"energy_proxy_joules"`
	PassAt1             float64                `json:"pass_at_1"`
	GenOk               float64                `json:"gen_ok"`
	VerifiedPassAt1     float64                `json:"verified_pass_at_1"`
	PartialCredit       float64                `json:"partial_credit"`
	JudgeScore          float64                `json:"judge_score"`
	IntentPreservation  float64                `json:"intent_preservation_rate"`
	HallucinationCount  int                    `json:"hallucination_count"`
	ToolCallSuccess     float64                `json:"tool_call_success_rate"`
	RetryCount          int                    `json:"retry_count"`
	FormatCompliance    float64                `json:"format_compliance_rate"`
	Reply               string                 `json:"reply"`
	ReplyFull           string                 `json:"reply_full,omitempty"`
	Prompt              string                 `json:"prompt"`
	Semantic            map[string]float64     `json:"semantic"`
	FailureAnalysis     map[string]interface{} `json:"failure_analysis"`
	ProgressTrace       []interface{}          `json:"progress_trace"`
	ChatTrace           []interface{}          `json:"chat_trace,omitempty"`
	TaskTitle           string                 `json:"task_title,omitempty"`
	TaskDescription     string                 `json:"task_description,omitempty"`
	Acceptance          string                 `json:"acceptance,omitempty"`
	Rubric              string                 `json:"rubric,omitempty"`
	Assignment          map[string]interface{} `json:"assignment,omitempty"`
	CreatedAt           string                 `json:"created_at"`
	CompletedAt         string                 `json:"completed_at"`
	ModelName           string                 `json:"model_name"`
	ModelVersion        string                 `json:"model_version"`
	Temperature         float64                `json:"temperature"`
	MaxTokens           int                    `json:"max_tokens"`
	TopP                float64                `json:"top_p"`
	Seed                int                    `json:"seed"`
	SystemPromptHash    string                 `json:"system_prompt_hash"`
	TaskType            string                 `json:"task_type"`
	ExpectedAnswer      string                 `json:"expected_answer"`
	ScoringMethod       string                 `json:"scoring_method"`
	TotalTokensIn       int                    `json:"total_tokens_in"`
	TotalTokensOut      int                    `json:"total_tokens_out"`
	CostUSD             float64                `json:"cost_usd"`
	ErrorMessage        string                 `json:"error_message"`
	ErrorCode           string                 `json:"error_code"`
	Metadata            map[string]string      `json:"metadata"`
	dualReadGenOkSet      bool                   `json:"-"`
	dualReadVerifiedSet   bool                   `json:"-"`
}

type ResultsData struct {
	Summary Summary `json:"summary"`
	Cells   []Cell  `json:"cells"`
}

type Envelope struct {
	JSONPath       string             `json:"jsonPath"`
	ExtraPaths     []string           `json:"extraPaths,omitempty"`
	ServerTS       string             `json:"serverTs"`
	Data           *ResultsData       `json:"data"`
	Warnings       []LintWarning      `json:"warnings,omitempty"`
	LintRunTS      string             `json:"lintRunTs,omitempty"`
	SuiteCoverage  []suiteCoverageRow `json:"suite_coverage,omitempty"`
}

type LintWarning struct {
	Code     string   `json:"code"`
	Severity string   `json:"severity"`
	Message  string   `json:"message"`
	Cells    []string `json:"cells,omitempty"`
}

// ---------------------------------------------------------------------------
// Ring buffer for /api/history (last 30 envelopes)
// ---------------------------------------------------------------------------

type RingBuffer struct {
	mu       sync.Mutex
	buf      []Envelope
	capacity int
}

func newRingBuffer(cap int) *RingBuffer {
	return &RingBuffer{buf: make([]Envelope, 0, cap), capacity: cap}
}

func (rb *RingBuffer) Push(e Envelope) {
	rb.mu.Lock()
	defer rb.mu.Unlock()
	if len(rb.buf) >= rb.capacity {
		copy(rb.buf, rb.buf[1:])
		rb.buf[len(rb.buf)-1] = e
	} else {
		rb.buf = append(rb.buf, e)
	}
}

// Snapshot returns a shallow copy of the buffer (oldest first).
func (rb *RingBuffer) Snapshot() []Envelope {
	rb.mu.Lock()
	defer rb.mu.Unlock()
	out := make([]Envelope, len(rb.buf))
	copy(out, rb.buf)
	return out
}

// Latest returns the newest envelope if any.
func (rb *RingBuffer) Latest() (Envelope, bool) {
	rb.mu.Lock()
	defer rb.mu.Unlock()
	if len(rb.buf) == 0 {
		return Envelope{}, false
	}
	return rb.buf[len(rb.buf)-1], true
}

// ---------------------------------------------------------------------------
// Globals
// ---------------------------------------------------------------------------

var (
	distDir    string
	dataPath   string
	extraPaths []string
)

// ---------------------------------------------------------------------------
// Data loading
// ---------------------------------------------------------------------------

func loadData() (*ResultsData, error) {
	base, err := loadResultsFile(dataPath)
	if err != nil {
		return nil, err
	}
	extras := make([]*ResultsData, 0, len(extraPaths))
	for _, p := range extraPaths {
		ex, err := loadResultsFile(p)
		if err != nil {
			log.Printf("warn: skip extra data %s: %v", p, err)
			continue
		}
		extras = append(extras, ex)
	}
	if len(extras) > 0 {
		base = mergeResults(base, extras...)
	}
	return base, nil
}

func enrichVariantThroughput(data *ResultsData) {
	if data == nil || len(data.Cells) == 0 {
		return
	}
	sums := summarizeByVariant(data.Cells)
	if data.Summary.ByVariant == nil {
		data.Summary.ByVariant = sums
		return
	}
	for v, enriched := range sums {
		cur := data.Summary.ByVariant[v]
		if cur.MeanTokensPerSecond == 0 && enriched.MeanTokensPerSecond > 0 {
			cur.MeanTokensPerSecond = enriched.MeanTokensPerSecond
		}
		// VerdictStrip historically read mean_tokens_read for Tok/s — keep it as tok/s rate.
		if enriched.MeanTokensPerSecond > 0 {
			cur.MeanTokensRead = enriched.MeanTokensPerSecond
			if cur.MeanTokensPerSecond == 0 {
				cur.MeanTokensPerSecond = enriched.MeanTokensPerSecond
			}
		}
		data.Summary.ByVariant[v] = cur
	}
}

// lintCells detects degenerate / vacuous-pass cells in the loaded data.
// A "degenerate" cell is one that scored pass_at_1 == 1.0 with markers
// consistent with a vacuous pass (empty prompt, near-zero wall-clock,
// trivial completion). Such cells inflate pass@1 to a meaningless 100%
// and must be surfaced before the user trusts the dashboard.
func lintCells(cells []Cell) []LintWarning {
	var warnings []LintWarning
	type key struct {
		suite  string
		taskid string
	}
	byKey := make(map[key][]Cell)
	for i := range cells {
		c := cells[i]
		byKey[key{c.Suite, c.TaskID}] = append(byKey[key{c.Suite, c.TaskID}], c)
	}

	// Rule 1: pass@1 == 1.0 with degenerate timing/content signals.
	// EvaluationReport contracts often omit prompt/reply text — empty
	// strings alone are not enough; require wall-clock or token evidence.
	var trivial []string
	for i := range cells {
		c := cells[i]
		if c.PassAt1 < 0.999 {
			continue
		}
		noIO := c.Prompt == "" && c.Reply == ""
		noTokens := c.TotalTokensIn+c.TotalTokensOut == 0
		fast := c.WallClockS < 0.05
		if fast || (noIO && noTokens && c.WallClockS < 1.0) {
			trivial = append(trivial, fmt.Sprintf("%s/%s/%s", c.Suite, c.TaskID, c.Variant))
		}
	}
	if len(trivial) > 0 {
		warnings = append(warnings, LintWarning{
			Code:     "degenerate_cell",
			Severity: "error",
			Message:  fmt.Sprintf("%d cell(s) scored 100%% with degenerate signals (empty prompt/reply or sub-50ms wall-clock). These are likely vacuous passes and must be re-verified.", len(trivial)),
			Cells:    trivial,
		})
	}

	// Rule 2: same (suite, task_id) is 100% across stock+ours ablation peers
	// → likely an unscored placeholder. Ignore aux arms (judge/eval matrices).
	var allPass []string
	for k, group := range byKey {
		var peers []Cell
		for _, c := range group {
			if c.Variant == "stock" || c.Variant == "ours" {
				peers = append(peers, c)
			}
		}
		if len(peers) < 2 {
			continue
		}
		all := true
		for _, c := range peers {
			if c.PassAt1 < 0.999 {
				all = false
				break
			}
		}
		if all {
			allPass = append(allPass, fmt.Sprintf("%s/%s", k.suite, k.taskid))
		}
	}
	if len(allPass) > 0 {
		warnings = append(warnings, LintWarning{
			Code:     "all_variants_pass",
			Severity: "warning",
			Message:  fmt.Sprintf("%d task(s) scored 100%% across stock+ours — likely unscored placeholder fixture.", len(allPass)),
			Cells:    allPass,
		})
	}

	// Rule 3: judge_score is missing or zero on a cell that otherwise
	// claims pass@1 == 1.0 → no actual scoring happened.
	// Skip when metadata already carries a deterministic judge + raw score
	// (EvaluationReport flatten maps raw_score → pass@1).
	var noJudge []string
	for i := range cells {
		c := cells[i]
		if c.PassAt1 < 0.999 || c.JudgeScore != 0 {
			continue
		}
		if c.Metadata != nil && (c.Metadata["judge"] == "deterministic" || c.Metadata["judge"] == "exact") {
			continue
		}
		noJudge = append(noJudge, fmt.Sprintf("%s/%s/%s", c.Suite, c.TaskID, c.Variant))
	}
	if len(noJudge) > 0 {
		warnings = append(warnings, LintWarning{
			Code:     "missing_judge_score",
			Severity: "warning",
			Message:  fmt.Sprintf("%d cell(s) have pass@1==1.0 with judge_score==0 — likely not actually scored.", len(noJudge)),
			Cells:    noJudge,
		})
	}

	// Rule 4: pass@1 == 1.0 with BOTH empty expected_answer and scoring_method
	// — smoking gun for the vacuous-pass bug (pheno-harness _verify).
	var vacuous []string
	for i := range cells {
		c := cells[i]
		if c.PassAt1 < 0.999 {
			continue
		}
		if c.ExpectedAnswer == "" && c.ScoringMethod == "" {
			vacuous = append(vacuous, fmt.Sprintf("%s/%s/%s", c.Suite, c.TaskID, c.Variant))
		}
	}
	if len(vacuous) > 0 {
		warnings = append(warnings, LintWarning{
			Code:     "vacuous_pass",
			Severity: "error",
			Message:  fmt.Sprintf("%d cell(s) scored 100%% with empty expected_answer and scoring_method — vacuous pass.", len(vacuous)),
			Cells:    vacuous,
		})
	}

	// Rule 5: synthetic evidence class — do not treat pass@1 as live-suite proof.
	var synthetic []string
	synCount := 0
	for i := range cells {
		c := cells[i]
		if c.Metadata != nil && c.Metadata["synthetic"] == "true" {
			synCount++
			if len(synthetic) < 20 {
				synthetic = append(synthetic, fmt.Sprintf("%s/%s/%s", c.Suite, c.TaskID, c.Variant))
			}
		}
	}
	if synCount > 0 {
		sev := "warning"
		msg := fmt.Sprintf("%d/%d cell(s) marked synthetic=true (evidence_label=reported). pass@1 is not live-suite promotion proof.", synCount, len(cells))
		if synCount == len(cells) {
			sev = "error"
			msg = fmt.Sprintf("ALL %d cells are synthetic=true — treat dashboard scores as reported/synthetic smoke, not FR-5 live evidence.", synCount)
		}
		warnings = append(warnings, LintWarning{
			Code:     "synthetic_100pct",
			Severity: sev,
			Message:  msg,
			Cells:    synthetic,
		})
	}

	return warnings
}

func buildEnvelope() (Envelope, error) {
	data, err := loadData()
	if err != nil {
		return Envelope{}, err
	}
	warnings := lintCells(data.Cells)
	return Envelope{
		JSONPath:      dataPath,
		ExtraPaths:    append([]string{}, extraPaths...),
		ServerTS:      time.Now().UTC().Format(time.RFC3339Nano),
		Data:          data,
		Warnings:      warnings,
		LintRunTS:     time.Now().UTC().Format(time.RFC3339Nano),
		SuiteCoverage: buildSuiteCoverage(data.Cells),
	}, nil
}

func marshalEnvelope(e Envelope) ([]byte, error) {
	return json.Marshal(e)
}

// ---------------------------------------------------------------------------
// Middleware
// ---------------------------------------------------------------------------

type statusWriter struct {
	http.ResponseWriter
	status int
}

func (sw *statusWriter) WriteHeader(code int) {
	sw.status = code
	sw.ResponseWriter.WriteHeader(code)
}

// Hijack pass-through so gorilla/websocket can upgrade (otherwise:
// "response does not implement http.Hijacker").
func (sw *statusWriter) Hijack() (net.Conn, *bufio.ReadWriter, error) {
	h, ok := sw.ResponseWriter.(http.Hijacker)
	if !ok {
		return nil, nil, fmt.Errorf("ResponseWriter does not implement http.Hijacker")
	}
	return h.Hijack()
}

func (sw *statusWriter) Flush() {
	if f, ok := sw.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

// logMiddleware logs every incoming request with method, path, status, duration.
// WebSocket upgrades skip the statusWriter wrapper so Hijack works even if an
// outer middleware forgets to forward Hijacker.
func logMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		if strings.EqualFold(r.Header.Get("Upgrade"), "websocket") {
			next.ServeHTTP(w, r)
			log.Printf("%s %s (ws-upgrade) %s", r.Method, r.URL.Path,
				time.Since(start).Round(time.Microsecond))
			return
		}
		sw := &statusWriter{ResponseWriter: w, status: http.StatusOK}
		next.ServeHTTP(sw, r)
		log.Printf("%s %s %d %s", r.Method, r.URL.Path, sw.status,
			time.Since(start).Round(time.Microsecond))
	})
}

// corsMiddleware adds permissive CORS headers for local development.
func corsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, r)
	})
}

// securityHeaders adds standard security headers.
func securityHeaders(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("X-Content-Type-Options", "nosniff")
		w.Header().Set("X-Frame-Options", "DENY")
		w.Header().Set("Referrer-Policy", "strict-origin-when-cross-origin")
		next.ServeHTTP(w, r)
	})
}

// ---------------------------------------------------------------------------
// SPA handler — serves static files from -dist with index.html fallback
// ---------------------------------------------------------------------------

func indexHandler(w http.ResponseWriter, r *http.Request) {
	path := filepath.Join(distDir, r.URL.Path)
	isIndex := r.URL.Path == "/" || r.URL.Path == "/index.html"
	if r.URL.Path == "/" {
		path = filepath.Join(distDir, "index.html")
	}
	if _, err := os.Stat(path); os.IsNotExist(err) {
		// Missing hashed assets must 404 — never SPA-fallback (that kept stale JS "alive").
		if strings.HasPrefix(r.URL.Path, "/assets/") {
			http.NotFound(w, r)
			return
		}
		path = filepath.Join(distDir, "index.html")
		isIndex = true
	}
	if isIndex || strings.HasSuffix(path, ".html") {
		w.Header().Set("Cache-Control", "no-store, no-cache, must-revalidate")
		w.Header().Set("Pragma", "no-cache")
	}
	http.ServeFile(w, r, path)
}

// ---------------------------------------------------------------------------
// API handlers
// ---------------------------------------------------------------------------

// GET /api/state — latest envelope (current data snapshot).
func apiStateHandler(ring *RingBuffer) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		env, err := buildEnvelope()
		if err != nil {
			log.Printf("/api/state error: %v", err)
			http.Error(w, `{"error":"failed to load data"}`, http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(env)
	}
}

// GET /api/data — backward-compat alias, identical to /api/state.
func apiDataHandler(ring *RingBuffer) http.HandlerFunc {
	return apiStateHandler(ring)
}

// GET /api/history — last 30 received data pushes (oldest first).
func apiHistoryHandler(ring *RingBuffer) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(ring.Snapshot())
	}
}

// GET /api/export — current data as downloadable JSON with content-disposition.
func apiExportHandler(ring *RingBuffer) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		env, err := buildEnvelope()
		if err != nil {
			log.Printf("/api/export error: %v", err)
			http.Error(w, `{"error":"failed to load data"}`, http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Content-Disposition",
			fmt.Sprintf(`attachment; filename="%s"`, filepath.Base(dataPath)))
		json.NewEncoder(w).Encode(env)
	}
}

// GET /api/health
func healthHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	fmt.Fprintf(w, `{"status":"ok","jsonPath":"%s","ts":"%s"}`,
		strings.ReplaceAll(dataPath, `\`, `\\`),
		time.Now().UTC().Format(time.RFC3339Nano),
	)
}

// ---------------------------------------------------------------------------
// WebSocket hub
// ---------------------------------------------------------------------------

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool { return true },
}

const (
	wsWriteWait      = 10 * time.Second
	wsPongWait       = 60 * time.Second
	wsPingPeriod     = 45 * time.Second // must be < PongWait
	wsMaxMessageSize = 512
)

type Hub struct {
	mu      sync.Mutex
	clients map[*websocket.Conn]bool
}

func newHub() *Hub {
	return &Hub{clients: make(map[*websocket.Conn]bool)}
}

func (h *Hub) add(conn *websocket.Conn) {
	h.mu.Lock()
	h.clients[conn] = true
	n := len(h.clients)
	h.mu.Unlock()
	log.Printf("ws client connected (%d total)", n)
}

func (h *Hub) remove(conn *websocket.Conn) {
	h.mu.Lock()
	delete(h.clients, conn)
	n := len(h.clients)
	h.mu.Unlock()
	conn.Close()
	log.Printf("ws client disconnected (%d total)", n)
}

// broadcast sends msg to every connected client, cleaning up stale ones.
func (h *Hub) broadcast(msg []byte) {
	h.mu.Lock()
	var stale []*websocket.Conn
	for conn := range h.clients {
		if err := conn.SetWriteDeadline(time.Now().Add(wsWriteWait)); err != nil {
			stale = append(stale, conn)
			continue
		}
		if err := conn.WriteMessage(websocket.TextMessage, msg); err != nil {
			stale = append(stale, conn)
		}
	}
	for _, conn := range stale {
		delete(h.clients, conn)
		conn.Close()
	}
	if len(stale) > 0 {
		log.Printf("ws broadcast: cleaned %d stale connection(s)", len(stale))
	}
	h.mu.Unlock()
}

func wsHandler(hub *Hub) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			log.Printf("ws upgrade: %v", err)
			return
		}
		hub.add(conn)

		// Slim notify only — full envelopes are ~1.5MB and trip browser/WS
		// frame limits (1009). Clients hydrate via GET /api/state.
		if env, err := buildEnvelope(); err == nil {
			notify, _ := json.Marshal(map[string]any{
				"type":     "reload",
				"serverTs": env.ServerTS,
			})
			if err := conn.WriteMessage(websocket.TextMessage, notify); err != nil {
				log.Printf("ws initial notify: %v", err)
				hub.remove(conn)
				return
			}
		}

		// Configure pong handler to reset the read deadline.
		conn.SetReadDeadline(time.Now().Add(wsPongWait))
		conn.SetPongHandler(func(string) error {
			conn.SetReadDeadline(time.Now().Add(wsPongWait))
			return nil
		})

		// Writer goroutine: sends periodic pings.
		go func() {
			ticker := time.NewTicker(wsPingPeriod)
			defer ticker.Stop()
			for range ticker.C {
				hub.mu.Lock()
				_, ok := hub.clients[conn]
				hub.mu.Unlock()
				if !ok {
					return
				}
				if err := conn.WriteControl(websocket.PingMessage, nil, time.Now().Add(wsWriteWait)); err != nil {
					log.Printf("ws ping: %v", err)
					hub.remove(conn)
					return
				}
			}
		}()

		// Reader goroutine: drain incoming messages and detect disconnect.
		defer hub.remove(conn)
		conn.SetReadLimit(wsMaxMessageSize)
		for {
			if _, _, err := conn.ReadMessage(); err != nil {
				if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseNormalClosure) {
					log.Printf("ws unexpected close: %v", err)
				}
				return
			}
		}
	}
}

// ---------------------------------------------------------------------------
// File watcher — watches the data file's directory for changes
// ---------------------------------------------------------------------------

func startWatcher(ctx context.Context, hub *Hub, ring *RingBuffer) {
	watcher, err := fsnotify.NewWatcher()
	if err != nil {
		log.Fatalf("fsnotify: %v", err)
	}
	defer watcher.Close()

	dir := filepath.Dir(dataPath)
	if err := watcher.Add(dir); err != nil {
		log.Fatalf("watcher add %s: %v", dir, err)
	}
	log.Printf("watching %s for changes", dir)

	var debounce *time.Timer
	var mu sync.Mutex

	for {
		select {
		case <-ctx.Done():
			if debounce != nil {
				debounce.Stop()
			}
			log.Println("watcher shutting down")
			return
		case event, ok := <-watcher.Events:
			if !ok {
				return
			}
			if event.Op&(fsnotify.Write|fsnotify.Create) == 0 {
				continue
			}
			if filepath.Base(event.Name) != filepath.Base(dataPath) {
				continue
			}
			mu.Lock()
			if debounce != nil {
				debounce.Stop()
			}
			debounce = time.AfterFunc(500*time.Millisecond, func() {
				env, err := buildEnvelope()
				if err != nil {
					log.Printf("build envelope: %v", err)
					return
				}
				body, err := marshalEnvelope(env)
				if err != nil {
					log.Printf("marshal envelope: %v", err)
					return
				}
				ring.Push(env) // record in ring buffer
				notify, _ := json.Marshal(map[string]any{
					"type":     "reload",
					"serverTs": env.ServerTS,
					"bytes":    len(body),
				})
				log.Printf("file changed: %s — reload notify (%d envelope bytes)", event.Name, len(body))
				hub.broadcast(notify)
			})
			mu.Unlock()
		case err, ok := <-watcher.Errors:
			if !ok {
				return
			}
			log.Printf("watcher error: %v", err)
		}
	}
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

func main() {
	distPath := flag.String("dist", "../dist", "path to SPA build output directory")
	flag.StringVar(&dataPath, "data", "", "path to results JSON file (required)")
	extraFlag := flag.String("extra", "", "comma-separated extra result/matrix JSON paths (or BENCH_EXTRA_DATA)")
	port := flag.Int("port", 8090, "listen port")
	flag.Parse()

	if dataPath == "" {
		log.Fatal("flag -data is required")
	}
	if _, err := os.Stat(dataPath); os.IsNotExist(err) {
		log.Fatalf("data file not found: %s", dataPath)
	}
	extraPaths = parseExtraPaths(*extraFlag)

	if *distPath != "" {
		distDir = *distPath
	} else {
		distDir = filepath.Join(".", "dist")
	}
	log.Printf("dist dir  : %s", distDir)
	log.Printf("data path : %s", dataPath)
	if len(extraPaths) > 0 {
		log.Printf("extra data: %s", strings.Join(extraPaths, ", "))
	}
	log.Printf("port      : %d", *port)

	hub := newHub()
	ring := newRingBuffer(30)

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	go startWatcher(ctx, hub, ring)

	// Route table
	mux := http.NewServeMux()
	mux.HandleFunc("/api/health", healthHandler)
	mux.HandleFunc("/api/state", apiStateHandler(ring))
	mux.HandleFunc("/api/data", apiDataHandler(ring))
	mux.HandleFunc("/api/history", apiHistoryHandler(ring))
	mux.HandleFunc("/api/export", apiExportHandler(ring))
	mux.HandleFunc("/api/cells/", apiCellRawHandler(ring))
	mux.HandleFunc("/api/langfuse/status", langfuseStatusHandler)
	mux.HandleFunc("/api/langfuse/setup", langfuseSetupHandler)
	mux.HandleFunc("/api/langfuse/traces", langfuseTracesHandler)
	mux.HandleFunc("/api/langfuse/evaluators", langfuseEvaluatorsHandler)
	mux.HandleFunc("/api/eval/run", portageRunHandler)
	mux.HandleFunc("/api/eval/runs/", portageRunStatusHandler)
	mux.HandleFunc("/api/ws", wsHandler(hub))
	mux.HandleFunc("/ws", wsHandler(hub)) // legacy alias; prefer /api/ws (avoids Vite HMR clash)
	mux.HandleFunc("/", indexHandler)

	handler := logMiddleware(corsMiddleware(securityHeaders(mux)))

	addr := fmt.Sprintf(":%d", *port)
	srv := &http.Server{
		Addr:              addr,
		Handler:           handler,
		ReadHeaderTimeout: 10 * time.Second,
		WriteTimeout:      30 * time.Second,
		IdleTimeout:       120 * time.Second,
	}

	// Start server in a goroutine so we can listen for shutdown.
	go func() {
		log.Printf("listening on %s", addr)
		fmt.Printf("Dashboard -> http://localhost%s\n", addr)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("listen: %v", err)
		}
	}()

	// Block until signal, then drain gracefully.
	<-ctx.Done()
	log.Println("shutting down...")

	shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := srv.Shutdown(shutdownCtx); err != nil {
		log.Printf("shutdown error: %v", err)
	}
	log.Println("server stopped")
}
