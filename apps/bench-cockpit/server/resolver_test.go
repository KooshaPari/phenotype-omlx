package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// ---------------------------------------------------------------------------
// firstExisting
// ---------------------------------------------------------------------------

func TestFirstExisting(t *testing.T) {
	dir := t.TempDir()
	a := filepath.Join(dir, "a.json")
	b := filepath.Join(dir, "b.json")
	if err := os.WriteFile(b, []byte("{}"), 0o644); err != nil {
		t.Fatalf("write b: %v", err)
	}
	// a missing, b exists → returns b
	if got := firstExisting([]string{a, b}); got != b {
		t.Fatalf("firstExisting wrong: got %q want %q", got, b)
	}
	// all missing → ""
	if got := firstExisting([]string{a, filepath.Join(dir, "nope")}); got != "" {
		t.Fatalf("all missing should return empty, got %q", got)
	}
	// empty strings are skipped
	if got := firstExisting([]string{"", b}); got != b {
		t.Fatalf("empty string should be skipped, got %q", got)
	}
	// prefer first existing
	if err := os.WriteFile(a, []byte("{}"), 0o644); err != nil {
		t.Fatalf("write a: %v", err)
	}
	if got := firstExisting([]string{a, b}); got != a {
		t.Fatalf("prefer first existing: got %q want %q", got, a)
	}
}

// ---------------------------------------------------------------------------
// repoCandidates — shape/behavior
// ---------------------------------------------------------------------------

func TestRepoCandidatesCoversGivenRel(t *testing.T) {
	rels := []string{"data/foo.json", "fixtures/bar.json"}
	got := repoCandidates(rels...)
	if len(got) == 0 {
		t.Fatal("repoCandidates returned nothing")
	}
	for _, r := range rels {
		found := false
		for _, p := range got {
			if strings.HasSuffix(filepath.ToSlash(p), r) {
				found = true
				break
			}
		}
		if !found {
			t.Fatalf("repoCandidates missing rel %q in %v", r, got)
		}
	}
	// All results should be absolute-joined (contain the rel as suffix).
	// Duplicates across wd/exe roots are allowed; we just check count >= rels*roots.
	if got2 := repoCandidates("only/one.json"); len(got2) < 2 {
		t.Fatalf("repoCandidates too few roots: %v", got2)
	}
}

// ---------------------------------------------------------------------------
// resolveDataPath
// ---------------------------------------------------------------------------

func TestResolveDataPath_BenchDataEnvOverride(t *testing.T) {
	// BENCH_DATA takes absolute precedence even when other files exist.
	f := filepath.Join(t.TempDir(), "override.json")
	if err := os.WriteFile(f, []byte("{}"), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	t.Setenv("BENCH_DATA", f)
	// Even though default candidates won't exist in this env, BENCH_DATA must win.
	// Save/restore defaultDataCandidates so we don't leak mutated slice.
	orig := defaultDataCandidates
	defaultDataCandidates = []string{filepath.Join(t.TempDir(), "should-not-exist.json")}
	t.Cleanup(func() { defaultDataCandidates = orig })
	if got := resolveDataPath(); got != f {
		t.Fatalf("BENCH_DATA override failed: got %q want %q", got, f)
	}
}

func TestResolveDataPath_BenchDataEnvMissingFileStillReturned(t *testing.T) {
	// Current behavior: BENCH_DATA is returned verbatim without existence check.
	// main() will later Fatal if it doesn't exist — that's tested at the call site.
	nonexistent := filepath.Join(t.TempDir(), "ghost.json")
	t.Setenv("BENCH_DATA", nonexistent)
	orig := defaultDataCandidates
	defaultDataCandidates = []string{}
	t.Cleanup(func() { defaultDataCandidates = orig })
	if got := resolveDataPath(); got != nonexistent {
		t.Fatalf("BENCH_DATA should be returned verbatim: got %q want %q", got, nonexistent)
	}
}

func TestResolveDataPath_CanonicalCandidate(t *testing.T) {
	t.Setenv("BENCH_DATA", "")
	dir := t.TempDir()
	p := filepath.Join(dir, "matrix.json")
	if err := os.WriteFile(p, []byte(`{"suites":[]}`), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	orig := defaultDataCandidates
	defaultDataCandidates = []string{p}
	t.Cleanup(func() { defaultDataCandidates = orig })
	if got := resolveDataPath(); got != p {
		t.Fatalf("canonical candidate: got %q want %q", got, p)
	}
}

func TestResolveDataPath_RepoCandidateFallback(t *testing.T) {
	// When BENCH_DATA empty and canonical missing, resolver should reach repoCandidates.
	t.Setenv("BENCH_DATA", "")
	orig := defaultDataCandidates
	defaultDataCandidates = []string{filepath.Join(t.TempDir(), "absent.json")}
	t.Cleanup(func() { defaultDataCandidates = orig })

	// Put a fixture under a temp repo-like root and chdir there so repoCandidates can find it.
	root := t.TempDir()
	dataDir := filepath.Join(root, "data")
	if err := os.MkdirAll(dataDir, 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	candidatePath := filepath.Join(dataDir, "run-v5-qwen35-08b.json")
	if err := os.WriteFile(candidatePath, []byte(`{}`), 0o644); err != nil {
		t.Fatalf("write candidate: %v", err)
	}
	origWd, _ := os.Getwd()
	if err := os.Chdir(root); err != nil {
		t.Fatalf("chdir: %v", err)
	}
	t.Cleanup(func() { _ = os.Chdir(origWd) })
	got := resolveDataPath()
	gotReal, err := filepath.EvalSymlinks(got)
	if err != nil {
		t.Fatalf("resolve path should exist: %v", err)
	}
	wantReal, err := filepath.EvalSymlinks(candidatePath)
	if err != nil {
		t.Fatalf("candidate path should exist: %v", err)
	}
	if gotReal != wantReal {
		t.Fatalf("repo candidate fallback: got %q want %q", gotReal, wantReal)
	}
}

func TestResolveDataPath_NoCandidatesReturnsEmpty(t *testing.T) {
	t.Setenv("BENCH_DATA", "")
	orig := defaultDataCandidates
	defaultDataCandidates = []string{filepath.Join(t.TempDir(), "nope.json")}
	t.Cleanup(func() { defaultDataCandidates = orig })
	// Chdir to empty dir so repoCandidates finds nothing.
	root := t.TempDir()
	origWd, _ := os.Getwd()
	if err := os.Chdir(root); err != nil {
		t.Fatalf("chdir: %v", err)
	}
	t.Cleanup(func() { _ = os.Chdir(origWd) })
	if got := resolveDataPath(); got != "" {
		t.Fatalf("no candidates should give empty: got %q", got)
	}
}

// ---------------------------------------------------------------------------
// resolveExtraDefaults — BENCH_EXTRA_DATA, dedup, self-extra skip
// ---------------------------------------------------------------------------

func TestResolveExtraDefaults_BenchExtraData(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "extra.json")
	if err := os.WriteFile(p, []byte(`{}`), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	t.Setenv("BENCH_EXTRA_DATA", p)
	orig := defaultExtraCandidates
	defaultExtraCandidates = []string{}
	t.Cleanup(func() { defaultExtraCandidates = orig })
	// Ensure dataPath points elsewhere so self-extra skip doesn't drop it.
	dataPath = filepath.Join(t.TempDir(), "different.json")
	t.Cleanup(func() { dataPath = "" })
	got := resolveExtraDefaults()
	if len(got) != 1 || got[0] != p {
		t.Fatalf("BENCH_EXTRA_DATA: got %v want [%q]", got, p)
	}
}

func TestResolveExtraDefaults_SkipsSelfExtra(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "same.json")
	if err := os.WriteFile(p, []byte(`{}`), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	t.Setenv("BENCH_EXTRA_DATA", "")
	orig := defaultExtraCandidates
	defaultExtraCandidates = []string{p}
	t.Cleanup(func() { defaultExtraCandidates = orig })
	dataPath = p
	t.Cleanup(func() { dataPath = "" })
	if got := resolveExtraDefaults(); len(got) != 0 {
		t.Fatalf("self-extra should be skipped: got %v", got)
	}
}

func TestResolveExtraDefaults_DedupByAbs(t *testing.T) {
	// Same file referenced two ways should appear once.
	dir := t.TempDir()
	p := filepath.Join(dir, "matrix.json")
	if err := os.WriteFile(p, []byte(`{}`), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	absP, _ := filepath.Abs(p)
	// BENCH_EXTRA_DATA duplicate of the canonical candidate.
	t.Setenv("BENCH_EXTRA_DATA", absP)
	orig := defaultExtraCandidates
	defaultExtraCandidates = []string{p}
	t.Cleanup(func() { defaultExtraCandidates = orig })
	dataPath = filepath.Join(t.TempDir(), "other.json")
	t.Cleanup(func() { dataPath = "" })
	got := resolveExtraDefaults()
	if len(got) != 1 {
		t.Fatalf("dedup expected 1, got %d: %v", len(got), got)
	}
}

func TestResolveExtraDefaults_MissingExtraDropped(t *testing.T) {
	// BENCH_EXTRA_DATA pointing nowhere is dropped (current behavior: firstExisting returns empty).
	t.Setenv("BENCH_EXTRA_DATA", "")
	orig := defaultExtraCandidates
	defaultExtraCandidates = []string{filepath.Join(t.TempDir(), "absent.json")}
	t.Cleanup(func() { defaultExtraCandidates = orig })
	dataPath = filepath.Join(t.TempDir(), "other.json")
	t.Cleanup(func() { dataPath = "" })
	if got := resolveExtraDefaults(); len(got) != 0 {
		t.Fatalf("missing extra should be dropped: %v", got)
	}
}
