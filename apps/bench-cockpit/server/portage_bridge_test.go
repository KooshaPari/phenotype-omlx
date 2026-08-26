package main

import (
	"os"
	"strings"
	"testing"
)

func TestHarborEnvironmentDefaultsToAppleContainer(t *testing.T) {
	// FR-HARBOR-APPLE-001
	t.Setenv("HARBOR_ENV", "")

	got, err := harborEnvironment()
	if err != nil {
		t.Fatalf("harborEnvironment: %v", err)
	}
	if got != "apple-container" {
		t.Fatalf("environment=%q want apple-container", got)
	}
}

func TestHarborEnvironmentRejectsNonAppleRuntimes(t *testing.T) {
	// FR-HARBOR-APPLE-001
	for _, value := range []string{"docker", "podman", "unknown"} {
		t.Run(value, func(t *testing.T) {
			t.Setenv("HARBOR_ENV", value)
			_, err := harborEnvironment()
			if err == nil {
				t.Fatalf("HARBOR_ENV=%q should be rejected", value)
			}
			if !strings.Contains(err.Error(), "-e docker --ek container_runtime=podman") {
				t.Fatalf("error should document Harbor's Podman override, got %q", err)
			}
		})
	}
}

func TestPortageCommandEnvironmentPrependsLocalBin(t *testing.T) {
	// FR-HARBOR-APPLE-001
	t.Setenv("PATH", "/usr/bin:/bin")

	env := portageCommandEnvironment()
	want := "PATH=/usr/local/bin" + string(os.PathListSeparator) + "/usr/bin:/bin"
	for _, entry := range env {
		if strings.HasPrefix(entry, "PATH=") {
			if entry != want {
				t.Fatalf("PATH entry=%q want %q", entry, want)
			}
			return
		}
	}
	t.Fatal("PATH missing from command environment")
}

func TestNormalizeJobEnvironmentDefaultsToAppleContainer(t *testing.T) {
	// FR-HARBOR-APPLE-001
	job := map[string]interface{}{"tasks": []interface{}{}}

	if err := normalizeJobEnvironment(job); err != nil {
		t.Fatalf("normalizeJobEnvironment: %v", err)
	}
	environment, ok := job["environment"].(map[string]interface{})
	if !ok {
		t.Fatalf("environment=%#v", job["environment"])
	}
	if environment["type"] != "apple-container" {
		t.Fatalf("environment type=%#v want apple-container", environment["type"])
	}
}

func TestNormalizeJobEnvironmentRejectsDocker(t *testing.T) {
	// FR-HARBOR-APPLE-001
	job := map[string]interface{}{
		"tasks":       []interface{}{},
		"environment": map[string]interface{}{"type": "docker"},
	}

	err := normalizeJobEnvironment(job)
	if err == nil || !strings.Contains(err.Error(), "docker") {
		t.Fatalf("expected docker rejection, got %v", err)
	}
}
