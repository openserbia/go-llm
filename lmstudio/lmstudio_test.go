package lmstudio_test

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/openserbia/go-llm/lmstudio"
)

// modelsBody is a trimmed GET /api/v1/models response: one loaded chat model
// with a context window smaller than its architectural maximum, one unloaded
// chat model, and one embedding model.
const modelsBody = `{
  "models": [
    {
      "key": "gemma-3-12b-it",
      "display_name": "Gemma 3 12B",
      "type": "llm",
      "max_context_length": 131072,
      "capabilities": {
        "vision": true,
        "trained_for_tool_use": true,
        "reasoning": {"allowed_options": ["off", "low", "high"]}
      },
      "loaded_instances": [{"config": {"context_length": 8192}}]
    },
    {
      "key": "qwen3-4b",
      "display_name": "Qwen3 4B",
      "type": "llm",
      "max_context_length": 32768,
      "capabilities": {"vision": false, "trained_for_tool_use": false},
      "loaded_instances": []
    },
    {
      "key": "bge-m3",
      "type": "embedding",
      "max_context_length": 8192,
      "loaded_instances": [{"config": {"context_length": 8192}}]
    }
  ]
}`

func modelsServer(t *testing.T) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/v1/models" {
			t.Errorf("requested %s, want /api/v1/models", r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
			return
		}
		_, _ = w.Write([]byte(modelsBody))
	}))
	t.Cleanup(srv.Close)
	return srv
}

func newClient(t *testing.T, baseURL string) *lmstudio.Client {
	t.Helper()
	c, err := lmstudio.New(lmstudio.Options{BaseURL: baseURL})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return c
}

func TestServerRootStripsVersionSegment(t *testing.T) {
	tests := map[string]string{
		"http://localhost:1234":         "http://localhost:1234",
		"http://localhost:1234/":        "http://localhost:1234",
		"http://localhost:1234/v1":      "http://localhost:1234",
		"http://localhost:1234/v1/":     "http://localhost:1234",
		"http://localhost:1234/api/v1":  "http://localhost:1234",
		"http://192.168.1.5:1234/v1":    "http://192.168.1.5:1234",
		"https://gpu.example.com/proxy": "https://gpu.example.com/proxy",
	}
	for in, want := range tests {
		if got := lmstudio.ServerRoot(in); got != want {
			t.Errorf("ServerRoot(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestModelsParsesCapabilities(t *testing.T) {
	srv := modelsServer(t)

	// The OpenAI-compatible base URL must work here without the caller
	// rewriting it.
	models, err := newClient(t, srv.URL+"/v1").Models(context.Background())
	if err != nil {
		t.Fatalf("Models: %v", err)
	}
	if len(models) != 3 {
		t.Fatalf("got %d models, want 3", len(models))
	}

	gemma := models[0]
	if !gemma.Capabilities.Vision || !gemma.Capabilities.ToolUse {
		t.Errorf("gemma capabilities = %+v, want vision and tool use", gemma.Capabilities)
	}
	if !gemma.SupportsReasoning() {
		t.Error("SupportsReasoning = false, want true")
	}
}

func TestContextLengthReportsLoadedNotMaximum(t *testing.T) {
	srv := modelsServer(t)

	loaded, err := newClient(t, srv.URL).Model(context.Background(), "gemma-3-12b-it")
	if err != nil {
		t.Fatalf("Model: %v", err)
	}
	// The whole point of the native API: 8192 is what was allocated, 131072 is
	// what the architecture allows. Prompts have to fit the first one.
	if got := loaded.ContextLength(); got != 8192 {
		t.Errorf("ContextLength = %d, want 8192", got)
	}
	if loaded.MaxContextLength != 131072 {
		t.Errorf("MaxContextLength = %d, want 131072", loaded.MaxContextLength)
	}
	if !loaded.Loaded() {
		t.Error("Loaded = false, want true")
	}
}

func TestContextLengthUnknownWhenUnloaded(t *testing.T) {
	srv := modelsServer(t)

	unloaded, err := newClient(t, srv.URL).Model(context.Background(), "qwen3-4b")
	if err != nil {
		t.Fatalf("Model: %v", err)
	}
	if unloaded.Loaded() {
		t.Error("Loaded = true, want false")
	}
	// 0, not MaxContextLength: an unloaded model has no allocated window, and
	// substituting the maximum would oversize every prompt built from it.
	if got := unloaded.ContextLength(); got != 0 {
		t.Errorf("ContextLength = %d, want 0", got)
	}
}

func TestModelNotFound(t *testing.T) {
	srv := modelsServer(t)

	_, err := newClient(t, srv.URL).Model(context.Background(), "no-such-model")
	if !errors.Is(err, lmstudio.ErrModelNotFound) {
		t.Fatalf("err = %v, want ErrModelNotFound", err)
	}
}

func TestLLMsFiltersEmbeddingModels(t *testing.T) {
	srv := modelsServer(t)

	models, err := newClient(t, srv.URL).Models(context.Background())
	if err != nil {
		t.Fatalf("Models: %v", err)
	}
	llms := lmstudio.LLMs(models)
	if len(llms) != 2 {
		t.Fatalf("got %d chat models, want 2", len(llms))
	}
	for _, m := range llms {
		if m.Type != lmstudio.TypeLLM {
			t.Errorf("model %q has type %q", m.Key, m.Type)
		}
	}
}

func discardLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(io.Discard, nil))
}

func TestEnsureLoadedReturnsLoadedModel(t *testing.T) {
	srv := modelsServer(t)

	model, err := lmstudio.EnsureLoaded(context.Background(), lmstudio.PreflightOptions{
		Options: lmstudio.Options{BaseURL: srv.URL + "/v1"},
		Model:   "gemma-3-12b-it",
		Logger:  discardLogger(),
	})
	if err != nil {
		t.Fatalf("EnsureLoaded: %v", err)
	}
	if model.ContextLength() != 8192 {
		t.Errorf("ContextLength = %d, want 8192", model.ContextLength())
	}
}

func TestEnsureLoadedRefusesUnloadedWithoutStrategy(t *testing.T) {
	srv := modelsServer(t)

	// Neither UseCLI nor WarmUp: claiming VRAM is not something a preflight
	// should do behind the caller's back.
	_, err := lmstudio.EnsureLoaded(context.Background(), lmstudio.PreflightOptions{
		Options: lmstudio.Options{BaseURL: srv.URL},
		Model:   "qwen3-4b",
		Logger:  discardLogger(),
	})
	if err == nil {
		t.Fatal("EnsureLoaded succeeded on an unloaded model, want error")
	}
}

func TestEnsureLoadedWarmUpLoadsModel(t *testing.T) {
	var warmed atomic.Bool

	mux := http.NewServeMux()
	mux.HandleFunc("/api/v1/models", func(w http.ResponseWriter, _ *http.Request) {
		instances := `[]`
		if warmed.Load() {
			instances = `[{"config": {"context_length": 4096}}]`
		}
		_, _ = w.Write([]byte(`{"models":[{"key":"qwen3-4b","type":"llm","max_context_length":32768,` +
			`"loaded_instances":` + instances + `}]}`))
	})
	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, _ *http.Request) {
		warmed.Store(true)
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"hi"}}]}`))
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)

	model, err := lmstudio.EnsureLoaded(context.Background(), lmstudio.PreflightOptions{
		Options: lmstudio.Options{BaseURL: srv.URL + "/v1"},
		Model:   "qwen3-4b",
		WarmUp:  true,
		Logger:  discardLogger(),
	})
	if err != nil {
		t.Fatalf("EnsureLoaded: %v", err)
	}
	if !warmed.Load() {
		t.Error("warm-up request was never sent")
	}
	if model.ContextLength() != 4096 {
		t.Errorf("ContextLength = %d, want 4096 after load", model.ContextLength())
	}
}

func TestEnsureLoadedRequiresModel(t *testing.T) {
	_, err := lmstudio.EnsureLoaded(context.Background(), lmstudio.PreflightOptions{
		Options: lmstudio.Options{BaseURL: "http://127.0.0.1:1"},
	})
	if err == nil {
		t.Fatal("EnsureLoaded succeeded without a model, want error")
	}
}

func TestEnsureLoadedWarmsUpEmbeddingModelViaEmbeddings(t *testing.T) {
	var (
		warmed       atomic.Bool
		chatAttempts atomic.Int32
	)

	mux := http.NewServeMux()
	mux.HandleFunc("/api/v1/models", func(w http.ResponseWriter, _ *http.Request) {
		instances := `[]`
		if warmed.Load() {
			instances = `[{"config": {"context_length": 8192}}]`
		}
		_, _ = w.Write([]byte(`{"models":[{"key":"bge-m3","type":"embedding","max_context_length":8192,` +
			`"loaded_instances":` + instances + `}]}`))
	})
	mux.HandleFunc("/v1/embeddings", func(w http.ResponseWriter, _ *http.Request) {
		warmed.Store(true)
		_, _ = w.Write([]byte(`{"data":[{"embedding":[0.1,0.2]}]}`))
	})
	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, _ *http.Request) {
		// An embedding model has no chat endpoint; reaching this would be the
		// warm-up asking the wrong question.
		chatAttempts.Add(1)
		w.WriteHeader(http.StatusBadRequest)
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)

	model, err := lmstudio.EnsureLoaded(context.Background(), lmstudio.PreflightOptions{
		Options: lmstudio.Options{BaseURL: srv.URL + "/v1"},
		Model:   "bge-m3",
		WarmUp:  true,
		Logger:  discardLogger(),
	})
	if err != nil {
		t.Fatalf("EnsureLoaded: %v", err)
	}
	if got := chatAttempts.Load(); got != 0 {
		t.Errorf("chat endpoint hit %d times for an embedding model, want 0", got)
	}
	if model.ContextLength() != 8192 {
		t.Errorf("ContextLength = %d, want 8192", model.ContextLength())
	}
}
