package llm_test

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/openserbia/go-llm"
)

func modelListServer(t *testing.T, ids ...string) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/models" {
			t.Errorf("requested %s, want /models", r.URL.Path)
		}
		entries := make([]string, 0, len(ids))
		for _, id := range ids {
			entries = append(entries, `{"id":"`+id+`","owned_by":"organization_owner"}`)
		}
		_, _ = w.Write([]byte(`{"object":"list","data":[` + strings.Join(entries, ",") + `]}`))
	}))
	t.Cleanup(srv.Close)
	return srv
}

func TestModelsLists(t *testing.T) {
	srv := modelListServer(t, "gemma-3-12b-it", "bge-m3")

	models, err := newClient(t, srv.URL).Models(context.Background())
	if err != nil {
		t.Fatalf("Models: %v", err)
	}
	if len(models) != 2 || models[0].ID != "gemma-3-12b-it" {
		t.Fatalf("models = %+v", models)
	}
}

func TestEnsureModelFound(t *testing.T) {
	srv := modelListServer(t, "other", "test-model")

	if err := newClient(t, srv.URL).EnsureModel(context.Background(), ""); err != nil {
		t.Fatalf("EnsureModel: %v", err)
	}
}

func TestEnsureModelMissingNamesAlternatives(t *testing.T) {
	srv := modelListServer(t, "gemma-3-12b-it", "qwen3-4b")

	err := newClient(t, srv.URL).EnsureModel(context.Background(), "typo-model")
	if err == nil {
		t.Fatal("EnsureModel succeeded, want error")
	}
	// The point of the error is telling the operator what to type instead.
	for _, want := range []string{"typo-model", "gemma-3-12b-it", "qwen3-4b"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q does not mention %q", err, want)
		}
	}
}

func TestEnsureModelPropagatesTransportFailure(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
	}))
	t.Cleanup(srv.Close)

	err := newClient(t, srv.URL).EnsureModel(context.Background(), "any")
	if llm.Status(err) != http.StatusUnauthorized {
		t.Fatalf("Status(err) = %d, want 401", llm.Status(err))
	}
}
