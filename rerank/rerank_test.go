package rerank_test

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/openserbia/go-llm/rerank"
)

func discardLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(io.Discard, nil))
}

func TestRerankDecodesBareArray(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		// TEI answers with a bare array, no envelope.
		_, _ = w.Write([]byte(`[{"index":2,"score":0.91},{"index":0,"score":0.12}]`))
	}))
	t.Cleanup(srv.Close)

	client, err := rerank.New(rerank.Options{BaseURL: srv.URL})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	got, err := client.Rerank(context.Background(), "query", []string{"a", "b", "c"})
	if err != nil {
		t.Fatalf("Rerank: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("got %d results, want 2", len(got))
	}
	if got[0].Index != 2 || got[0].Score != 0.91 {
		t.Errorf("results[0] = %+v, want {Index:2 Score:0.91}", got[0])
	}
}

func TestRerankEmptyDocumentsSkipsRequest(t *testing.T) {
	called := false
	srv := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { called = true }))
	t.Cleanup(srv.Close)

	client, err := rerank.New(rerank.Options{BaseURL: srv.URL})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if _, err := client.Rerank(context.Background(), "query", nil); err != nil {
		t.Fatalf("Rerank: %v", err)
	}
	if called {
		t.Error("empty documents still contacted the backend")
	}
}

func TestHealthCheck503IsRetryable(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte("model is loading"))
	}))
	t.Cleanup(srv.Close)

	client, err := rerank.New(rerank.Options{BaseURL: srv.URL})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, err = client.HealthCheck(context.Background())

	var retryable rerank.RetryableError
	if !errors.As(err, &retryable) {
		t.Fatalf("err = %v, want rerank.RetryableError", err)
	}
}

func TestHealthCheck404IsNotRetryable(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNotFound)
	}))
	t.Cleanup(srv.Close)

	client, err := rerank.New(rerank.Options{BaseURL: srv.URL})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, err = client.HealthCheck(context.Background())

	var retryable rerank.RetryableError
	if errors.As(err, &retryable) {
		t.Error("404 classified as retryable; a wrong URL never becomes right by waiting")
	}
	if err == nil {
		t.Fatal("HealthCheck succeeded on 404, want error")
	}
}

func TestHealthCheckPlainTextOK(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		// TEI's /health is plain text, not JSON.
		_, _ = w.Write([]byte("OK"))
	}))
	t.Cleanup(srv.Close)

	client, err := rerank.New(rerank.Options{BaseURL: srv.URL})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	info, err := client.HealthCheck(context.Background())
	if err != nil {
		t.Fatalf("HealthCheck: %v", err)
	}
	if info.Status != "ok" {
		t.Errorf("Status = %q, want ok", info.Status)
	}
}

// flakyClient is unready for the first n probes, then healthy.
type flakyClient struct {
	unreadyFor int32
	calls      atomic.Int32
}

func (f *flakyClient) Rerank(context.Context, string, []string) ([]rerank.Result, error) {
	return nil, nil
}

func (f *flakyClient) HealthCheck(context.Context) (*rerank.HealthInfo, error) {
	if f.calls.Add(1) <= f.unreadyFor {
		return nil, rerank.RetryableError{Err: errors.New("HTTP 503: loading")}
	}
	return &rerank.HealthInfo{Status: "ok", Model: "bge-reranker"}, nil
}

func TestWaitUntilReadyPollsUntilHealthy(t *testing.T) {
	client := &flakyClient{unreadyFor: 2}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	t.Cleanup(cancel)

	info, err := rerank.WaitUntilReady(ctx, client, discardLogger())
	if err != nil {
		t.Fatalf("WaitUntilReady: %v", err)
	}
	if info.Model != "bge-reranker" {
		t.Errorf("Model = %q, want bge-reranker", info.Model)
	}
	if got := client.calls.Load(); got != 3 {
		t.Errorf("probes = %d, want 3", got)
	}
}

// deadClient is never ready.
type deadClient struct{}

func (deadClient) Rerank(context.Context, string, []string) ([]rerank.Result, error) { return nil, nil }

func (deadClient) HealthCheck(context.Context) (*rerank.HealthInfo, error) {
	return nil, rerank.RetryableError{Err: errors.New("HTTP 503")}
}

func TestWaitUntilReadyRespectsDeadline(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	t.Cleanup(cancel)

	start := time.Now()
	if _, err := rerank.WaitUntilReady(ctx, deadClient{}, discardLogger()); err == nil {
		t.Fatal("WaitUntilReady succeeded, want deadline error")
	}
	if elapsed := time.Since(start); elapsed > 5*time.Second {
		t.Errorf("returned after %v, want the ctx deadline to cut it short", elapsed)
	}
}

// brokenClient fails in a way waiting cannot fix.
type brokenClient struct{ calls int }

func (b *brokenClient) Rerank(context.Context, string, []string) ([]rerank.Result, error) {
	return nil, nil
}

func (b *brokenClient) HealthCheck(context.Context) (*rerank.HealthInfo, error) {
	b.calls++
	return nil, errors.New("HTTP 401: unauthorized")
}

func TestWaitUntilReadyGivesUpOnNonRetryable(t *testing.T) {
	client := &brokenClient{}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	t.Cleanup(cancel)

	if _, err := rerank.WaitUntilReady(ctx, client, discardLogger()); err == nil {
		t.Fatal("WaitUntilReady succeeded, want error")
	}
	if client.calls != 1 {
		t.Errorf("probes = %d, want 1 — bad credentials do not resolve by waiting", client.calls)
	}
}
