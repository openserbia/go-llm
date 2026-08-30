// Package rerank is a client for cross-encoder reranking over HuggingFace's
// Text-Embeddings-Inference (TEI) /rerank endpoint:
//
//	POST /rerank
//	{"query": "...", "texts": ["text 1", "text 2"], "return_documents": false}
//
//	200, a bare array with no envelope:
//	[{"index": 0, "score": 0.95}, {"index": 1, "score": 0.12}]
//
// Reranker products disagree on the wire shape — Cohere and Jina take
// `documents` and answer with a `{"results": [...]}` envelope — so this is
// coded against TEI's native shape. Adapting to another is a small wrapper
// around Rerank, not a rewrite.
//
// A cross-encoder scores each (query, document) pair independently, which
// makes it better calibrated for "is this on topic?" than asking a generative
// model to judge a candidate set: there is no global comparison to drift.
package rerank

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"time"
)

// Result is one (document index, score) pair. Index refers back into the
// documents slice passed to Rerank. Score scales differently per model but is
// monotonic: higher is more relevant.
type Result struct {
	Index int     `json:"index"`
	Score float64 `json:"score"`
}

// HealthInfo is optional metadata from /health. TEI answers with plain text
// "OK" and no body, in which case only Status is populated; backends that
// answer with JSON can fill in the rest.
type HealthInfo struct {
	Status string `json:"status,omitempty"`
	Device string `json:"device,omitempty"`
	Model  string `json:"model,omitempty"`
}

// RetryableError means the backend is not ready yet rather than misconfigured:
// HTTP 503, or a transport failure such as connection refused while the
// container is still starting. Everything else — other 4xx and 5xx, decode
// failures — is returned unwrapped and should not be retried.
type RetryableError struct{ Err error }

// Error implements error.
func (e RetryableError) Error() string { return e.Err.Error() }

// Unwrap exposes the underlying cause.
func (e RetryableError) Unwrap() error { return e.Err }

// Client is the rerank surface.
type Client interface {
	Rerank(ctx context.Context, query string, documents []string) ([]Result, error)
	HealthCheck(ctx context.Context) (*HealthInfo, error)
}

// Options configures a Client.
type Options struct {
	// BaseURL is the reranker root, e.g. "http://localhost:8081". Required.
	BaseURL string

	// APIKey is sent as a Bearer token when non-empty.
	APIKey string

	// Timeout bounds one rerank request. Zero means DefaultTimeout.
	Timeout time.Duration

	// HTTPClient replaces the default transport.
	HTTPClient *http.Client
}

// DefaultTimeout bounds one rerank request when Options.Timeout is zero.
const DefaultTimeout = 30 * time.Second

// errorBodyLimit caps how much of an error response we quote back.
const errorBodyLimit = 4096

func successful(status int) bool {
	return status >= http.StatusOK && status < http.StatusMultipleChoices
}

// HealthProbeTimeout caps a single /health request, so a hung backend cannot
// stall the WaitUntilReady loop. The overall budget for "wait for the model to
// load" belongs on the context passed to WaitUntilReady.
const HealthProbeTimeout = 10 * time.Second

// HTTPClient talks to a TEI-shaped reranker.
type HTTPClient struct {
	opts Options
	http *http.Client
}

var _ Client = (*HTTPClient)(nil)

// New builds a client.
func New(opts Options) (*HTTPClient, error) {
	if opts.BaseURL == "" {
		return nil, errors.New("rerank: BaseURL is required")
	}
	httpClient := opts.HTTPClient
	if httpClient == nil {
		httpClient = &http.Client{}
	}
	return &HTTPClient{opts: opts, http: httpClient}, nil
}

type wireRequest struct {
	Query           string   `json:"query"`
	Texts           []string `json:"texts"`
	ReturnDocuments bool     `json:"return_documents"`
	Truncate        bool     `json:"truncate"`
}

// Rerank scores documents against query. Results come back in the backend's
// order, which for TEI is descending by score. An empty documents slice
// returns nil without contacting the backend.
func (c *HTTPClient) Rerank(ctx context.Context, query string, documents []string) ([]Result, error) {
	if len(documents) == 0 {
		return nil, nil
	}

	timeout := c.opts.Timeout
	if timeout <= 0 {
		timeout = DefaultTimeout
	}
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	buf, err := json.Marshal(wireRequest{
		Query:           query,
		Texts:           documents,
		ReturnDocuments: false,
		// Safety net for documents longer than the model's max input length.
		Truncate: true,
	})
	if err != nil {
		return nil, fmt.Errorf("rerank: encode request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.url("/rerank"), bytes.NewReader(buf))
	if err != nil {
		return nil, fmt.Errorf("rerank: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	c.authorize(req)

	resp, err := c.http.Do(req)
	if err != nil {
		return nil, fmt.Errorf("rerank: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if !successful(resp.StatusCode) {
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, errorBodyLimit))
		return nil, fmt.Errorf("rerank: HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(raw)))
	}

	var out []Result
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, fmt.Errorf("rerank: decode response: %w", err)
	}
	return out, nil
}

// HealthCheck probes /health with a short per-request timeout:
//
//	200            ready; HealthInfo populated when the backend returns JSON.
//	503            RetryableError, the model is still loading.
//	transport error RetryableError, nothing is listening yet.
//	other status   plain error, the backend is misconfigured or broken.
func (c *HTTPClient) HealthCheck(ctx context.Context) (*HealthInfo, error) {
	probeCtx, cancel := context.WithTimeout(ctx, HealthProbeTimeout)
	defer cancel()

	req, err := http.NewRequestWithContext(probeCtx, http.MethodGet, c.url("/health"), http.NoBody)
	if err != nil {
		return nil, fmt.Errorf("rerank: build health request: %w", err)
	}
	c.authorize(req)

	resp, err := c.http.Do(req)
	if err != nil {
		return nil, RetryableError{Err: fmt.Errorf("rerank: health: %w", err)}
	}
	defer func() { _ = resp.Body.Close() }()

	body, _ := io.ReadAll(io.LimitReader(resp.Body, errorBodyLimit))

	switch resp.StatusCode {
	case http.StatusOK:
		// Best-effort decode: a plain-text "OK" leaves the extra fields empty,
		// which is still success.
		info := &HealthInfo{Status: "ok"}
		_ = json.Unmarshal(body, info)
		return info, nil
	case http.StatusServiceUnavailable:
		return nil, RetryableError{Err: fmt.Errorf("rerank: health: HTTP 503: %s", strings.TrimSpace(string(body)))}
	default:
		return nil, fmt.Errorf("rerank: health: HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))
	}
}

func (c *HTTPClient) url(path string) string {
	return strings.TrimRight(c.opts.BaseURL, "/") + path
}

func (c *HTTPClient) authorize(req *http.Request) {
	if c.opts.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.opts.APIKey)
	}
}

// Backoff bounds for WaitUntilReady.
const (
	minBackoff    = time.Second
	maxBackoff    = 10 * time.Second
	backoffFactor = 2
)

// WaitUntilReady polls HealthCheck until the backend is ready, ctx expires, or
// a non-retryable error surfaces. Backoff grows from 1s to 10s, and each wait
// is logged so a long first-time model download is visible rather than looking
// like a hang.
//
// ctx must carry a deadline. Without one, a backend stuck at 503 will be
// polled forever.
func WaitUntilReady(ctx context.Context, c Client, logger *slog.Logger) (*HealthInfo, error) {
	backoff := minBackoff
	for {
		info, err := c.HealthCheck(ctx)
		if err == nil {
			return info, nil
		}

		var retryable RetryableError
		if !errors.As(err, &retryable) {
			return nil, err
		}

		logger.Info("rerank: backend not ready, waiting",
			slog.String("err", err.Error()),
			slog.Duration("backoff", backoff),
		)

		select {
		case <-ctx.Done():
			return nil, fmt.Errorf("rerank: timed out waiting for backend: %w", err)
		case <-time.After(backoff):
		}

		backoff = min(backoff*backoffFactor, maxBackoff)
	}
}
