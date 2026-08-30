// Package embed is a client for OpenAI-compatible /embeddings endpoints.
package embed

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"strings"
	"time"
)

// DefaultTimeout bounds one embedding request when Options.Timeout is zero.
const DefaultTimeout = time.Minute

// errorBodyLimit caps how much of an error response we quote back.
const errorBodyLimit = 4096

func successful(status int) bool {
	return status >= http.StatusOK && status < http.StatusMultipleChoices
}

// Options configures a Client.
type Options struct {
	// BaseURL is the API root including any version segment, e.g.
	// "http://localhost:1234/v1". Required.
	BaseURL string

	// APIKey is sent as a Bearer token when non-empty.
	APIKey string

	// Model names the embedding model. Required by most backends.
	Model string

	// Dimensions, when non-zero, is asserted against the first vector the
	// backend returns. This catches the failure that is otherwise silent and
	// expensive: pointing at a different model than the one whose width your
	// storage was provisioned for, and only finding out when writes start
	// failing partway through a long run.
	Dimensions int

	// Normalize scales each vector to unit length. Do this when your distance
	// metric is cosine or inner product and your storage expects normalized
	// input; leave it off when the backend already normalizes or when you need
	// the raw magnitudes.
	Normalize bool

	// Timeout bounds one request via context. Zero means DefaultTimeout.
	Timeout time.Duration

	// HTTPClient replaces the default transport.
	HTTPClient *http.Client
}

// DimensionMismatchError reports that the backend returned vectors of an
// unexpected width.
type DimensionMismatchError struct {
	Model     string
	Got, Want int
}

// Error implements error.
func (e *DimensionMismatchError) Error() string {
	return fmt.Sprintf("embed: model %q returned %d-dimensional vectors, expected %d", e.Model, e.Got, e.Want)
}

// Client is the embedding surface.
type Client interface {
	Embed(ctx context.Context, inputs []string) ([][]float32, error)
}

// HTTPClient talks to an OpenAI-compatible embeddings endpoint.
type HTTPClient struct {
	opts       Options
	http       *http.Client
	dimChecked bool
}

var _ Client = (*HTTPClient)(nil)

// New builds a client.
func New(opts Options) (*HTTPClient, error) {
	if opts.BaseURL == "" {
		return nil, errors.New("embed: BaseURL is required")
	}
	httpClient := opts.HTTPClient
	if httpClient == nil {
		httpClient = &http.Client{}
	}
	return &HTTPClient{opts: opts, http: httpClient}, nil
}

type wireRequest struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
}

type wireResponse struct {
	Data []struct {
		Embedding []float32 `json:"embedding"`
	} `json:"data"`
}

// Embed returns one vector per input, in input order. An empty input slice
// returns nil without contacting the backend.
//
// Batch size is the caller's choice: this sends whatever it is given as a
// single request. Backends differ in how many inputs and how many tokens they
// accept per call, so the split belongs with the code that knows the model.
func (c *HTTPClient) Embed(ctx context.Context, inputs []string) ([][]float32, error) {
	if len(inputs) == 0 {
		return nil, nil
	}

	timeout := c.opts.Timeout
	if timeout <= 0 {
		timeout = DefaultTimeout
	}
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	body, err := json.Marshal(wireRequest{Model: c.opts.Model, Input: inputs})
	if err != nil {
		return nil, fmt.Errorf("embed: encode request: %w", err)
	}

	url := strings.TrimRight(c.opts.BaseURL, "/") + "/embeddings"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("embed: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if c.opts.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.opts.APIKey)
	}

	resp, err := c.http.Do(req)
	if err != nil {
		return nil, fmt.Errorf("embed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if !successful(resp.StatusCode) {
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, errorBodyLimit))
		return nil, fmt.Errorf("embed: HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(raw)))
	}

	var out wireResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, fmt.Errorf("embed: decode response: %w", err)
	}
	if len(out.Data) != len(inputs) {
		return nil, fmt.Errorf("embed: expected %d vectors, got %d", len(inputs), len(out.Data))
	}

	vectors := make([][]float32, len(out.Data))
	for i, item := range out.Data {
		if c.opts.Dimensions > 0 && !c.dimChecked {
			if len(item.Embedding) != c.opts.Dimensions {
				return nil, &DimensionMismatchError{Model: c.opts.Model, Got: len(item.Embedding), Want: c.opts.Dimensions}
			}
			c.dimChecked = true
		}
		if c.opts.Normalize {
			vectors[i] = L2Normalize(item.Embedding)
			continue
		}
		vectors[i] = item.Embedding
	}
	return vectors, nil
}

// L2Normalize returns v scaled to unit length. A zero vector is returned
// unchanged.
func L2Normalize(v []float32) []float32 {
	var sumsq float64
	for _, x := range v {
		sumsq += float64(x) * float64(x)
	}
	if sumsq == 0 {
		return v
	}
	norm := float32(math.Sqrt(sumsq))
	out := make([]float32, len(v))
	for i, x := range v {
		out[i] = x / norm
	}
	return out
}

// Truncate clamps s to maxChars runes. A non-positive maxChars returns s
// unchanged.
func Truncate(s string, maxChars int) string {
	if maxChars <= 0 {
		return s
	}
	r := []rune(s)
	if len(r) <= maxChars {
		return s
	}
	return string(r[:maxChars])
}
