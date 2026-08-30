// Package lmstudio reads LM Studio's native REST API and prepares models for
// use.
//
// LM Studio serves two APIs on the same port. The OpenAI-compatible one under
// /v1 is what you send inference to, and its /v1/models returns bare
// identifiers. The native one under /api/v1 additionally reports each model's
// capabilities and — for a model that is currently loaded — the context length
// it was actually loaded with, which is not the same number as the model's
// theoretical maximum and is the one your prompts have to fit inside.
//
// Reading it up front is how you size a prompt before sending it, instead of
// discovering the limit by parsing an overflow error afterwards.
package lmstudio

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// DefaultTimeout bounds one native API request when Options.Timeout is zero.
const DefaultTimeout = 10 * time.Second

// errorBodyLimit caps how much of an error response we quote back.
const errorBodyLimit = 4096

func successful(status int) bool {
	return status >= http.StatusOK && status < http.StatusMultipleChoices
}

// ErrModelNotFound means the server does not know the requested model, whether
// or not it is downloaded.
var ErrModelNotFound = errors.New("lmstudio: model not found")

// Model type values reported by the native API.
const (
	TypeLLM       = "llm"
	TypeEmbedding = "embedding"
)

// Model is one entry from GET /api/v1/models.
type Model struct {
	Key         string `json:"key"`
	DisplayName string `json:"display_name"`
	// Type distinguishes chat models from embedding models; see TypeLLM.
	Type string `json:"type"`
	// MaxContextLength is what the model architecture supports, which can be
	// far larger than what any loaded instance was given. Use ContextLength
	// for the number that binds your prompts.
	MaxContextLength int              `json:"max_context_length"`
	Capabilities     Capabilities     `json:"capabilities"`
	LoadedInstances  []LoadedInstance `json:"loaded_instances"`
}

// Capabilities describes what the model can be asked to do.
type Capabilities struct {
	Vision bool `json:"vision"`
	// ToolUse reports training for tool calling. A model without it will
	// usually still emit something when given tools, just unreliably.
	ToolUse   bool       `json:"trained_for_tool_use"`
	Reasoning *Reasoning `json:"reasoning"`
}

// Reasoning is present for models with an adjustable reasoning budget.
type Reasoning struct {
	AllowedOptions []string `json:"allowed_options"`
}

// LoadedInstance is one running copy of a model.
type LoadedInstance struct {
	Config InstanceConfig `json:"config"`
}

// InstanceConfig is the load-time configuration of a running instance.
type InstanceConfig struct {
	ContextLength int `json:"context_length"`
}

// Loaded reports whether at least one instance is running.
func (m Model) Loaded() bool { return len(m.LoadedInstances) > 0 }

// ContextLength returns the context window the model is loaded with, or 0 when
// it is not loaded. LM Studio does not expose the saved load configuration of
// an unloaded model, so 0 means "unknowable until it loads" rather than
// "unlimited" — do not substitute MaxContextLength for it, which would size
// prompts against a window that was never allocated.
func (m Model) ContextLength() int {
	for _, inst := range m.LoadedInstances {
		if inst.Config.ContextLength > 0 {
			return inst.Config.ContextLength
		}
	}
	return 0
}

// SupportsReasoning reports whether the model exposes reasoning effort levels.
func (m Model) SupportsReasoning() bool {
	return m.Reasoning() != nil && len(m.Reasoning().AllowedOptions) > 0
}

// Reasoning returns the reasoning capability block, or nil.
func (m Model) Reasoning() *Reasoning { return m.Capabilities.Reasoning }

// Options configures a native API client.
type Options struct {
	// BaseURL is the LM Studio server address. Either the server root
	// ("http://localhost:1234") or the OpenAI-compatible root
	// ("http://localhost:1234/v1") is accepted — a trailing version segment is
	// stripped, so the same value you configured for inference works here.
	BaseURL string

	// APIKey is sent as a Bearer token when non-empty. LM Studio ignores it
	// unless configured otherwise.
	APIKey string

	// Timeout bounds one request. Zero means DefaultTimeout.
	Timeout time.Duration

	// HTTPClient replaces the default transport.
	HTTPClient *http.Client
}

// Client reads LM Studio's native API.
type Client struct {
	root    string
	apiKey  string
	timeout time.Duration
	http    *http.Client
}

// New builds a native API client.
func New(opts Options) (*Client, error) {
	if opts.BaseURL == "" {
		return nil, errors.New("lmstudio: BaseURL is required")
	}
	httpClient := opts.HTTPClient
	if httpClient == nil {
		httpClient = &http.Client{}
	}
	timeout := opts.Timeout
	if timeout <= 0 {
		timeout = DefaultTimeout
	}
	return &Client{
		root:    ServerRoot(opts.BaseURL),
		apiKey:  opts.APIKey,
		timeout: timeout,
		http:    httpClient,
	}, nil
}

// ServerRoot strips a trailing OpenAI-compatible version segment from a base
// URL, so that a value configured for inference can address the native API.
func ServerRoot(baseURL string) string {
	root := strings.TrimRight(baseURL, "/")
	for _, suffix := range []string{"/api/v1", "/v1"} {
		if strings.HasSuffix(root, suffix) {
			return strings.TrimSuffix(root, suffix)
		}
	}
	return root
}

type modelsResponse struct {
	Models []Model `json:"models"`
}

// Models lists every model the server knows about, loaded or not.
func (c *Client) Models(ctx context.Context) ([]Model, error) {
	ctx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.root+"/api/v1/models", http.NoBody)
	if err != nil {
		return nil, fmt.Errorf("lmstudio: build request: %w", err)
	}
	if c.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
	}

	resp, err := c.http.Do(req)
	if err != nil {
		return nil, fmt.Errorf("lmstudio: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if !successful(resp.StatusCode) {
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, errorBodyLimit))
		return nil, fmt.Errorf("lmstudio: HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(raw)))
	}

	var out modelsResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, fmt.Errorf("lmstudio: decode models: %w", err)
	}
	return out.Models, nil
}

// Model looks up one model by key. It returns ErrModelNotFound when the server
// does not list it.
func (c *Client) Model(ctx context.Context, key string) (Model, error) {
	models, err := c.Models(ctx)
	if err != nil {
		return Model{}, err
	}
	for _, m := range models {
		if m.Key == key {
			return m, nil
		}
	}
	return Model{}, fmt.Errorf("%w: %q", ErrModelNotFound, key)
}

// LLMs returns only the chat models, dropping embedding models and anything
// else the server exposes.
func LLMs(models []Model) []Model {
	out := make([]Model, 0, len(models))
	for _, m := range models {
		if m.Type == TypeLLM {
			out = append(out, m)
		}
	}
	return out
}
