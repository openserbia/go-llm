package llm

import (
	"context"
	"encoding/json/v2"
	"errors"
	"fmt"
	"net/http"
	"strings"
)

// ModelInfo is one entry from the OpenAI-compatible GET /models.
//
// This endpoint reports identifiers and little else. When the backend is LM
// Studio, [github.com/openserbia/go-llm/lmstudio] reads a native endpoint that
// also reports capabilities and the context length a model is loaded with.
type ModelInfo struct {
	ID      string `json:"id"`
	OwnedBy string `json:"owned_by,omitempty"`
}

type modelsResponse struct {
	Data []ModelInfo `json:"data"`
}

// Models lists the models the backend exposes.
func (c *HTTPClient) Models(ctx context.Context) ([]ModelInfo, error) {
	ctx, cancel := context.WithTimeout(ctx, c.opts.timeout())
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.opts.endpoint("/models"), http.NoBody)
	if err != nil {
		return nil, fmt.Errorf("llm: build models request: %w", err)
	}
	if c.opts.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.opts.APIKey)
	}

	resp, err := c.http.Do(req)
	if err != nil {
		return nil, fmt.Errorf("llm: models: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if !successful(resp.StatusCode) {
		return nil, &APIError{Op: "llm: models", Status: resp.StatusCode, Body: readErrorBody(resp.Body)}
	}

	var out modelsResponse
	if err := json.UnmarshalRead(resp.Body, &out); err != nil {
		return nil, fmt.Errorf("llm: models: decode response: %w", err)
	}
	return out.Data, nil
}

// EnsureModel verifies that the backend serves the given model, or
// Options.Model when id is empty. The error names what is available instead,
// which is the difference between a usable failure at startup and a confusing
// one on the first request.
//
// A model being listed does not mean it is loaded — LM Studio lists models it
// would load on demand. Use lmstudio.EnsureLoaded when you need the stronger
// guarantee, and its context length.
func (c *HTTPClient) EnsureModel(ctx context.Context, id string) error {
	if id == "" {
		id = c.opts.Model
	}
	if id == "" {
		return errors.New("llm: EnsureModel: no model given and Options.Model is empty")
	}

	models, err := c.Models(ctx)
	if err != nil {
		return err
	}

	available := make([]string, 0, len(models))
	for _, m := range models {
		if m.ID == id {
			return nil
		}
		available = append(available, m.ID)
	}
	return fmt.Errorf("llm: model %q is not available at %s (available: %s)",
		id, c.opts.BaseURL, strings.Join(available, ", "))
}
