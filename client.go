package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
)

// Role is a chat message author.
type Role string

// The chat roles this library sends.
const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
)

// Message is one turn in a conversation.
type Message struct {
	Role    Role   `json:"role"`
	Content string `json:"content"`
}

// System builds a system-role Message.
func System(content string) Message { return Message{Role: RoleSystem, Content: content} }

// User builds a user-role Message.
func User(content string) Message { return Message{Role: RoleUser, Content: content} }

// Assistant builds an assistant-role Message, for replaying prior turns.
func Assistant(content string) Message { return Message{Role: RoleAssistant, Content: content} }

// ChatRequest is a chat completion request.
type ChatRequest struct {
	// Messages is the conversation so far. Required.
	Messages []Message

	// Model overrides Options.Model for this request.
	Model string

	// Temperature is passed through as-is. Zero means deterministic-ish
	// decoding, which is what most extraction and classification work wants.
	Temperature float64

	// MaxTokens caps the completion length. Zero omits the field and lets the
	// backend apply its own default.
	MaxTokens int

	// ResponseFormat requests structured output. Nil asks for free text.
	ResponseFormat *ResponseFormat

	// OnOverflow is read only by the Retry middleware, which consults it when
	// the backend reports a context overflow. A bare Client ignores it.
	OnOverflow OnOverflowFunc
}

// ChatResponse is a completed chat response.
type ChatResponse struct {
	Content      string
	FinishReason string
	Usage        Usage
}

// Usage reports token accounting when the backend supplies it. Not every
// backend does; zero values mean "not reported", not "zero tokens".
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// Client is the chat surface. HTTPClient implements it directly; middleware in
// this package wraps it; tests can substitute a stub.
type Client interface {
	Chat(ctx context.Context, req ChatRequest) (ChatResponse, error)
}

// HTTPClient talks to an OpenAI-compatible endpoint over HTTP.
type HTTPClient struct {
	opts Options
	http *http.Client
}

// Compile-time proof that the concrete client satisfies both interfaces.
var (
	_ Client   = (*HTTPClient)(nil)
	_ Streamer = (*HTTPClient)(nil)
)

// New builds a client. It fails only on options that cannot produce a valid
// request; it does not contact the backend. Use
// [github.com/openserbia/go-llm/lmstudio.EnsureLoaded] if you want to verify
// the model is actually there before your first real request.
func New(opts Options) (*HTTPClient, error) {
	if opts.BaseURL == "" {
		return nil, errors.New("llm: BaseURL is required")
	}
	return &HTTPClient{opts: opts, http: opts.httpClient()}, nil
}

type wireRequest struct {
	Model          string          `json:"model"`
	Messages       []Message       `json:"messages"`
	Temperature    float64         `json:"temperature"`
	MaxTokens      int             `json:"max_tokens,omitempty"`
	Stream         bool            `json:"stream,omitempty"`
	ResponseFormat *ResponseFormat `json:"response_format,omitempty"`
}

type wireResponse struct {
	Choices []struct {
		Message      Message `json:"message"`
		FinishReason string  `json:"finish_reason"`
	} `json:"choices"`
	Usage Usage `json:"usage"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error"`
}

func (c *HTTPClient) wire(req ChatRequest, format *ResponseFormat, stream bool) wireRequest {
	model := req.Model
	if model == "" {
		model = c.opts.Model
	}
	return wireRequest{
		Model:          model,
		Messages:       req.Messages,
		Temperature:    req.Temperature,
		MaxTokens:      req.MaxTokens,
		Stream:         stream,
		ResponseFormat: format,
	}
}

// Chat sends a chat completion request.
//
// When ResponseFormat asks for something the backend does not implement, Chat
// retries at successively weaker levels — a strict schema falls back to plain
// JSON object mode, which falls back to unconstrained text. Callers that must
// have real schema enforcement should set ResponseFormat.Required.
func (c *HTTPClient) Chat(ctx context.Context, req ChatRequest) (ChatResponse, error) {
	if len(req.Messages) == 0 {
		return ChatResponse{}, errors.New("llm: chat: no messages")
	}

	ctx, cancel := context.WithTimeout(ctx, c.opts.timeout())
	defer cancel()

	var lastErr error
	for _, format := range req.ResponseFormat.degradeChain() {
		resp, err := c.do(ctx, c.wire(req, format, false))
		if err == nil {
			return resp, nil
		}
		lastErr = err
		if !IsUnsupportedResponseFormat(err) {
			return ChatResponse{}, err
		}
	}
	return ChatResponse{}, lastErr
}

// Complete is a convenience wrapper for the common single system + single user
// prompt shape.
func (c *HTTPClient) Complete(ctx context.Context, system, user string, temperature float64, format *ResponseFormat) (string, error) {
	resp, err := c.Chat(ctx, ChatRequest{
		Messages:       []Message{System(system), User(user)},
		Temperature:    temperature,
		ResponseFormat: format,
	})
	if err != nil {
		return "", err
	}
	return resp.Content, nil
}

func (c *HTTPClient) do(ctx context.Context, body wireRequest) (ChatResponse, error) {
	req, err := c.buildRequest(ctx, body)
	if err != nil {
		return ChatResponse{}, err
	}

	httpResp, err := c.http.Do(req)
	if err != nil {
		return ChatResponse{}, fmt.Errorf("llm: chat: %w", err)
	}
	defer func() { _ = httpResp.Body.Close() }()

	if !successful(httpResp.StatusCode) {
		return ChatResponse{}, &APIError{Op: "llm: chat", Status: httpResp.StatusCode, Body: readErrorBody(httpResp.Body)}
	}

	var out wireResponse
	if err := json.NewDecoder(httpResp.Body).Decode(&out); err != nil {
		return ChatResponse{}, fmt.Errorf("llm: chat: decode response: %w", err)
	}
	// Some backends answer 200 with an error envelope instead of a status code.
	if out.Error != nil {
		return ChatResponse{}, fmt.Errorf("llm: chat: %s", out.Error.Message)
	}
	if len(out.Choices) == 0 {
		return ChatResponse{}, errors.New("llm: chat: no choices in response")
	}
	return ChatResponse{
		Content:      out.Choices[0].Message.Content,
		FinishReason: out.Choices[0].FinishReason,
		Usage:        out.Usage,
	}, nil
}

func (c *HTTPClient) buildRequest(ctx context.Context, body wireRequest) (*http.Request, error) {
	buf, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("llm: encode request: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.opts.endpoint("/chat/completions"), bytes.NewReader(buf))
	if err != nil {
		return nil, fmt.Errorf("llm: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if c.opts.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.opts.APIKey)
	}
	if body.Stream {
		req.Header.Set("Accept", "text/event-stream")
	}
	return req, nil
}

// errorBodyLimit caps how much of an error response we read. Enough for any
// real diagnostic, bounded against a backend that answers an error with a
// megabyte of HTML.
const errorBodyLimit = 4096

// successful reports whether status is 2xx.
func successful(status int) bool {
	return status >= http.StatusOK && status < http.StatusMultipleChoices
}

func readErrorBody(r io.Reader) string {
	raw, _ := io.ReadAll(io.LimitReader(r, errorBodyLimit))
	return string(raw)
}
