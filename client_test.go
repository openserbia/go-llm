package llm_test

import (
	"context"
	"encoding/json/v2"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/openserbia/go-llm"
)

// capture records what every request carried, so a test can assert on the
// fallback sequence and on the exact wire body rather than just the answer.
type capture struct {
	formats []*llm.ResponseFormat
	bodies  [][]byte
	auth    string
}

func newServer(t *testing.T, rec *capture, handle func(attempt int, w http.ResponseWriter)) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read request body: %v", err)
		}
		rec.bodies = append(rec.bodies, raw)

		var body struct {
			ResponseFormat *llm.ResponseFormat `json:"response_format"`
		}
		if err := json.Unmarshal(raw, &body); err != nil {
			t.Errorf("decode request: %v", err)
		}
		rec.formats = append(rec.formats, body.ResponseFormat)
		rec.auth = r.Header.Get("Authorization")
		handle(len(rec.formats), w)
	}))
	t.Cleanup(srv.Close)
	return srv
}

func okResponse(w http.ResponseWriter, content string) {
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"` + content + `"},"finish_reason":"stop"}],` +
		`"usage":{"prompt_tokens":11,"completion_tokens":3,"total_tokens":14}}`))
}

func newClient(t *testing.T, baseURL string) *llm.HTTPClient {
	t.Helper()
	c, err := llm.New(llm.Options{BaseURL: baseURL, Model: "test-model", APIKey: "secret"})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return c
}

func TestChatReturnsContentAndUsage(t *testing.T) {
	rec := &capture{}
	srv := newServer(t, rec, func(_ int, w http.ResponseWriter) { okResponse(w, "hello") })

	resp, err := newClient(t, srv.URL).Chat(context.Background(), llm.ChatRequest{
		Messages: []llm.Message{llm.User("hi")},
	})
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if resp.Content != "hello" {
		t.Errorf("Content = %q, want %q", resp.Content, "hello")
	}
	if resp.FinishReason != "stop" {
		t.Errorf("FinishReason = %q, want %q", resp.FinishReason, "stop")
	}
	if resp.Usage.TotalTokens != 14 {
		t.Errorf("Usage.TotalTokens = %d, want 14", resp.Usage.TotalTokens)
	}
	if rec.auth != "Bearer secret" {
		t.Errorf("Authorization = %q, want %q", rec.auth, "Bearer secret")
	}
}

func TestChatOmitsMaxTokensWhenZero(t *testing.T) {
	rec := &capture{}
	srv := newServer(t, rec, func(_ int, w http.ResponseWriter) { okResponse(w, "hi") })

	if _, err := newClient(t, srv.URL).Chat(context.Background(), llm.ChatRequest{
		Messages: []llm.Message{llm.User("hi")},
	}); err != nil {
		t.Fatalf("Chat: %v", err)
	}

	var sent map[string]any
	if err := json.Unmarshal(rec.bodies[0], &sent); err != nil {
		t.Fatalf("decode sent body: %v", err)
	}
	// A zero MaxTokens means "let the backend decide", not "produce nothing".
	if _, ok := sent["max_tokens"]; ok {
		t.Errorf("request sent max_tokens = %v, want the key absent", sent["max_tokens"])
	}
	if _, ok := sent["stream"]; ok {
		t.Errorf("request sent stream = %v, want the key absent", sent["stream"])
	}
}

func TestChatDegradesSchemaToObjectToNone(t *testing.T) {
	rec := &capture{}
	// A backend with no structured-output support at all: every response_format
	// is rejected, and only the bare request succeeds.
	srv := newServer(t, rec, func(attempt int, w http.ResponseWriter) {
		if attempt < 3 {
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte(`{"error":"response_format is not supported"}`))
			return
		}
		okResponse(w, "plain text")
	})

	resp, err := newClient(t, srv.URL).Chat(context.Background(), llm.ChatRequest{
		Messages:       []llm.Message{llm.User("hi")},
		ResponseFormat: llm.JSONSchemaOf("answer", map[string]any{"type": "object"}),
	})
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if resp.Content != "plain text" {
		t.Errorf("Content = %q", resp.Content)
	}

	if len(rec.formats) != 3 {
		t.Fatalf("attempts = %d, want 3", len(rec.formats))
	}
	if got := rec.formats[0]; got == nil || got.Type != llm.FormatJSONSchema {
		t.Errorf("attempt 1 format = %+v, want json_schema", got)
	}
	if got := rec.formats[1]; got == nil || got.Type != llm.FormatJSONObject {
		t.Errorf("attempt 2 format = %+v, want json_object", got)
	}
	if rec.formats[2] != nil {
		t.Errorf("attempt 3 format = %+v, want none", rec.formats[2])
	}
}

func TestChatRequiredFormatDoesNotDegrade(t *testing.T) {
	rec := &capture{}
	srv := newServer(t, rec, func(_ int, w http.ResponseWriter) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte("no grammar engine here"))
	})

	format := llm.JSONSchemaOf("answer", map[string]any{"type": "object"})
	format.Required = true

	_, err := newClient(t, srv.URL).Chat(context.Background(), llm.ChatRequest{
		Messages:       []llm.Message{llm.User("hi")},
		ResponseFormat: format,
	})
	if err == nil {
		t.Fatal("Chat succeeded, want error")
	}
	if len(rec.formats) != 1 {
		t.Errorf("attempts = %d, want 1 — Required must not fall back", len(rec.formats))
	}
	if llm.Status(err) != http.StatusBadRequest {
		t.Errorf("Status(err) = %d, want 400", llm.Status(err))
	}
}

func TestChatServerErrorIsNotRetried(t *testing.T) {
	rec := &capture{}
	srv := newServer(t, rec, func(_ int, w http.ResponseWriter) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte("boom"))
	})

	_, err := newClient(t, srv.URL).Chat(context.Background(), llm.ChatRequest{
		Messages:       []llm.Message{llm.User("hi")},
		ResponseFormat: llm.JSONObject(),
	})
	if err == nil {
		t.Fatal("Chat succeeded, want error")
	}
	// 500 is the backend failing, not the format being unsupported: retrying a
	// weaker format would just multiply the failure.
	if len(rec.formats) != 1 {
		t.Errorf("attempts = %d, want 1", len(rec.formats))
	}

	var apiErr *llm.APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("err = %v, want *llm.APIError", err)
	}
	if apiErr.Status != http.StatusInternalServerError {
		t.Errorf("Status = %d, want 500", apiErr.Status)
	}
}

func TestChatErrorEnvelopeOn200(t *testing.T) {
	rec := &capture{}
	srv := newServer(t, rec, func(_ int, w http.ResponseWriter) {
		_, _ = w.Write([]byte(`{"error":{"message":"model not loaded"}}`))
	})

	_, err := newClient(t, srv.URL).Chat(context.Background(), llm.ChatRequest{
		Messages: []llm.Message{llm.User("hi")},
	})
	if err == nil {
		t.Fatal("Chat succeeded, want error")
	}
	if got := err.Error(); got != "llm: chat: model not loaded" {
		t.Errorf("err = %q", got)
	}
}

func TestChatRejectsEmptyMessages(t *testing.T) {
	c := newClient(t, "http://127.0.0.1:1")
	if _, err := c.Chat(context.Background(), llm.ChatRequest{}); err == nil {
		t.Fatal("Chat succeeded on empty messages, want error")
	}
}

func TestChatTimeoutIsApplied(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, r *http.Request) {
		// Bounded, so the server shuts down even if the client fails to cancel.
		select {
		case <-r.Context().Done():
		case <-time.After(300 * time.Millisecond):
		}
	}))
	t.Cleanup(srv.Close)

	c, err := llm.New(llm.Options{BaseURL: srv.URL, Model: "m", Timeout: 50 * time.Millisecond})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	start := time.Now()
	if _, err := c.Chat(context.Background(), llm.ChatRequest{Messages: []llm.Message{llm.User("hi")}}); err == nil {
		t.Fatal("Chat succeeded, want timeout")
	}
	if elapsed := time.Since(start); elapsed > 2*time.Second {
		t.Errorf("Chat took %v, want the 50ms timeout to apply", elapsed)
	}
}

func TestNewRequiresBaseURL(t *testing.T) {
	if _, err := llm.New(llm.Options{Model: "m"}); err == nil {
		t.Fatal("New succeeded without BaseURL, want error")
	}
}

func TestCompleteSendsSystemAndUser(t *testing.T) {
	var roles []llm.Role
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Messages []llm.Message `json:"messages"`
		}
		_ = json.UnmarshalRead(r.Body, &body)
		for _, m := range body.Messages {
			roles = append(roles, m.Role)
		}
		okResponse(w, "done")
	}))
	t.Cleanup(srv.Close)

	got, err := newClient(t, srv.URL).Complete(context.Background(), "be terse", "why?", 0.1, nil)
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if got != "done" {
		t.Errorf("Complete = %q, want %q", got, "done")
	}
	want := []llm.Role{llm.RoleSystem, llm.RoleUser}
	if len(roles) != len(want) || roles[0] != want[0] || roles[1] != want[1] {
		t.Errorf("roles = %v, want %v", roles, want)
	}
}
