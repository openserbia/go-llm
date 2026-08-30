package llm_test

import (
	"context"
	"encoding/json/v2"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/openserbia/go-llm"
)

func sseServer(t *testing.T, events ...string) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, e := range events {
			_, _ = w.Write([]byte(e + "\n\n"))
		}
	}))
	t.Cleanup(srv.Close)
	return srv
}

func delta(content string) string {
	return `data: {"choices":[{"delta":{"content":"` + content + `"}}]}`
}

func TestChatStreamCollectsDeltas(t *testing.T) {
	srv := sseServer(t, delta("Hello"), delta(", "), delta("world"), "data: [DONE]")

	ch, err := newClient(t, srv.URL).ChatStream(context.Background(), llm.ChatRequest{
		Messages: []llm.Message{llm.User("hi")},
	})
	if err != nil {
		t.Fatalf("ChatStream: %v", err)
	}
	got, err := llm.Collect(ch)
	if err != nil {
		t.Fatalf("Collect: %v", err)
	}
	if got != "Hello, world" {
		t.Errorf("Collect = %q, want %q", got, "Hello, world")
	}
}

func TestChatStreamStopsOnFinishReason(t *testing.T) {
	// The reader no longer stops at finish_reason — with
	// stream_options.include_usage the usage event follows it — so the fixture
	// sends [DONE] after the finish_reason event to terminate the stream.
	// The trailing delta proves nothing after the terminator is delivered.
	srv := sseServer(t,
		delta("one"),
		`data: {"choices":[{"delta":{"content":"two"},"finish_reason":"stop"}]}`,
		"data: [DONE]",
		delta("never delivered"),
	)

	ch, err := newClient(t, srv.URL).ChatStream(context.Background(), llm.ChatRequest{
		Messages: []llm.Message{llm.User("hi")},
	})
	if err != nil {
		t.Fatalf("ChatStream: %v", err)
	}
	got, err := llm.Collect(ch)
	if err != nil {
		t.Fatalf("Collect: %v", err)
	}
	if got != "onetwo" {
		t.Errorf("Collect = %q, want %q", got, "onetwo")
	}
}

func TestChatStreamIgnoresKeepAliveAndComments(t *testing.T) {
	srv := sseServer(t, ": keep-alive", "", delta("text"), "data: [DONE]")

	ch, err := newClient(t, srv.URL).ChatStream(context.Background(), llm.ChatRequest{
		Messages: []llm.Message{llm.User("hi")},
	})
	if err != nil {
		t.Fatalf("ChatStream: %v", err)
	}
	got, err := llm.Collect(ch)
	if err != nil {
		t.Fatalf("Collect: %v", err)
	}
	if got != "text" {
		t.Errorf("Collect = %q, want %q", got, "text")
	}
}

func TestChatStreamFallsBackWhenStreamingRejected(t *testing.T) {
	var sawStream []bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		streaming := strings.Contains(string(body), `"stream":true`)
		sawStream = append(sawStream, streaming)
		if streaming {
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte("streaming is not supported"))
			return
		}
		okResponse(w, "whole answer at once")
	}))
	t.Cleanup(srv.Close)

	ch, err := newClient(t, srv.URL).ChatStream(context.Background(), llm.ChatRequest{
		Messages: []llm.Message{llm.User("hi")},
	})
	if err != nil {
		t.Fatalf("ChatStream: %v", err)
	}
	got, err := llm.Collect(ch)
	if err != nil {
		t.Fatalf("Collect: %v", err)
	}
	if got != "whole answer at once" {
		t.Errorf("Collect = %q", got)
	}
	if len(sawStream) != 2 || !sawStream[0] || sawStream[1] {
		t.Errorf("request stream flags = %v, want [true false]", sawStream)
	}
}

func TestChatStreamSurfacesServerError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte("backend died"))
	}))
	t.Cleanup(srv.Close)

	if _, err := newClient(t, srv.URL).ChatStream(context.Background(), llm.ChatRequest{
		Messages: []llm.Message{llm.User("hi")},
	}); err == nil {
		t.Fatal("ChatStream succeeded, want error")
	}
}

func TestChatStreamMalformedEventIsReported(t *testing.T) {
	srv := sseServer(t, delta("partial"), "data: {not json")

	ch, err := newClient(t, srv.URL).ChatStream(context.Background(), llm.ChatRequest{
		Messages: []llm.Message{llm.User("hi")},
	})
	if err != nil {
		t.Fatalf("ChatStream: %v", err)
	}
	if _, err := llm.Collect(ch); err == nil {
		t.Fatal("Collect succeeded on malformed event, want error")
	}
}

func TestChatStreamTerminalChunkCarriesMetadata(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		// The usage-only event arrives after the finish_reason event when
		// stream_options.include_usage is set, so the reader must not stop at
		// the first finish_reason.
		for _, event := range []string{
			`{"choices":[{"delta":{"content":"hi"}}]}`,
			`{"choices":[{"delta":{},"finish_reason":"length"}]}`,
			`{"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":2,"total_tokens":9}}`,
			`[DONE]`,
		} {
			_, _ = w.Write([]byte("data: " + event + "\n\n"))
			w.(http.Flusher).Flush()
		}
	}))
	t.Cleanup(srv.Close)

	ch, err := newClient(t, srv.URL).ChatStream(context.Background(), llm.ChatRequest{
		Messages: []llm.Message{llm.User("hi")},
	})
	if err != nil {
		t.Fatalf("ChatStream: %v", err)
	}

	var last llm.StreamChunk
	var text strings.Builder
	for chunk := range ch {
		if chunk.Err != nil {
			t.Fatalf("stream error: %v", chunk.Err)
		}
		text.WriteString(chunk.Delta)
		last = chunk
	}

	if text.String() != "hi" {
		t.Errorf("text = %q, want %q", text.String(), "hi")
	}
	if !last.Done {
		t.Error("last chunk Done = false, want true")
	}
	if last.FinishReason != "length" {
		t.Errorf("FinishReason = %q, want %q", last.FinishReason, "length")
	}
	if last.Usage.TotalTokens != 9 {
		t.Errorf("Usage.TotalTokens = %d, want 9", last.Usage.TotalTokens)
	}
}

func TestChatStreamRequestsUsage(t *testing.T) {
	var body []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	t.Cleanup(srv.Close)

	ch, err := newClient(t, srv.URL).ChatStream(context.Background(), llm.ChatRequest{
		Messages: []llm.Message{llm.User("hi")},
	})
	if err != nil {
		t.Fatalf("ChatStream: %v", err)
	}
	if _, err := llm.Collect(ch); err != nil {
		t.Fatalf("Collect: %v", err)
	}

	var sent map[string]any
	if err := json.Unmarshal(body, &sent); err != nil {
		t.Fatalf("decode sent body: %v", err)
	}
	opts, ok := sent["stream_options"].(map[string]any)
	if !ok {
		t.Fatalf("stream_options missing from %s", body)
	}
	if opts["include_usage"] != true {
		t.Errorf("include_usage = %v, want true", opts["include_usage"])
	}
}
