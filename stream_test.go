package llm_test

import (
	"context"
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
	// A backend that ends with finish_reason and never sends [DONE] still
	// terminates the stream.
	srv := sseServer(t,
		delta("one"),
		`data: {"choices":[{"delta":{"content":"two"},"finish_reason":"stop"}]}`,
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
