package llm_test

import (
	"context"
	"errors"
	"testing"

	"github.com/openserbia/go-llm"
)

// stubClient answers with a scripted sequence of errors, then success.
type stubClient struct {
	failures int // how many leading calls fail with an overflow error
	calls    int
	sizes    []int // len(Messages) seen on each call
}

func (s *stubClient) Chat(_ context.Context, req llm.ChatRequest) (llm.ChatResponse, error) {
	s.calls++
	s.sizes = append(s.sizes, len(req.Messages))
	if s.calls <= s.failures {
		return llm.ChatResponse{}, errors.New("Context size has been exceeded (n_ctx: 4096)")
	}
	return llm.ChatResponse{Content: "ok"}, nil
}

// dropOldest is a shrink policy: throw away the oldest message each attempt.
func dropOldest(prev llm.ChatRequest, _ int) (llm.ChatRequest, bool) {
	if len(prev.Messages) <= 1 {
		return prev, false
	}
	next := prev
	next.Messages = prev.Messages[1:]
	return next, true
}

func messages(n int) []llm.Message {
	out := make([]llm.Message, n)
	for i := range out {
		out[i] = llm.User("message")
	}
	return out
}

func TestOverflowRetryShrinksUntilItFits(t *testing.T) {
	stub := &stubClient{failures: 2}
	client := llm.NewOverflowRetry(stub, 4)

	resp, err := client.Chat(context.Background(), llm.ChatRequest{
		Messages:   messages(5),
		OnOverflow: dropOldest,
	})
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if resp.Content != "ok" {
		t.Errorf("Content = %q, want %q", resp.Content, "ok")
	}
	want := []int{5, 4, 3}
	if len(stub.sizes) != len(want) {
		t.Fatalf("calls = %v, want %v", stub.sizes, want)
	}
	for i, n := range want {
		if stub.sizes[i] != n {
			t.Errorf("call %d had %d messages, want %d", i+1, stub.sizes[i], n)
		}
	}
}

func TestOverflowRetryWithoutHookPassesThrough(t *testing.T) {
	stub := &stubClient{failures: 1}
	client := llm.NewOverflowRetry(stub, 4)

	// No OnOverflow: the middleware must not invent a shrink policy of its own.
	if _, err := client.Chat(context.Background(), llm.ChatRequest{Messages: messages(3)}); err == nil {
		t.Fatal("Chat succeeded, want the overflow error passed through")
	}
	if stub.calls != 1 {
		t.Errorf("calls = %d, want 1", stub.calls)
	}
}

func TestOverflowRetryStopsWhenPolicyGivesUp(t *testing.T) {
	stub := &stubClient{failures: 99}
	client := llm.NewOverflowRetry(stub, 10)

	// dropOldest refuses below one message, so this bottoms out after 2 retries
	// rather than burning the full attempt budget.
	if _, err := client.Chat(context.Background(), llm.ChatRequest{
		Messages:   messages(3),
		OnOverflow: dropOldest,
	}); err == nil {
		t.Fatal("Chat succeeded, want error")
	}
	if stub.calls != 3 {
		t.Errorf("calls = %d, want 3", stub.calls)
	}
}

func TestOverflowRetryRespectsAttemptBudget(t *testing.T) {
	stub := &stubClient{failures: 99}
	client := llm.NewOverflowRetry(stub, 2)

	keepShrinking := func(prev llm.ChatRequest, _ int) (llm.ChatRequest, bool) { return prev, true }
	if _, err := client.Chat(context.Background(), llm.ChatRequest{
		Messages:   messages(3),
		OnOverflow: keepShrinking,
	}); err == nil {
		t.Fatal("Chat succeeded, want error")
	}
	if stub.calls != 3 { // 1 initial + 2 retries
		t.Errorf("calls = %d, want 3", stub.calls)
	}
}

func TestOverflowRetryIgnoresUnrelatedErrors(t *testing.T) {
	stub := &stubUnrelated{}
	client := llm.NewOverflowRetry(stub, 4)

	if _, err := client.Chat(context.Background(), llm.ChatRequest{
		Messages:   messages(3),
		OnOverflow: dropOldest,
	}); err == nil {
		t.Fatal("Chat succeeded, want error")
	}
	if stub.calls != 1 {
		t.Errorf("calls = %d, want 1 — a non-overflow error must not retry", stub.calls)
	}
}

type stubUnrelated struct{ calls int }

func (s *stubUnrelated) Chat(context.Context, llm.ChatRequest) (llm.ChatResponse, error) {
	s.calls++
	return llm.ChatResponse{}, errors.New("connection refused")
}

func TestOverflowRetryStreamRequiresStreamer(t *testing.T) {
	client := llm.NewOverflowRetry(&stubUnrelated{}, 4)
	if _, err := client.ChatStream(context.Background(), llm.ChatRequest{
		Messages: messages(1),
	}); err == nil {
		t.Fatal("ChatStream succeeded on a non-streaming client, want error")
	}
}
