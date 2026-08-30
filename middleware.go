package llm

import (
	"context"
	"errors"
)

// OnOverflowFunc produces a smaller version of prev after the backend reported
// a context overflow. attempt starts at 1 for the first retry. Returning
// ok=false means "cannot shrink further": the middleware stops and returns the
// backend's error.
type OnOverflowFunc func(prev ChatRequest, attempt int) (next ChatRequest, ok bool)

// DefaultOverflowAttempts is the retry budget used when NewOverflowRetry is
// given a non-positive value.
const DefaultOverflowAttempts = 4

// OverflowRetry retries context-overflow failures with a smaller request.
//
// The shrink policy is supplied per request rather than baked in, because the
// two halves of the problem live in different places: detecting "the prompt
// did not fit" is generic across backends, but making a prompt smaller is
// specific to what the prompt is made of. Dropping retrieved excerpts, cutting
// low-similarity candidates and summarising history are all valid answers, and
// only the caller knows which applies.
//
// Requests without an OnOverflow hook pass through untouched, so wrapping a
// client is safe for callers that have not opted in.
type OverflowRetry struct {
	Next        Client
	MaxAttempts int
}

var (
	_ Client   = (*OverflowRetry)(nil)
	_ Streamer = (*OverflowRetry)(nil)
)

// NewOverflowRetry wraps next. A non-positive maxAttempts means
// DefaultOverflowAttempts.
func NewOverflowRetry(next Client, maxAttempts int) *OverflowRetry {
	if maxAttempts <= 0 {
		maxAttempts = DefaultOverflowAttempts
	}
	return &OverflowRetry{Next: next, MaxAttempts: maxAttempts}
}

// Chat forwards to the wrapped client, retrying with a smaller request while
// the backend reports a context overflow and the request's OnOverflow hook is
// willing to shrink further.
func (m *OverflowRetry) Chat(ctx context.Context, req ChatRequest) (ChatResponse, error) {
	resp, err := m.Next.Chat(ctx, req)
	if err == nil || req.OnOverflow == nil || !IsContextOverflow(err) {
		return resp, err
	}

	cur := req
	for attempt := 1; attempt <= m.MaxAttempts; attempt++ {
		next, ok := cur.OnOverflow(cur, attempt)
		if !ok {
			return ChatResponse{}, err
		}
		cur = next

		resp, err = m.Next.Chat(ctx, cur)
		if err == nil {
			return resp, nil
		}
		if !IsContextOverflow(err) {
			return ChatResponse{}, err
		}
	}
	return ChatResponse{}, err
}

// ChatStream passes through to the wrapped client without overflow handling.
// Restarting a stream that has already emitted tokens would mean discarding
// visible output and re-rendering it, which reads worse than a clean failure;
// callers who need overflow protection while streaming should size the prompt
// before opening the stream.
func (m *OverflowRetry) ChatStream(ctx context.Context, req ChatRequest) (<-chan StreamChunk, error) {
	streamer, ok := m.Next.(Streamer)
	if !ok {
		return nil, errors.New("llm: OverflowRetry: wrapped client does not stream")
	}
	return streamer.ChatStream(ctx, req)
}
