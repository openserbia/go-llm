package llm

import (
	"bufio"
	"context"
	"encoding/json/v2"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// StreamChunk is one piece of a streamed completion. Done marks the end of the
// stream; Err is non-nil only when the stream terminated abnormally. Exactly
// one terminal chunk (Done or Err) is delivered before the channel closes.
//
// FinishReason and Usage are populated only on the terminal chunk, and only
// when the backend reported them. A FinishReason of "length" means the token
// budget ended the completion, which matters most when the caller asked for
// structured output: a truncated JSON document never parses.
type StreamChunk struct {
	Delta        string
	Done         bool
	Err          error
	FinishReason string
	Usage        Usage
}

// Streamer extends Client with token streaming.
type Streamer interface {
	Client
	ChatStream(ctx context.Context, req ChatRequest) (<-chan StreamChunk, error)
}

// streamBuffer is the channel depth. Deep enough that a caller doing modest
// per-token work does not stall the reader on every token.
const streamBuffer = 16

// Maximum SSE line size. Tokens are tiny, but a backend that inlines a whole
// tool-call payload into one event is not unusual.
const (
	streamLineInitial = 64 * 1024
	streamLineMax     = 1024 * 1024
)

// singleShotBuffer holds the one delta plus the terminator produced when a
// stream is emulated from a non-streaming response.
const singleShotBuffer = 2

// isStreamRejection reports whether a failed streaming request is worth
// retrying as a single non-streaming call.
//
// Deliberately broader than IsUnsupportedResponseFormat, which it used to
// share an implementation with. The single-shot path runs Chat's own degrade
// chain, so it recovers from a rejected response_format as well as from a
// backend with no SSE support at all, and guessing wrong costs one request
// that fails the same way it just did. A 4xx here is worth that one retry; a
// 5xx is the backend being broken, and repeating it would only multiply the
// failure.
func isStreamRejection(err error) bool {
	switch Status(err) {
	case http.StatusBadRequest, http.StatusUnprocessableEntity:
		return true
	default:
		return false
	}
}

// ChatStream opens a server-sent events stream and delivers token deltas on
// the returned channel, which closes when the stream ends.
//
// Unlike Chat, this applies no timeout of its own: bound the stream with ctx.
//
// A backend that rejects streaming falls back to a single non-streaming
// request whose whole response arrives as one delta. Callers therefore do not
// need to know whether the backend supports SSE.
func (c *HTTPClient) ChatStream(ctx context.Context, req ChatRequest) (<-chan StreamChunk, error) {
	if len(req.Messages) == 0 {
		return nil, errors.New("llm: chat-stream: no messages")
	}

	httpReq, err := c.buildRequest(ctx, c.wire(req, req.ResponseFormat, true))
	if err != nil {
		return nil, err
	}

	resp, err := c.http.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("llm: chat-stream: %w", err)
	}

	if !successful(resp.StatusCode) {
		body := readErrorBody(resp.Body)
		_ = resp.Body.Close()
		apiErr := &APIError{Op: "llm: chat-stream", Status: resp.StatusCode, Body: body}
		if isStreamRejection(apiErr) {
			return c.streamViaSingleShot(ctx, req)
		}
		return nil, apiErr
	}

	ch := make(chan StreamChunk, streamBuffer)
	go consume(resp.Body, ch)
	return ch, nil
}

type streamEvent struct {
	Choices []struct {
		Delta struct {
			Content string `json:"content"`
		} `json:"delta"`
		FinishReason *string `json:"finish_reason"`
	} `json:"choices"`
	Usage *Usage `json:"usage"`
}

func consume(body io.ReadCloser, ch chan<- StreamChunk) {
	defer close(ch)
	defer func() { _ = body.Close() }()

	var terminal StreamChunk
	terminal.Done = true

	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 0, streamLineInitial), streamLineMax)
	for scanner.Scan() {
		line := strings.TrimRight(scanner.Text(), "\r")
		if line == "" || !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "[DONE]" {
			ch <- terminal
			return
		}

		var event streamEvent
		if err := json.Unmarshal([]byte(data), &event); err != nil {
			ch <- StreamChunk{Err: fmt.Errorf("llm: chat-stream: parse event: %w", err)}
			return
		}
		if event.Usage != nil {
			terminal.Usage = *event.Usage
		}
		if len(event.Choices) == 0 {
			continue
		}
		if delta := event.Choices[0].Delta.Content; delta != "" {
			ch <- StreamChunk{Delta: delta}
		}
		// Record the reason but keep reading: with stream_options.include_usage
		// the usage-only event follows this one, and stopping here would drop it.
		if reason := event.Choices[0].FinishReason; reason != nil {
			terminal.FinishReason = *reason
		}
	}

	if err := scanner.Err(); err != nil && !errors.Is(err, io.EOF) {
		ch <- StreamChunk{Err: fmt.Errorf("llm: chat-stream: read: %w", err)}
		return
	}
	// Clean EOF without an explicit terminator still ends the stream.
	ch <- terminal
}

func (c *HTTPClient) streamViaSingleShot(ctx context.Context, req ChatRequest) (<-chan StreamChunk, error) {
	resp, err := c.Chat(ctx, req)
	if err != nil {
		return nil, err
	}
	ch := make(chan StreamChunk, singleShotBuffer)
	if resp.Content != "" {
		ch <- StreamChunk{Delta: resp.Content}
	}
	ch <- StreamChunk{Done: true, FinishReason: resp.FinishReason, Usage: resp.Usage}
	close(ch)
	return ch, nil
}

// Collect drains a stream into the full text. It returns the first error the
// stream reports, discarding any text received before it.
func Collect(ch <-chan StreamChunk) (string, error) {
	var b strings.Builder
	for chunk := range ch {
		if chunk.Err != nil {
			return "", chunk.Err
		}
		b.WriteString(chunk.Delta)
	}
	return b.String(), nil
}
