package llm_test

import (
	"errors"
	"testing"

	"github.com/openserbia/go-llm"
)

func TestIsContextOverflow(t *testing.T) {
	tests := []struct {
		name string
		err  error
		want bool
	}{
		{"nil", nil, false},
		{"lm studio phrasing", errors.New("Context size has been exceeded"), true},
		{"llama.cpp phrasing", errors.New("n_keep: 5000 >= n_ctx: 4096"), true},
		{"openai code", errors.New(`{"code":"context_length_exceeded"}`), true},
		{"openai prose", errors.New("This model's maximum context length is 8192 tokens"), true},
		{"anthropic bridge", errors.New("prompt is too long: 250000 tokens"), true},
		{"unrelated", errors.New("connection refused"), false},
		{"unrelated 500", errors.New("llm: chat: HTTP 500: internal error"), false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := llm.IsContextOverflow(tt.err); got != tt.want {
				t.Errorf("IsContextOverflow(%v) = %v, want %v", tt.err, got, tt.want)
			}
		})
	}
}

func TestIsContextOverflowThroughWrappedError(t *testing.T) {
	wrapped := &llm.APIError{Op: "llm: chat", Status: 400, Body: "Context size has been exceeded"}
	if !llm.IsContextOverflow(wrapped) {
		t.Error("IsContextOverflow = false on an APIError carrying the overflow text, want true")
	}
}

func TestStatusOnNonAPIError(t *testing.T) {
	if got := llm.Status(errors.New("plain")); got != 0 {
		t.Errorf("Status = %d, want 0", got)
	}
}
