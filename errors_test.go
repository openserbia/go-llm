package llm_test

import (
	"errors"
	"testing"

	"github.com/openserbia/go-llm"
)

func TestIsUnsupportedResponseFormat(t *testing.T) {
	tests := []struct {
		name string
		err  error
		want bool
	}{
		{
			"named rejection",
			&llm.APIError{Op: "llm: chat", Status: 400, Body: `{"error":"response_format is not supported"}`},
			true,
		},
		{
			"grammar engine missing",
			&llm.APIError{Op: "llm: chat", Status: 400, Body: "no grammar engine here"},
			true,
		},
		{
			"vllm guided decoding",
			&llm.APIError{Op: "llm: chat", Status: 400, Body: "guided_json is not enabled on this server"},
			true,
		},
		{
			"unprocessable naming the schema",
			&llm.APIError{Op: "llm: chat", Status: 422, Body: "json_schema: unknown keyword minItems"},
			true,
		},
		{
			// A 400 that says nothing about the format is not evidence about
			// the format. Degrading here would hide the real problem and burn
			// two more requests.
			"unrelated bad request",
			&llm.APIError{Op: "llm: chat", Status: 400, Body: `{"error":"model 'gemma-3' not found"}`},
			false,
		},
		{
			// The fix is one word in the prompt, not a weaker format.
			"json mode needs the word json",
			&llm.APIError{Op: "llm: chat", Status: 400, Body: "'messages' must contain the word 'json' in some form"},
			false,
		},
		{
			"overflow reported as 400",
			&llm.APIError{Op: "llm: chat", Status: 400, Body: "Context size has been exceeded"},
			false,
		},
		{"server error", &llm.APIError{Op: "llm: chat", Status: 500, Body: "response_format"}, false},
		{"not an APIError", errors.New("response_format is not supported"), false},
		{"nil", nil, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := llm.IsUnsupportedResponseFormat(tt.err); got != tt.want {
				t.Errorf("IsUnsupportedResponseFormat(%v) = %v, want %v", tt.err, got, tt.want)
			}
		})
	}
}

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
