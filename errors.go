package llm

import (
	"errors"
	"fmt"
	"net/http"
	"strings"
)

// APIError is a non-2xx response from the backend. Body is truncated to a few
// kilobytes: backends vary wildly in how much detail they return, and the
// useful part is always at the front.
type APIError struct {
	Op     string
	Status int
	Body   string
}

// Error implements error.
func (e *APIError) Error() string {
	return fmt.Sprintf("%s: HTTP %d: %s", e.Op, e.Status, strings.TrimSpace(e.Body))
}

// Status reports the HTTP status of err if it is an APIError, else 0.
func Status(err error) int {
	var apiErr *APIError
	if errors.As(err, &apiErr) {
		return apiErr.Status
	}
	return 0
}

// IsUnsupportedResponseFormat reports whether err looks like the backend
// rejecting the response_format we asked for, rather than rejecting the
// request as a whole. Backends without a grammar engine answer a
// json_schema request with 400 or 422 and no machine-readable marker, so
// status is the only signal available; Chat uses this to decide whether
// retrying at a weaker structured-output level is worth attempting.
func IsUnsupportedResponseFormat(err error) bool {
	switch Status(err) {
	case http.StatusBadRequest, http.StatusUnprocessableEntity:
		return true
	default:
		return false
	}
}

// IsContextOverflow reports whether err is the backend saying the prompt did
// not fit in the model's context window.
//
// There is no standard error code for this, so it is string matching against
// the wordings the common backends emit. LM Studio has several depending on
// build ("Context size has been exceeded", or the llama.cpp-shaped "n_keep: X
// >= n_ctx: Y"); OpenAI uses "context_length_exceeded" / "maximum context
// length"; Anthropic-compatible bridges use "prompt_too_long". Any one match
// is enough.
//
// Prefer sizing the prompt correctly up front where you can — the LM Studio
// native model API reports the context a model is actually loaded with, see
// [github.com/openserbia/go-llm/lmstudio] — and treat this as the backstop.
func IsContextOverflow(err error) bool {
	if err == nil {
		return false
	}
	s := err.Error()
	for _, needle := range overflowSignatures {
		if strings.Contains(s, needle) {
			return true
		}
	}
	return false
}

var overflowSignatures = []string{
	"Context size has been exceeded",
	"context_length_exceeded",
	"maximum context length",
	"greater than the context length",
	"n_ctx:",
	"prompt is too long",
	"prompt_too_long",
}
