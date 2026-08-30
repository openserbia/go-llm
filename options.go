package llm

import (
	"net/http"
	"time"

	"github.com/openserbia/go-llm/internal/httpx"
)

// DefaultTimeout bounds a single non-streaming request when Options.Timeout
// is left at zero.
const DefaultTimeout = 2 * time.Minute

// Options configures a Client.
type Options struct {
	// BaseURL is the API root including any version segment, e.g.
	// "http://localhost:1234/v1". A trailing slash is accepted. Required.
	BaseURL string

	// APIKey is sent as a Bearer token when non-empty. Local backends
	// generally ignore it; hosted ones require it.
	APIKey string

	// Model names the model to use for requests that do not set their own.
	Model string

	// Timeout bounds one non-streaming request, applied through the request
	// context. Zero means DefaultTimeout. It deliberately does not apply to
	// streaming: a stream is long-lived by construction, and a transport-level
	// deadline would sever it mid-response. Bound streams with the ctx you
	// pass to ChatStream.
	Timeout time.Duration

	// HTTPClient replaces the default transport. Leave nil unless you need
	// custom TLS, proxying, or instrumentation. A Timeout set on a supplied
	// client applies to streaming too, and will cut long streams short.
	HTTPClient *http.Client
}

func (o Options) timeout() time.Duration {
	return httpx.Timeout(o.Timeout, DefaultTimeout)
}

func (o Options) httpClient() *http.Client {
	return httpx.Client(o.HTTPClient)
}

// endpoint joins the base URL with a path segment such as "/chat/completions".
func (o Options) endpoint(path string) string {
	return httpx.Join(o.BaseURL, path)
}
