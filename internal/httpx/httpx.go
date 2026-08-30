// Package httpx holds the HTTP mechanics shared by the client packages.
//
// Every client in this module talks to an OpenAI-shaped JSON endpoint over
// HTTP with an optional bearer token, and every one of them had grown its own
// copy of the same six helpers. They live here instead so that a change to the
// error-body bound, the redirect policy or the retry rule happens once.
//
// This package is internal on purpose. The knobs it holds are decisions the
// library makes on the caller's behalf; a caller who disagrees supplies their
// own *http.Client through the Options of the package they are using.
package httpx

import (
	"errors"
	"io"
	"net"
	"net/http"
	"strings"
	"time"
)

// ErrorBodyLimit caps how much of an error response is quoted back. Enough for
// any real diagnostic, bounded against a backend that answers an error with a
// megabyte of HTML.
const ErrorBodyLimit = 4096

// DialRetries is how many extra attempts a request gets when the connection
// could not be established. Small on purpose: it covers a backend that is
// still binding its port, not one that is down.
const DialRetries = 2

// The wait between dial attempts is bounded by these.
const (
	minDialBackoff = 100 * time.Millisecond
	maxDialBackoff = time.Second
)

// Successful reports whether status is 2xx.
func Successful(status int) bool {
	return status >= http.StatusOK && status < http.StatusMultipleChoices
}

// ReadErrorBody reads at most ErrorBodyLimit bytes and trims surrounding
// space. Read errors are discarded: a body that cannot be read is reported as
// the empty string, since the status alone is still worth surfacing.
func ReadErrorBody(r io.Reader) string {
	raw, _ := io.ReadAll(io.LimitReader(r, ErrorBodyLimit))
	return strings.TrimSpace(string(raw))
}

// Join appends path to a base URL, tolerating any number of trailing slashes.
func Join(baseURL, path string) string {
	return strings.TrimRight(baseURL, "/") + path
}

// SetAuth sets a bearer token when apiKey is non-empty. Local backends
// generally ignore it; hosted ones require it.
func SetAuth(req *http.Request, apiKey string) {
	if apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}
}

// Timeout returns d, or fallback when d is non-positive.
func Timeout(d, fallback time.Duration) time.Duration {
	if d <= 0 {
		return fallback
	}
	return d
}

// Client returns supplied when non-nil, so a caller who provided a client gets
// exactly that client — their transport, their timeout, their redirect policy.
//
// The default client deliberately has no Timeout: per-request deadlines come
// from the context, so that streaming and non-streaming can be bounded
// differently. It also refuses redirects. A redirect from a local inference
// backend is a misconfiguration rather than a route, and following one both
// hides that and forwards the Authorization header across a scheme or port
// change on the same host. Returning the 3xx makes it fail Successful and
// surface as an ordinary API error carrying the body.
func Client(supplied *http.Client) *http.Client {
	if supplied != nil {
		return supplied
	}
	return &http.Client{
		CheckRedirect: func(_ *http.Request, _ []*http.Request) error {
			return http.ErrUseLastResponse
		},
	}
}

// Do sends req, retrying only when the connection could not be established.
//
// A dial failure proves the request never reached the server, so retrying
// cannot re-run a completion or re-emit stream tokens. Every other transport
// error is returned unchanged — a reset mid-response may follow a request the
// server already acted on, and an HTTP error status is a response, which is
// the caller's to classify.
func Do(c *http.Client, req *http.Request) (*http.Response, error) {
	var lastErr error
	for attempt := 0; attempt <= DialRetries; attempt++ {
		if attempt > 0 {
			if err := rewind(req); err != nil {
				return nil, lastErr
			}
			select {
			case <-req.Context().Done():
				return nil, req.Context().Err()
			case <-time.After(Backoff(attempt-1, minDialBackoff, maxDialBackoff)):
			}
		}
		resp, err := c.Do(req) //nolint:gosec // req is built from the library's own Options, not attacker input
		if err == nil {
			return resp, nil
		}
		lastErr = err
		if !isDialFailure(err) {
			return nil, err
		}
	}
	return nil, lastErr
}

// errNoRewind means a request body cannot be replayed, so the request must not
// be retried.
var errNoRewind = errors.New("httpx: request body cannot be rewound")

// rewind restores a request body that the previous attempt consumed.
//
// The failed attempt drained the body and http.Client closed it, so sending
// the same request again would transmit an empty one — a silent corruption
// that only appears on the retry path. http.NewRequestWithContext supplies
// GetBody for the in-memory readers this module uses; a body without one
// cannot be replayed safely, and the caller gets the original dial error
// rather than a request the backend would misread.
func rewind(req *http.Request) error {
	if req.Body == nil {
		return nil
	}
	if req.GetBody == nil {
		return errNoRewind
	}
	body, err := req.GetBody()
	if err != nil {
		return errNoRewind
	}
	req.Body = body
	return nil
}

// isDialFailure reports whether err is the connection never being established,
// as opposed to a failure partway through an exchange.
func isDialFailure(err error) bool {
	var opErr *net.OpError
	return errors.As(err, &opErr) && opErr.Op == "dial"
}

// Backoff returns the wait before the given zero-based attempt, doubling from
// minD and capped at maxD.
func Backoff(attempt int, minD, maxD time.Duration) time.Duration {
	d := minD
	for range attempt {
		d *= 2
		if d >= maxD {
			return maxD
		}
	}
	return min(d, maxD)
}
