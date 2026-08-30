package httpx_test

import (
	"bytes"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/openserbia/go-llm/internal/httpx"
)

func TestSuccessful(t *testing.T) {
	for status, want := range map[int]bool{199: false, 200: true, 204: true, 299: true, 300: false, 400: false, 500: false} {
		if got := httpx.Successful(status); got != want {
			t.Errorf("Successful(%d) = %v, want %v", status, got, want)
		}
	}
}

func TestReadErrorBodyIsBoundedAndTrimmed(t *testing.T) {
	if got := httpx.ReadErrorBody(strings.NewReader("  boom\n")); got != "boom" {
		t.Errorf("ReadErrorBody = %q, want %q", got, "boom")
	}
	// A backend answering an error with a megabyte of HTML must not be
	// quoted back in full.
	huge := strings.Repeat("x", httpx.ErrorBodyLimit*2)
	if got := httpx.ReadErrorBody(strings.NewReader(huge)); len(got) != httpx.ErrorBodyLimit {
		t.Errorf("ReadErrorBody length = %d, want %d", len(got), httpx.ErrorBodyLimit)
	}
}

func TestJoin(t *testing.T) {
	for _, tt := range []struct{ base, path, want string }{
		{"http://h/v1", "/chat", "http://h/v1/chat"},
		{"http://h/v1/", "/chat", "http://h/v1/chat"},
		{"http://h/v1///", "/chat", "http://h/v1/chat"},
	} {
		if got := httpx.Join(tt.base, tt.path); got != tt.want {
			t.Errorf("Join(%q, %q) = %q, want %q", tt.base, tt.path, got, tt.want)
		}
	}
}

func TestSetAuthOnlyWhenKeyPresent(t *testing.T) {
	req, _ := http.NewRequestWithContext(t.Context(), http.MethodGet, "http://h", http.NoBody)
	httpx.SetAuth(req, "")
	if got := req.Header.Get("Authorization"); got != "" {
		t.Errorf("Authorization = %q, want empty", got)
	}
	httpx.SetAuth(req, "secret")
	if got := req.Header.Get("Authorization"); got != "Bearer secret" {
		t.Errorf("Authorization = %q", got)
	}
}

func TestTimeoutFallback(t *testing.T) {
	if got := httpx.Timeout(0, time.Minute); got != time.Minute {
		t.Errorf("Timeout(0) = %v, want 1m", got)
	}
	if got := httpx.Timeout(-1, time.Minute); got != time.Minute {
		t.Errorf("Timeout(-1) = %v, want 1m", got)
	}
	if got := httpx.Timeout(time.Second, time.Minute); got != time.Second {
		t.Errorf("Timeout(1s) = %v, want 1s", got)
	}
}

func TestClientDoesNotFollowRedirects(t *testing.T) {
	var reachedTarget bool
	target := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		reachedTarget = true
		w.WriteHeader(http.StatusOK)
	}))
	t.Cleanup(target.Close)

	redirector := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, target.URL, http.StatusTemporaryRedirect)
	}))
	t.Cleanup(redirector.Close)

	req, _ := http.NewRequestWithContext(t.Context(), http.MethodGet, redirector.URL, http.NoBody)
	resp, err := httpx.Client(nil).Do(req)
	if err != nil {
		t.Fatalf("Do: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()

	// The 3xx is handed back rather than followed, so it fails Successful and
	// surfaces as an ordinary API error naming the redirect.
	if resp.StatusCode != http.StatusTemporaryRedirect {
		t.Errorf("StatusCode = %d, want 307", resp.StatusCode)
	}
	if reachedTarget {
		t.Error("redirect was followed, want it refused")
	}
}

func TestClientKeepsSuppliedClientUntouched(t *testing.T) {
	supplied := &http.Client{Timeout: 3 * time.Second}
	got := httpx.Client(supplied)
	if got != supplied {
		t.Error("Client replaced a caller-supplied client, want it used as given")
	}
	if got.CheckRedirect != nil {
		t.Error("Client imposed a redirect policy on a caller-supplied client")
	}
}

func TestClientDefaultHasNoTimeout(t *testing.T) {
	// A client-level timeout would sever a long stream mid-response; per
	// request deadlines come from the context instead.
	if got := httpx.Client(nil).Timeout; got != 0 {
		t.Errorf("default client Timeout = %v, want 0", got)
	}
}

func TestDoRetriesDialFailures(t *testing.T) {
	// Nothing is listening, so every attempt fails to dial.
	req, _ := http.NewRequestWithContext(t.Context(), http.MethodGet, "http://127.0.0.1:1", http.NoBody)
	resp, err := httpx.Do(httpx.Client(nil), req)
	if resp != nil {
		_ = resp.Body.Close()
	}
	if err == nil {
		t.Fatal("Do succeeded against a closed port, want error")
	}
}

func TestDoReturnsNonDialErrorsUnchanged(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	t.Cleanup(srv.Close)

	req, _ := http.NewRequestWithContext(t.Context(), http.MethodGet, srv.URL, http.NoBody)
	resp, err := httpx.Do(httpx.Client(nil), req)
	if err != nil {
		t.Fatalf("Do: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()
	// A 5xx is a response, not a dial failure: it is handed back for the
	// caller to classify, never retried.
	if resp.StatusCode != http.StatusInternalServerError {
		t.Errorf("StatusCode = %d, want 500", resp.StatusCode)
	}
}

func TestBackoffGrowsAndCaps(t *testing.T) {
	const minD, maxD = time.Second, 10 * time.Second
	if got := httpx.Backoff(0, minD, maxD); got != minD {
		t.Errorf("Backoff(0) = %v, want %v", got, minD)
	}
	if got := httpx.Backoff(1, minD, maxD); got != 2*time.Second {
		t.Errorf("Backoff(1) = %v, want 2s", got)
	}
	if got := httpx.Backoff(99, minD, maxD); got != maxD {
		t.Errorf("Backoff(99) = %v, want %v", got, maxD)
	}
}

// flakyTransport fails the first attempt with a dial error and records the
// body each attempt actually carried.
type flakyTransport struct {
	attempts int
	bodies   []string
}

func (f *flakyTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	f.attempts++
	got := ""
	if r.Body != nil {
		raw, _ := io.ReadAll(r.Body)
		got = string(raw)
	}
	f.bodies = append(f.bodies, got)
	if f.attempts == 1 {
		return nil, &net.OpError{Op: "dial", Err: errors.New("connection refused")}
	}
	return &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader("ok")),
		Header:     make(http.Header),
	}, nil
}

func TestDoRetryResendsTheBody(t *testing.T) {
	const payload = `{"model":"m"}`
	transport := &flakyTransport{}

	req, err := http.NewRequestWithContext(t.Context(), http.MethodPost,
		"http://example.invalid/v1/chat", bytes.NewReader([]byte(payload)))
	if err != nil {
		t.Fatalf("NewRequestWithContext: %v", err)
	}

	resp, err := httpx.Do(&http.Client{Transport: transport}, req)
	if err != nil {
		t.Fatalf("Do: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if transport.attempts != 2 {
		t.Fatalf("attempts = %d, want 2", transport.attempts)
	}
	// The first attempt drains the body and http.Client closes it. Without a
	// rewind the retry reaches the backend with an empty body, which it would
	// answer with a confusing 400 rather than an obvious failure.
	if transport.bodies[1] != payload {
		t.Errorf("retry sent body %q, want %q", transport.bodies[1], payload)
	}
}

func TestDoDoesNotRetryUnrewindableBody(t *testing.T) {
	transport := &flakyTransport{}

	// A stream with no GetBody: http.NewRequestWithContext cannot supply one
	// for an arbitrary reader.
	req, err := http.NewRequestWithContext(t.Context(), http.MethodPost,
		"http://example.invalid/v1/chat", struct{ io.Reader }{strings.NewReader("x")})
	if err != nil {
		t.Fatalf("NewRequestWithContext: %v", err)
	}

	resp, err := httpx.Do(&http.Client{Transport: transport}, req)
	if resp != nil {
		_ = resp.Body.Close()
	}
	if err == nil {
		t.Fatal("Do succeeded, want the dial error surfaced")
	}
	if transport.attempts != 1 {
		t.Errorf("attempts = %d, want 1 — an unrewindable body must not be retried", transport.attempts)
	}
}
