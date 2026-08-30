package embed_test

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/openserbia/go-llm/embed"
)

func vectorServer(t *testing.T, vectors ...[]float32) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		type item struct {
			Embedding []float32 `json:"embedding"`
		}
		payload := struct {
			Data []item `json:"data"`
		}{}
		for _, v := range vectors {
			payload.Data = append(payload.Data, item{Embedding: v})
		}
		_ = json.NewEncoder(w).Encode(payload)
	}))
	t.Cleanup(srv.Close)
	return srv
}

func TestEmbedReturnsVectorsInOrder(t *testing.T) {
	srv := vectorServer(t, []float32{1, 0, 0}, []float32{0, 2, 0})

	client, err := embed.New(embed.Options{BaseURL: srv.URL, Model: "bge-m3"})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	got, err := client.Embed(context.Background(), []string{"a", "b"})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("got %d vectors, want 2", len(got))
	}
	// Without Normalize the raw magnitudes must survive.
	if got[1][1] != 2 {
		t.Errorf("vector[1][1] = %v, want 2 (unnormalized)", got[1][1])
	}
}

func TestEmbedNormalize(t *testing.T) {
	srv := vectorServer(t, []float32{3, 4})

	client, err := embed.New(embed.Options{BaseURL: srv.URL, Model: "bge-m3", Normalize: true})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	got, err := client.Embed(context.Background(), []string{"a"})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if math.Abs(float64(got[0][0])-0.6) > 1e-6 || math.Abs(float64(got[0][1])-0.8) > 1e-6 {
		t.Errorf("normalized = %v, want [0.6 0.8]", got[0])
	}
}

func TestEmbedDimensionMismatch(t *testing.T) {
	srv := vectorServer(t, []float32{1, 2, 3})

	client, err := embed.New(embed.Options{BaseURL: srv.URL, Model: "wrong-model", Dimensions: 1024})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, err = client.Embed(context.Background(), []string{"a"})

	var mismatch *embed.DimensionMismatchError
	if !errors.As(err, &mismatch) {
		t.Fatalf("err = %v, want *embed.DimensionMismatchError", err)
	}
	if mismatch.Got != 3 || mismatch.Want != 1024 {
		t.Errorf("mismatch = %+v, want got=3 want=1024", mismatch)
	}
}

func TestEmbedCountMismatch(t *testing.T) {
	srv := vectorServer(t, []float32{1, 2})

	client, err := embed.New(embed.Options{BaseURL: srv.URL, Model: "m"})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	// Two inputs, one vector back: silently zipping these would misalign every
	// vector with its text.
	if _, err := client.Embed(context.Background(), []string{"a", "b"}); err == nil {
		t.Fatal("Embed succeeded on a short response, want error")
	}
}

func TestEmbedEmptyInputSkipsRequest(t *testing.T) {
	called := false
	srv := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { called = true }))
	t.Cleanup(srv.Close)

	client, err := embed.New(embed.Options{BaseURL: srv.URL, Model: "m"})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	got, err := client.Embed(context.Background(), nil)
	if err != nil || got != nil {
		t.Fatalf("Embed = %v, %v; want nil, nil", got, err)
	}
	if called {
		t.Error("empty input still contacted the backend")
	}
}

func TestL2NormalizeZeroVector(t *testing.T) {
	v := []float32{0, 0}
	if got := embed.L2Normalize(v); got[0] != 0 || got[1] != 0 {
		t.Errorf("L2Normalize = %v, want the zero vector unchanged", got)
	}
}

func TestTruncateCountsRunesNotBytes(t *testing.T) {
	// Serbian Cyrillic is two bytes per rune: a byte-based clamp would cut a
	// character in half and produce invalid UTF-8.
	const s = "Београд"
	if got := embed.Truncate(s, 3); got != "Бео" {
		t.Errorf("Truncate = %q, want %q", got, "Бео")
	}
	if got := embed.Truncate(s, 0); got != s {
		t.Errorf("Truncate with 0 = %q, want unchanged", got)
	}
	if got := embed.Truncate(s, 100); got != s {
		t.Errorf("Truncate beyond length = %q, want unchanged", got)
	}
}
