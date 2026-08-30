package llm_test

import (
	"encoding/json/v2"
	"strings"
	"testing"
	"time"

	"github.com/openserbia/go-llm"
)

type address struct {
	City string `json:"city"`
}

type person struct {
	Name      string    `json:"name"`
	Age       int       `json:"age"`
	Height    float64   `json:"height"`
	Verified  bool      `json:"verified"`
	Nickname  *string   `json:"nickname"`
	Tags      []string  `json:"tags"`
	Home      address   `json:"home"`
	CreatedAt time.Time `json:"created_at"`
	Secret    string    `json:"-"`
	unexposed int       //nolint:unused // exercised through reflection, never referenced
}

func TestSchemaOfStruct(t *testing.T) {
	format, err := llm.SchemaOf[person]()
	if err != nil {
		t.Fatalf("SchemaOf: %v", err)
	}
	if format.Type != llm.FormatJSONSchema {
		t.Errorf("Type = %q, want %q", format.Type, llm.FormatJSONSchema)
	}
	if format.JSONSchema.Name != "person" {
		t.Errorf("Name = %q, want %q", format.JSONSchema.Name, "person")
	}
	if !format.JSONSchema.Strict {
		t.Error("Strict = false, want true")
	}

	// v2 does not sort map keys unless asked, so the golden comparison forces
	// it. On the wire the order is irrelevant — JSON objects are unordered —
	// but a golden test needs a stable rendering.
	got, err := json.Marshal(format.JSONSchema.Schema, json.Deterministic(true))
	if err != nil {
		t.Fatalf("marshal schema: %v", err)
	}
	const want = `{"additionalProperties":false,"properties":{` +
		`"age":{"type":"integer"},` +
		`"created_at":{"format":"date-time","type":"string"},` +
		`"height":{"type":"number"},` +
		`"home":{"additionalProperties":false,"properties":{"city":{"type":"string"}},"required":["city"],"type":"object"},` +
		`"name":{"type":"string"},` +
		`"nickname":{"type":["string","null"]},` +
		`"tags":{"items":{"type":"string"},"type":"array"},` +
		`"verified":{"type":"boolean"}},` +
		`"required":["name","age","height","verified","nickname","tags","home","created_at"],` +
		`"type":"object"}`
	if string(got) != want {
		t.Errorf("schema mismatch\n got: %s\nwant: %s", got, want)
	}
}

type embedded struct {
	ID string `json:"id"`
}

type withEmbedded struct {
	embedded
	Note string `json:"note"`
}

func TestSchemaOfFlattensEmbedded(t *testing.T) {
	format, err := llm.SchemaOf[withEmbedded]()
	if err != nil {
		t.Fatalf("SchemaOf: %v", err)
	}
	got, err := json.Marshal(format.JSONSchema.Schema, json.Deterministic(true))
	if err != nil {
		t.Fatalf("marshal schema: %v", err)
	}
	if !strings.Contains(string(got), `"id":{"type":"string"}`) {
		t.Errorf("embedded field not promoted: %s", got)
	}
	if !strings.Contains(string(got), `"required":["id","note"]`) {
		t.Errorf("required list wrong: %s", got)
	}
}

type recursive struct {
	Next *recursive `json:"next"`
}

type withMap struct {
	Attrs map[string]string `json:"attrs"`
}

type withAny struct {
	Payload any `json:"payload"`
}

func TestSchemaOfRejectsInexpressibleTypes(t *testing.T) {
	if _, err := llm.SchemaOf[recursive](); err == nil {
		t.Error("SchemaOf[recursive] succeeded, want an error")
	}
	if _, err := llm.SchemaOf[withMap](); err == nil {
		t.Error("SchemaOf[withMap] succeeded, want an error")
	}
	if _, err := llm.SchemaOf[withAny](); err == nil {
		t.Error("SchemaOf[withAny] succeeded, want an error")
	}
	if _, err := llm.SchemaOf[int](); err != nil {
		t.Errorf("SchemaOf[int] = %v, want a bare integer schema", err)
	}
}
