package llm_test

import (
	"encoding/json/v2"
	"testing"

	"github.com/openserbia/go-llm"
)

func TestJSONSchemaOfMarshalsStrictSchema(t *testing.T) {
	format := llm.JSONSchemaOf("translation", map[string]any{
		"type":       "object",
		"properties": map[string]any{"text": map[string]any{"type": "string"}},
	})

	raw, err := json.Marshal(format)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}

	var got map[string]any
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if got["type"] != llm.FormatJSONSchema {
		t.Errorf("type = %v, want %v", got["type"], llm.FormatJSONSchema)
	}
	schema, ok := got["json_schema"].(map[string]any)
	if !ok {
		t.Fatalf("json_schema missing from %v", got)
	}
	if schema["name"] != "translation" {
		t.Errorf("name = %v, want translation", schema["name"])
	}
	if schema["strict"] != true {
		t.Errorf("strict = %v, want true", schema["strict"])
	}
	// Required is control flow for this library, not part of the wire format.
	if _, present := got["required"]; present {
		t.Error("Required leaked into the request body")
	}
}

func TestJSONObjectHasNoSchema(t *testing.T) {
	raw, err := json.Marshal(llm.JSONObject())
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	if got, want := string(raw), `{"type":"json_object"}`; got != want {
		t.Errorf("Marshal = %s, want %s", got, want)
	}
}
