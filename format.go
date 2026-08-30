package llm

// Structured-output modes, matching the OpenAI `response_format.type` values.
const (
	// FormatJSONObject asks for syntactically valid JSON with no schema.
	// Widely supported.
	FormatJSONObject = "json_object"
	// FormatJSONSchema asks for grammar-constrained decoding against a schema,
	// which makes the shape a property of the sampler rather than a hope about
	// the prompt. Supported by LM Studio, vLLM and OpenAI; absent elsewhere.
	FormatJSONSchema = "json_schema"
)

// ResponseFormat requests structured output.
type ResponseFormat struct {
	Type       string      `json:"type"`
	JSONSchema *JSONSchema `json:"json_schema,omitempty"`

	// Required disables the fallback chain. By default a backend that rejects
	// this format is retried at a weaker level; with Required set, that
	// rejection is returned to the caller instead. Set it when malformed
	// output is worse than no output — an unattended batch job writing to a
	// database, say, rather than an interactive prompt a human will read.
	Required bool `json:"-"`
}

// JSONSchema is the schema half of a FormatJSONSchema request. Schema holds
// the JSON Schema document, either as a map or as any value that marshals to
// one.
type JSONSchema struct {
	Name   string `json:"name"`
	Strict bool   `json:"strict"`
	Schema any    `json:"schema"`
}

// JSONObject asks for valid JSON without a schema.
func JSONObject() *ResponseFormat {
	return &ResponseFormat{Type: FormatJSONObject}
}

// JSONSchemaOf asks for output constrained to schema. name is an arbitrary
// label the backend may echo back in errors.
func JSONSchemaOf(name string, schema any) *ResponseFormat {
	return &ResponseFormat{
		Type:       FormatJSONSchema,
		JSONSchema: &JSONSchema{Name: name, Strict: true, Schema: schema},
	}
}

// degradeChain returns the formats to try, strongest first. A nil format means
// "send no response_format at all", which is both the free-text case and the
// last resort after every structured mode has been rejected.
func (f *ResponseFormat) degradeChain() []*ResponseFormat {
	switch {
	case f == nil:
		return []*ResponseFormat{nil}
	case f.Required:
		return []*ResponseFormat{f}
	case f.Type == FormatJSONSchema:
		return []*ResponseFormat{f, JSONObject(), nil}
	default:
		return []*ResponseFormat{f, nil}
	}
}
