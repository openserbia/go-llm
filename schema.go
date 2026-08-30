package llm

import (
	"fmt"
	"reflect"
	"strings"
	"time"
)

// SchemaOf derives a strict-mode JSON Schema from T.
//
// Strict mode has structural requirements that are easy to violate by hand and
// that backends report as an opaque 400: every object must set
// additionalProperties to false, and every property must appear in required.
// Deriving the schema from the type makes both unhittable.
//
// One consequence is worth knowing before you hit it: `omitempty` does not make
// a field optional here. Strict mode requires every property in required, so
// optionality is expressed as a pointer, which becomes a nullable union.
//
// Types with no faithful strict-mode expression — maps, interfaces, channels,
// functions, and anything recursive — are refused rather than turned into a
// schema the backend would reject. Write the schema by hand and pass it
// through JSONSchemaOf if you need one of those.
func SchemaOf[T any]() (*ResponseFormat, error) {
	t := reflect.TypeFor[T]()
	schema, err := schemaFor(t, map[reflect.Type]bool{})
	if err != nil {
		return nil, err
	}
	name := t.Name()
	if name == "" {
		name = "response"
	}
	return JSONSchemaOf(name, schema), nil
}

var timeType = reflect.TypeFor[time.Time]()

// jsonTypeString is the JSON Schema "string" type value and jsonKeyType the
// "type" keyword. Both recur often enough to trip goconst, so they are named.
const (
	jsonTypeString = "string"
	jsonKeyType    = "type"
)

func schemaFor(t reflect.Type, seen map[reflect.Type]bool) (map[string]any, error) {
	if t == timeType {
		return map[string]any{jsonKeyType: jsonTypeString, "format": "date-time"}, nil
	}

	switch t.Kind() {
	case reflect.Bool:
		return map[string]any{jsonKeyType: "boolean"}, nil
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return map[string]any{jsonKeyType: "integer"}, nil
	case reflect.Float32, reflect.Float64:
		return map[string]any{jsonKeyType: "number"}, nil
	case reflect.String:
		return map[string]any{jsonKeyType: jsonTypeString}, nil

	case reflect.Pointer:
		if t.Elem().Kind() == reflect.Pointer {
			return nil, fmt.Errorf("llm: SchemaOf: %s: a pointer to a pointer has no JSON Schema form", t)
		}
		inner, err := schemaFor(t.Elem(), seen)
		if err != nil {
			return nil, err
		}
		inner[jsonKeyType] = []any{inner[jsonKeyType], "null"}
		return inner, nil

	case reflect.Slice:
		// encoding/json encodes a byte slice as a base64 string.
		if t.Elem().Kind() == reflect.Uint8 {
			return map[string]any{jsonKeyType: jsonTypeString}, nil
		}
		return arraySchema(t, seen)
	case reflect.Array:
		return arraySchema(t, seen)

	case reflect.Struct:
		return structSchema(t, seen)

	default:
		return nil, fmt.Errorf("llm: SchemaOf: cannot express %s (kind %s) in a strict JSON schema", t, t.Kind())
	}
}

func arraySchema(t reflect.Type, seen map[reflect.Type]bool) (map[string]any, error) {
	items, err := schemaFor(t.Elem(), seen)
	if err != nil {
		return nil, err
	}
	return map[string]any{jsonKeyType: "array", "items": items}, nil
}

func structSchema(t reflect.Type, seen map[reflect.Type]bool) (map[string]any, error) {
	if seen[t] {
		// Strict-mode support for $ref recursion is inconsistent across local
		// backends, so a recursive type is refused rather than emitted as a
		// schema that works on one server and 400s on the next.
		return nil, fmt.Errorf("llm: SchemaOf: %s is recursive, which has no portable strict-mode form", t)
	}
	seen[t] = true
	defer delete(seen, t)

	properties := map[string]any{}
	required := []string{}
	if err := collectFields(t, seen, properties, &required); err != nil {
		return nil, err
	}
	return map[string]any{
		jsonKeyType:            "object",
		"properties":           properties,
		"required":             required,
		"additionalProperties": false,
	}, nil
}

// collectFields walks a struct's fields in declaration order, flattening
// embedded structs the way encoding/json does so the schema matches what the
// decoder will actually accept.
func collectFields(t reflect.Type, seen map[reflect.Type]bool, properties map[string]any, required *[]string) error {
	for i := range t.NumField() {
		field := t.Field(i)

		name, _, _ := strings.Cut(field.Tag.Get("json"), ",")
		if name == "-" {
			continue
		}

		if field.Anonymous && name == "" {
			embeddedType := field.Type
			if embeddedType.Kind() == reflect.Pointer {
				embeddedType = embeddedType.Elem()
			}
			if embeddedType.Kind() == reflect.Struct && embeddedType != timeType {
				if err := collectFields(embeddedType, seen, properties, required); err != nil {
					return err
				}
				continue
			}
		}

		if !field.IsExported() {
			continue
		}
		if name == "" {
			name = field.Name
		}

		schema, err := schemaFor(field.Type, seen)
		if err != nil {
			return fmt.Errorf("%s.%s: %w", t.Name(), field.Name, err)
		}
		properties[name] = schema
		*required = append(*required, name)
	}
	return nil
}
