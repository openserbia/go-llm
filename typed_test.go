package llm_test

import (
	"context"
	"encoding/json/v2"
	"net/http"
	"testing"

	"github.com/openserbia/go-llm"
)

type answer struct {
	City       string `json:"city"`
	Population int    `json:"population"`
}

func TestChatAsDecodesAndDerivesSchema(t *testing.T) {
	rec := &capture{}
	srv := newServer(t, rec, func(_ int, w http.ResponseWriter) {
		okResponse(w, `{\"city\":\"Novi Sad\",\"population\":231798}`)
	})

	got, err := llm.ChatAs[answer](context.Background(), newClient(t, srv.URL), llm.ChatRequest{
		Messages: []llm.Message{llm.User("describe Novi Sad")},
	})
	if err != nil {
		t.Fatalf("ChatAs: %v", err)
	}
	if got.City != "Novi Sad" || got.Population != 231798 {
		t.Errorf("got %+v", got)
	}
	if rec.formats[0] == nil || rec.formats[0].Type != llm.FormatJSONSchema {
		t.Fatalf("format = %+v, want a derived json_schema", rec.formats[0])
	}
	if rec.formats[0].JSONSchema.Name != "answer" {
		t.Errorf("schema name = %q, want %q", rec.formats[0].JSONSchema.Name, "answer")
	}
}

func TestChatAsKeepsCallerSuppliedFormat(t *testing.T) {
	rec := &capture{}
	srv := newServer(t, rec, func(_ int, w http.ResponseWriter) {
		okResponse(w, `{\"city\":\"Niš\",\"population\":183164}`)
	})

	if _, err := llm.ChatAs[answer](context.Background(), newClient(t, srv.URL), llm.ChatRequest{
		Messages:       []llm.Message{llm.User("hi")},
		ResponseFormat: llm.JSONSchemaOf("handwritten", map[string]any{"type": "object"}),
	}); err != nil {
		t.Fatalf("ChatAs: %v", err)
	}
	if rec.formats[0].JSONSchema.Name != "handwritten" {
		t.Errorf("schema name = %q, want the caller's schema untouched", rec.formats[0].JSONSchema.Name)
	}
}

func TestChatAsRepairsOnce(t *testing.T) {
	rec := &capture{}
	srv := newServer(t, rec, func(attempt int, w http.ResponseWriter) {
		if attempt == 1 {
			// Valid JSON, wrong shape: passes Chat's verify, fails the decode.
			okResponse(w, `{\"city\":\"Zrenjanin\",\"population\":\"lots\"}`)
			return
		}
		okResponse(w, `{\"city\":\"Zrenjanin\",\"population\":67129}`)
	})

	got, err := llm.ChatAs[answer](context.Background(), newClient(t, srv.URL), llm.ChatRequest{
		Messages: []llm.Message{llm.User("hi")},
	})
	if err != nil {
		t.Fatalf("ChatAs: %v", err)
	}
	if got.Population != 67129 {
		t.Errorf("Population = %d, want 67129", got.Population)
	}
	if len(rec.bodies) != 2 {
		t.Fatalf("requests = %d, want 2", len(rec.bodies))
	}

	var repair struct {
		Messages []llm.Message `json:"messages"`
	}
	if err := json.Unmarshal(rec.bodies[1], &repair); err != nil {
		t.Fatalf("decode repair request: %v", err)
	}
	if len(repair.Messages) != 3 {
		t.Fatalf("repair messages = %d, want 3", len(repair.Messages))
	}
	if repair.Messages[1].Role != llm.RoleAssistant {
		t.Errorf("message 2 role = %q, want assistant", repair.Messages[1].Role)
	}
	if repair.Messages[2].Role != llm.RoleUser {
		t.Errorf("message 3 role = %q, want user", repair.Messages[2].Role)
	}
}

func TestChatAsRepairsUnknownMembers(t *testing.T) {
	rec := &capture{}
	srv := newServer(t, rec, func(attempt int, w http.ResponseWriter) {
		if attempt == 1 {
			// The schema said additionalProperties:false. Under encoding/json
			// v1 this would decode silently and drop the extra field.
			okResponse(w, `{\"city\":\"Subotica\",\"population\":105681,\"note\":\"invented\"}`)
			return
		}
		okResponse(w, `{\"city\":\"Subotica\",\"population\":105681}`)
	})

	if _, err := llm.ChatAs[answer](context.Background(), newClient(t, srv.URL), llm.ChatRequest{
		Messages: []llm.Message{llm.User("hi")},
	}); err != nil {
		t.Fatalf("ChatAs: %v", err)
	}
	if len(rec.bodies) != 2 {
		t.Errorf("requests = %d, want 2 — an unknown member must trigger repair", len(rec.bodies))
	}
}

func TestChatAsGivesUpAfterOneRepair(t *testing.T) {
	rec := &capture{}
	srv := newServer(t, rec, func(_ int, w http.ResponseWriter) {
		okResponse(w, `{\"city\":\"Pančevo\",\"population\":\"still not a number\"}`)
	})

	if _, err := llm.ChatAs[answer](context.Background(), newClient(t, srv.URL), llm.ChatRequest{
		Messages: []llm.Message{llm.User("hi")},
	}); err == nil {
		t.Fatal("ChatAs succeeded, want error")
	}
	if len(rec.bodies) != 2 {
		t.Errorf("requests = %d, want 2 — one repair, not a loop", len(rec.bodies))
	}
}
