// Package llm is a client for OpenAI-compatible chat completion endpoints.
//
// It targets the subset of the OpenAI API that self-hosted backends actually
// implement — LM Studio, Ollama, vLLM, llama.cpp, text-generation-inference —
// and degrades gracefully when a backend is missing a feature rather than
// requiring the caller to know which backend it is talking to.
//
// The three pieces most callers need:
//
//	client, err := llm.New(llm.Options{BaseURL: "http://localhost:1234/v1", Model: "gemma-3-12b-it"})
//	resp, err := client.Chat(ctx, llm.ChatRequest{Messages: []llm.Message{llm.User("hello")}})
//
// Structured output is requested through ResponseFormat. JSONSchema asks for
// grammar-constrained decoding against a schema; JSONObject asks only for
// "some valid JSON". A backend that rejects either one is detected from its
// error response, and a backend that accepts the field and ignores it is
// detected from the bytes it returns, so the same code path works against a
// backend with a grammar engine and one without — and, unlike an error-only
// check, notices the second case at all.
//
// ChatAs derives the schema from a Go type and decodes into it:
//
//	type Answer struct {
//		City       string `json:"city"`
//		Population int    `json:"population"`
//	}
//	got, err := llm.ChatAs[Answer](ctx, client, llm.ChatRequest{
//		Messages: []llm.Message{llm.User("describe Novi Sad")},
//	})
//
// Sibling packages cover the rest of a local inference stack:
// [github.com/openserbia/go-llm/embed] for embeddings,
// [github.com/openserbia/go-llm/rerank] for cross-encoder reranking, and
// [github.com/openserbia/go-llm/lmstudio] for LM Studio's native model API
// and model-load preflight.
//
// The library reads no environment variables and parses no config files.
// Everything is passed in through Options; mapping your own configuration
// onto it is the caller's job.
package llm
