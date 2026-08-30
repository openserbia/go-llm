# go-llm

A Go client for OpenAI-compatible inference endpoints, written against what
self-hosted backends actually implement.

LM Studio, Ollama, vLLM, llama.cpp and text-generation-inference all claim to
speak the OpenAI API, and all of them are missing a different part of it. This
library detects those gaps — from the backend's error responses, and from the
bytes a request actually returns — and degrades instead of failing, so the
calling code does not have to know which backend it is pointed at.

Zero dependencies outside the standard library. Requires Go 1.27 — the library
uses `encoding/json/v2`, which decodes into a typed result strictly enough to
catch a model inventing a field.

```bash
go get github.com/openserbia/go-llm
```

## Chat

```go
client, err := llm.New(llm.Options{
    BaseURL: "http://localhost:1234/v1",
    Model:   "gemma-3-12b-it",
})

resp, err := client.Chat(ctx, llm.ChatRequest{
    Messages: []llm.Message{
        llm.System("Answer in one sentence."),
        llm.User("What is a paušalac?"),
    },
    Temperature: 0.1,
})
fmt.Println(resp.Content, resp.Usage.TotalTokens)
```

`EnsureModel` fails at startup, naming what the backend does serve, rather than
letting a typo surface as a confusing first request:

```go
if err := client.EnsureModel(ctx, ""); err != nil {
    return err // llm: model "gemma-3-12b" is not available at ... (available: gemma-3-12b-it, bge-m3)
}
```

## Structured output

`JSONSchemaOf` asks for grammar-constrained decoding: the backend's sampler is
restricted to tokens that keep the output valid against the schema, so malformed
JSON becomes impossible rather than unlikely.

```go
resp, err := client.Chat(ctx, llm.ChatRequest{
    Messages: []llm.Message{llm.User("Translate to Russian: dobar dan")},
    ResponseFormat: llm.JSONSchemaOf("translation", map[string]any{
        "type":       "object",
        "properties": map[string]any{"text": map[string]any{"type": "string"}},
        "required":   []string{"text"},
    }),
})
```

Not every backend has a grammar engine. When one rejects the schema, the request
is retried as plain JSON mode, then as unconstrained text:

```
json_schema  ->  json_object  ->  no response_format
```

Set `Required` when that is the wrong trade — when unparseable output is worse
than an error:

```go
format := llm.JSONSchemaOf("translation", schema)
format.Required = true // fail instead of degrading
```

### Typed responses

`ChatAs` derives the schema from the type, decodes into it, and retries once
with the decoder's error if the model got it wrong:

```go
type Answer struct {
    City       string `json:"city"`
    Population int    `json:"population"`
}

got, err := llm.ChatAs[Answer](ctx, client, llm.ChatRequest{
    Messages: []llm.Message{llm.User("describe Novi Sad")},
})
```

The derived schema sets `additionalProperties: false` and lists every property
in `required`, which strict mode demands and which backends report as an opaque
400 when you get it wrong by hand. `omitempty` does not make a field optional —
use a pointer, which becomes a nullable union. The decode rejects unknown
members too, so a reply carrying a field the schema forbade is the failure that
spends the one repair retry rather than being silently dropped.

`ChatAs` takes the client as an argument rather than being a method, so it
composes with `OverflowRetry` and any other wrapper. Types that have no
portable strict-mode form — maps, `any`, and anything recursive — are refused
by `SchemaOf` up front; write the schema by hand and pass it through
`JSONSchemaOf` if you need one.

### When a backend ignores the request

Not every backend that accepts `response_format` honours it. Ollama's
OpenAI-compatible endpoint takes the field and ignores it, because its
structured output lives on a separate parameter — the reply is a 200 with
ordinary prose. `Chat` checks that a structured request produced JSON and
treats a prose answer the same as an outright rejection, so it degrades rather
than handing you unconstrained text that looks like a success. Set
`ResponseFormat.Required` to get a `*llm.FormatIgnoredError` instead.

A separate case does not degrade: a structured reply cut short by the token
budget comes back with `finish_reason: "length"`, and a truncated document
under a grammar is never valid JSON. A smaller schema would not buy back the
budget, so `Chat` returns a `*llm.TruncatedError` telling you to raise
`MaxTokens` rather than burning two more requests to reach the same failure.

## Streaming

```go
ch, err := client.ChatStream(ctx, llm.ChatRequest{Messages: msgs})
for chunk := range ch {
    if chunk.Err != nil {
        return chunk.Err
    }
    if chunk.Done {
        log.Printf("%s, %d tokens", chunk.FinishReason, chunk.Usage.TotalTokens)
        break
    }
    fmt.Print(chunk.Delta)
}
```

The terminal chunk carries `FinishReason` and `Usage`, so a streaming caller
can tell a model that finished from one that hit `MaxTokens` — `"length"` means
the output is cut, which matters most when you asked for JSON, since a truncated
document never parses. Getting the token count requires asking for it, so
`ChatStream` sets `stream_options.include_usage`; backends that ignore the field
simply report zero.

That usage event arrives *after* the `finish_reason` event, so the reader keeps
scanning past it rather than stopping. Content deltas arriving after a
`finish_reason` are dropped: the completion is over at that point, and appending
to text the caller has already been told is finished would be worse than losing
it.

A backend that rejects `stream: true` transparently falls back to one
non-streaming request, whose whole response arrives as a single delta followed
by the usual terminal chunk. `ChatStream` applies no timeout of its own — bound
it with `ctx`.

## Context overflow

`OverflowRetry` retries a prompt that did not fit, using a shrink policy you
supply:

```go
client := llm.NewOverflowRetry(base, 4)

resp, err := client.Chat(ctx, llm.ChatRequest{
    Messages: msgs,
    OnOverflow: func(prev llm.ChatRequest, attempt int) (llm.ChatRequest, bool) {
        if len(prev.Messages) <= 2 {
            return prev, false // cannot shrink further
        }
        next := prev
        next.Messages = prev.Messages[1:] // drop the oldest turn
        return next, true
    },
})
```

Detecting "the prompt did not fit" is generic. Making a prompt smaller is not:
dropping retrieved passages, cutting low-similarity candidates and summarising
history are all correct answers in different applications, and only the caller
knows which. Requests without an `OnOverflow` hook pass through untouched.

Prefer sizing prompts correctly up front — see below.

## LM Studio

`lmstudio` reads LM Studio's native API, which reports things the
OpenAI-compatible `/v1/models` does not: model capabilities, and the context
length a loaded model was **actually** given.

That second number is the useful one. A model with a 131k-token architecture
loaded with an 8k window will accept an 8k prompt and reject a 9k one, and
`/v1/models` cannot tell you which it is.

```go
model, err := lmstudio.EnsureLoaded(ctx, lmstudio.PreflightOptions{
    Options: lmstudio.Options{BaseURL: "http://localhost:1234/v1"},
    Model:   "gemma-3-12b-it",
    WarmUp:  true,
})

budget := model.ContextLength()      // 8192 — what you have to fit inside
ceiling := model.MaxContextLength    // 131072 — what the architecture allows
tools := model.Capabilities.ToolUse
```

`EnsureLoaded` will not load a model unless you ask it to, since loading claims
VRAM that another workload may be using. Two ways to opt in:

- `UseCLI` shells out to `lms load`. Works only against a local daemon.
- `WarmUp` sends a minimal inference request, which makes LM Studio load the
  model on demand. Works remotely and inside containers, and costs one throwaway
  request. It matches the endpoint to the model type reported by the native API,
  so it is safe to use on embedding models.

The same `BaseURL` you use for inference works here; a trailing `/v1` is stripped.

## Embeddings

```go
client, err := embed.New(embed.Options{
    BaseURL:    "http://localhost:1234/v1",
    Model:      "bge-m3",
    Dimensions: 1024, // asserted against the first response
    Normalize:  true, // unit length, for cosine / inner-product search
})
vectors, err := client.Embed(ctx, []string{"first", "second"})
```

`Dimensions` catches the failure that is otherwise silent and expensive:
pointing at a different embedding model than the one your vector column was
provisioned for.

Batching is left to the caller — backends differ on how many inputs and tokens
they take per request, so the split belongs with the code that knows the model.

## Reranking

Cross-encoder reranking against a Text-Embeddings-Inference `/rerank` endpoint,
with a readiness probe for backends that take a while to load a model:

```go
client, err := rerank.New(rerank.Options{BaseURL: "http://localhost:8081"})

ctx, cancel := context.WithTimeout(ctx, 5*time.Minute)
defer cancel()
if _, err := rerank.WaitUntilReady(ctx, client, logger); err != nil {
    return err
}

results, err := client.Rerank(ctx, query, documents)
```

`WaitUntilReady` distinguishes "not ready yet" (503, connection refused) from
"never going to work" (401, 404) and gives up immediately on the second kind.

## Configuration

The library reads no environment variables and parses no config files.
Everything arrives through `Options`; mapping your own configuration onto it is
your job. That keeps the same client usable from a service reading `env` tags, a
CLI reading flags, and a test constructing options inline.

## Development

```bash
devbox run -- task lint
devbox run -- task test
```

## License

Apache 2.0
