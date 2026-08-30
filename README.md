# go-llm

A Go client for OpenAI-compatible inference endpoints, written against what
self-hosted backends actually implement.

LM Studio, Ollama, vLLM, llama.cpp and text-generation-inference all claim to
speak the OpenAI API, and all of them are missing a different part of it. This
library detects those gaps from the backend's own error responses and degrades
instead of failing, so the calling code does not have to know which backend it
is pointed at.

Zero dependencies outside the standard library.

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

## Streaming

```go
ch, err := client.ChatStream(ctx, llm.ChatRequest{Messages: msgs})
for chunk := range ch {
    if chunk.Err != nil {
        return chunk.Err
    }
    fmt.Print(chunk.Delta)
}
```

A backend that rejects `stream: true` transparently falls back to one
non-streaming request delivered as a single chunk. `ChatStream` applies no
timeout of its own — bound it with `ctx`.

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
- `WarmUp` sends a one-token completion, which makes LM Studio load the model on
  demand. Works remotely and inside containers, and costs one throwaway request.

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
