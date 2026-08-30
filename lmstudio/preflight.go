package lmstudio

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os/exec"
	"strings"

	"github.com/openserbia/go-llm"
)

// PreflightOptions configures EnsureLoaded.
type PreflightOptions struct {
	Options

	// Model is the model key to make ready. Required.
	Model string

	// Logger receives progress for the slow paths — loading a model takes tens
	// of seconds and is otherwise indistinguishable from a hang. Defaults to
	// slog.Default().
	Logger *slog.Logger

	// UseCLI allows shelling out to the `lms` command to load a downloaded but
	// unloaded model. It has no effect when `lms` is not on PATH.
	//
	// Leave it off when the process shares a machine with something else that
	// manages LM Studio: loading a model claims VRAM, and an unexpected load
	// can evict a model another workload is mid-way through using. It is also
	// useless against a remote server, since the CLI drives the local daemon.
	UseCLI bool

	// WarmUp allows triggering a load by sending a minimal chat request, which
	// makes LM Studio load the model on demand. This works where UseCLI does
	// not — a remote server, or a container without the CLI — at the cost of
	// one throwaway completion.
	WarmUp bool
}

func (o PreflightOptions) logger() *slog.Logger {
	if o.Logger != nil {
		return o.Logger
	}
	return slog.Default()
}

// EnsureLoaded returns the requested model once it is loaded and ready to
// serve, or an error explaining what is missing.
//
// The returned Model carries the real load configuration, so
// Model.ContextLength reports the window the model actually has rather than
// the architectural maximum. Sizing prompts against it is the point of calling
// this at startup.
//
// Loading is attempted only through the paths you enable. With neither UseCLI
// nor WarmUp, a known-but-unloaded model is an error rather than something
// this silently loads for you.
func EnsureLoaded(ctx context.Context, opts PreflightOptions) (Model, error) {
	if opts.Model == "" {
		return Model{}, errors.New("lmstudio: preflight: Model is required")
	}
	client, err := New(opts.Options)
	if err != nil {
		return Model{}, err
	}

	model, err := client.Model(ctx, opts.Model)
	if err != nil {
		if errors.Is(err, ErrModelNotFound) {
			return Model{}, notFoundError(ctx, opts)
		}
		return Model{}, err
	}
	if model.Loaded() {
		return model, nil
	}

	log := opts.logger()
	log.Info("lmstudio: model is not loaded", slog.String("model", opts.Model))

	if err := load(ctx, opts, log); err != nil {
		return Model{}, err
	}

	model, err = client.Model(ctx, opts.Model)
	if err != nil {
		return Model{}, err
	}
	if !model.Loaded() {
		return Model{}, fmt.Errorf("lmstudio: model %q still reports no loaded instance after load", opts.Model)
	}
	log.Info("lmstudio: model loaded",
		slog.String("model", opts.Model),
		slog.Int("context_length", model.ContextLength()),
	)
	return model, nil
}

// load runs whichever load strategy the caller enabled.
func load(ctx context.Context, opts PreflightOptions, log *slog.Logger) error {
	if opts.UseCLI && cliAvailable() {
		log.Info("lmstudio: loading via lms CLI", slog.String("model", opts.Model))
		return cliLoad(ctx, opts.Model)
	}
	if opts.WarmUp {
		log.Info("lmstudio: loading via warm-up request", slog.String("model", opts.Model))
		return warmUp(ctx, opts)
	}
	return fmt.Errorf(
		"lmstudio: model %q is not loaded; load it in LM Studio, or enable UseCLI or WarmUp",
		opts.Model,
	)
}

// notFoundError distinguishes "not downloaded" from "downloaded but the server
// has not indexed it", which need different fixes, and says which one applies
// when the CLI is there to tell us.
func notFoundError(ctx context.Context, opts PreflightOptions) error {
	if !cliAvailable() {
		return fmt.Errorf(
			"lmstudio: model %q is not available at %s; download it with: lms get %q",
			opts.Model, ServerRoot(opts.BaseURL), opts.Model,
		)
	}
	downloaded, err := cliIsDownloaded(ctx, opts.Model)
	if err != nil {
		return fmt.Errorf("lmstudio: lms ls: %w", err)
	}
	if !downloaded {
		return fmt.Errorf("lmstudio: model %q is not downloaded; run: lms get %q -y", opts.Model, opts.Model)
	}
	return fmt.Errorf("lmstudio: model %q is downloaded but the server does not list it", opts.Model)
}

// warmUp sends the smallest possible completion. LM Studio loads a model on
// its first inference request, so the response is discarded — only the side
// effect matters.
func warmUp(ctx context.Context, opts PreflightOptions) error {
	client, err := llm.New(llm.Options{
		BaseURL: strings.TrimRight(ServerRoot(opts.BaseURL), "/") + "/v1",
		APIKey:  opts.APIKey,
		Model:   opts.Model,
	})
	if err != nil {
		return err
	}
	_, err = client.Chat(ctx, llm.ChatRequest{
		Messages:  []llm.Message{llm.User("hi")},
		MaxTokens: 1,
	})
	if err != nil {
		return fmt.Errorf("lmstudio: warm-up request failed: %w", err)
	}
	return nil
}

func cliAvailable() bool {
	_, err := exec.LookPath("lms")
	return err == nil
}

func cliIsDownloaded(ctx context.Context, model string) (bool, error) {
	out, err := exec.CommandContext(ctx, "lms", "ls").Output()
	if err != nil {
		return false, err
	}
	return listsIdentifier(string(out), model), nil
}

func cliLoad(ctx context.Context, model string) error {
	// The model name is configuration supplied by the operator, and is passed
	// as an argv element rather than through a shell.
	out, err := exec.CommandContext(ctx, "lms", "load", model).CombinedOutput() //nolint:gosec // argv, not a shell string
	if err != nil {
		return fmt.Errorf("lmstudio: lms load %q: %w: %s", model, err, strings.TrimSpace(string(out)))
	}
	return nil
}

// listsIdentifier scans the whitespace-aligned table `lms ls` prints for an
// exact identifier in the first column. Substring matching would confuse a
// model with one whose name it is a prefix of.
func listsIdentifier(table, model string) bool {
	for line := range strings.SplitSeq(table, "\n") {
		fields := strings.Fields(line)
		if len(fields) > 0 && fields[0] == model {
			return true
		}
	}
	return false
}
