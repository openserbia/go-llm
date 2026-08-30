package llm

import (
	"context"
	"encoding/json/v2"
	"fmt"
)

// repairPrompt asks for a correction with the decoder's own complaint
// attached. A model shown the specific reason its output failed does far
// better than one asked to try again.
const repairPrompt = "That response is not valid against the required schema: %s. " +
	"Reply with only the corrected JSON."

// repairMessages is how many messages the repair turn appends to the original
// conversation: the model's bad reply and the correction request.
const repairMessages = 2

// ChatAs sends req and decodes the reply into T.
//
// When req.ResponseFormat is nil it is derived from T with SchemaOf, so the
// type is the schema; a format the caller set is left alone. Decoding rejects
// unknown members, which makes it the client-side half of the
// additionalProperties:false that the derived schema declares — a backend that
// answered with fields the schema forbade is a failure, not something to
// silently discard.
//
// A decode failure is retried exactly once, with the bad output and the
// decoder's error appended to the conversation. One repair, not a loop: a
// model that fails the same schema twice is not going to be talked into it.
//
// It takes a Client rather than being a method, so it composes with
// OverflowRetry and anything else wrapping the client. (Go 1.27 allows generic
// methods, but not on an interface, so Client could not carry this either way.)
func ChatAs[T any](ctx context.Context, c Client, req ChatRequest) (T, error) {
	var zero T

	if req.ResponseFormat == nil {
		format, err := SchemaOf[T]()
		if err != nil {
			return zero, err
		}
		req.ResponseFormat = format
	}

	resp, err := c.Chat(ctx, req)
	if err != nil {
		return zero, err
	}

	var out T
	decodeErr := decodeStrict(resp.Content, &out)
	if decodeErr == nil {
		return out, nil
	}

	repair := req
	// Copy rather than append in place: req.Messages belongs to the caller and
	// may share an array with a slice they still hold.
	repair.Messages = make([]Message, 0, len(req.Messages)+repairMessages)
	repair.Messages = append(repair.Messages, req.Messages...)
	repair.Messages = append(repair.Messages,
		Assistant(resp.Content),
		User(fmt.Sprintf(repairPrompt, decodeErr)),
	)

	resp, err = c.Chat(ctx, repair)
	if err != nil {
		return zero, err
	}

	var repaired T
	if err := decodeStrict(resp.Content, &repaired); err != nil {
		return zero, fmt.Errorf("llm: ChatAs: response did not decode after one repair attempt: %w", err)
	}
	return repaired, nil
}

func decodeStrict(content string, out any) error {
	return json.Unmarshal([]byte(content), out, json.RejectUnknownMembers(true))
}
