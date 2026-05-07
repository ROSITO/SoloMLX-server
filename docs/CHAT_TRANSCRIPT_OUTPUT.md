# Chat prompt format and “transcript / roleplay” output

## Symptom

The assistant answer sometimes looked like a chat log: `assistant: …`, `user: …`, or a truncated prefix such as `Ass` at the end of the reply.

## Root cause

`/v1/chat/completions` used to build the model input like this:

```text
system: …
user: …
assistant: …
```

Instruction-tuned models (e.g. Qwen2.5 Instruct) are trained with the **tokenizer chat template** (special tokens and turn boundaries), not with raw `role: text` lines. Feeding an out-of-distribution prefix makes the model continue a **pseudo-dialogue** or echo role names—exactly what looked like “roleplay” in the UI.

The trailing `Ass` case is typical of a stream that was cut while the model was starting another `Assistant` token; filtering only at the start of the stream cannot fix that without also handling **end-of-string role fragments**.

## Fix (server)

1. **MLX path**: After the model is loaded, the prompt is built with `tokenizer.apply_chat_template(..., tokenize=False, add_generation_prompt=True)` when `has_chat_template` is true (see `MLXLMBackend.build_chat_prompt`).
2. **Stub path**: Still uses the simple `role: content` join so tests stay deterministic without pulling a tokenizer.
3. **Sanitization**: `_sanitize_completion_text` removes transcript markers such as `\nuser:` / `assistant:` and strips a **trailing incomplete `assistant` word** (e.g. ` Ass`). Markers **without** a colon after the role name (e.g. a naive `\nuser` match) are avoided so Python lines like `username = …` are not cut off mid-code.

## Client

The mini UI applies the same trailing-fragment rule in JavaScript so display stays consistent even if the API changes slightly.

## Operational note

If you call completion-style APIs with a **single user string** in another tool, prefer models’ chat templates there as well; MLXServe’s chat endpoint now does this automatically for MLX backends.

---

## Voir aussi (documentation projet)

- [README principal](../README.md) — installation, API, captures d’écran
- [Index docs](README.md) — OPERATIONS, reverse proxy, etc.
