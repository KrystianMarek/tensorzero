# Anthropic per-block `cache_control` preservation spike

**Ticket:** tensorzero-ick  
**Date:** 2026-04-21  
**Author:** Krystian Marek

---

## 1. Current state (verified)

**Ingress deserialization — present.**  
`crates/tensorzero-core/src/endpoints/anthropic_compatible/types/messages.rs` already deserializes per-block `cache_control` on every relevant variant:

- `AnthropicContentBlockOwned::Text { text, cache_control: Option<CacheControlOwned> }` — line 220
- `Image { cache_control, ... }` — line 223
- `Document { cache_control, ... }` — line 227
- `ToolUse { id, name, input, cache_control }` — line 234
- `ToolResult { cache_control, ... }` — line 240
- `AnthropicToolOwned { ..., cache_control }` — line 443

**Ingress translation — dropped.**  
`blocks_into_input_content()` (line 1080) discards every one of those fields with `..` rest patterns:

- `AnthropicContentBlockOwned::Text { text, .. }` → `InputMessageContent::Text(Text { text })` — line 1087
- `ToolUse { id, name, input, .. }` → drops `cache_control` — line 1118
- `ToolResult { .. }` → drops `cache_control` — line 1129
- `Image { source, .. }` → drops `cache_control` — line 1167
- `Document { source, .. }` → drops `cache_control` — line 1202

Tool-level `cache_control` is also dropped: `AnthropicToolOwned::cache_control` is ignored when building `crate::tool::FunctionTool` in `try_into_params()` at line 924.

**Internal types — no representation.**  
`tensorzero-types/src/message.rs` line 35: `InputMessageContent` variants (`Text`, `ToolCall`, `ToolResult`, `File`, `Thought`, `Unknown`, etc.) carry no `cache_control` field.  
`crates/tensorzero-inference-types/src/lib.rs` line 480: `ContentBlock` (the templated/provider-facing enum) has no `cache_control` field.  
`crates/tensorzero-core/src/providers/anthropic.rs` line 673: outbound `AnthropicMessageContent<'a>` has no `cache_control` field.  
Line 882: `AnthropicSystemBlock::Text { text }` has a code comment explicitly noting "cache control … we will ignore these for now."

**Current workaround — `extra_body`.**  
E2E tests inject cache markers via `extra_body` JSON pointers (`/system/0/cache_control`, etc.) which are merged into the serialized request body by `inject_extra_request_data_and_send()` in `providers/helpers.rs` (line 229). This works for native TensorZero API users but does not help Anthropic-SDK clients sending per-block markers.

---

## 2. Internal `Input` model shape

Neither `InputMessageContent` nor `ContentBlock` carries `Option<CacheControl>` today.

| Type | Crate | Has `cache_control`? |
|------|-------|----------------------|
| `InputMessageContent` | `tensorzero-types` | No |
| `ContentBlock` | `tensorzero-inference-types` | No |
| `RequestMessage` | `tensorzero-inference-types` | No |
| `Text` (struct) | `tensorzero-types` | No |
| `ToolCall` | `tensorzero-types` | No |
| `ToolResult` | `tensorzero-types` | No |

To support Approach A (extend internal model), we would need to add `Option<CacheControl>` to every `InputMessageContent` and `ContentBlock` variant that can carry it, or wrap the variants. Because `ContentBlock` is serialized (tagged enum) and used by every provider, adding a field changes the JSON schema and every provider match arm.

---

## 3. Blast radius

If we added `Option<CacheControl>` to `ContentBlock` variants, the following provider files pattern-match on `ContentBlock` and would need compile fixes:

| Provider | File | Would need update? | Note |
|----------|------|-------------------|------|
| Anthropic | `providers/anthropic.rs` | **Yes** | Must emit `cache_control` on outbound blocks. |
| GCP Vertex Anthropic | `providers/gcp_vertex_anthropic.rs` | **Yes** | Reuses `AnthropicMessage` / `AnthropicMessageContent` from `anthropic.rs`; same gap. |
| AWS Bedrock | `providers/aws_bedrock.rs` | **Yes** | Has its own `cachePoint` mechanism; would need mapping. |
| OpenAI | `providers/openai/mod.rs` | Compile fix only | OpenAI auto-caches; can ignore the field. |
| Azure | `providers/azure.rs` | Compile fix only | OpenAI-compatible; ignore. |
| Groq | `providers/groq.rs` | Compile fix only | OpenAI-compatible; ignore. |
| xAI | `providers/xai.rs` | Compile fix only | OpenAI-compatible; ignore. |
| Mistral | `providers/mistral.rs` | Compile fix only | OpenAI-compatible; ignore. |
| Fireworks | `providers/fireworks/mod.rs` | Compile fix only | OpenAI-compatible; ignore. |
| Together | `providers/together.rs` | Compile fix only | OpenAI-compatible; ignore. |
| DeepSeek | `providers/deepseek.rs` | Compile fix only | OpenAI-compatible; ignore. |
| OpenRouter | `providers/openrouter.rs` | Compile fix only | OpenAI-compatible; ignore. |
| vLLM | `providers/vllm.rs` | Compile fix only | OpenAI-compatible; ignore. |
| SGLang | `providers/sglang.rs` | Compile fix only | OpenAI-compatible; ignore. |
| Hyperbolic | `providers/hyperbolic.rs` | Compile fix only | OpenAI-compatible; ignore. |
| TGI | `providers/tgi.rs` | Compile fix only | OpenAI-compatible; ignore. |
| GCP Vertex Gemini | `providers/gcp_vertex_gemini/mod.rs` | Compile fix only | Gemini format; ignore. |
| Google AI Studio Gemini | `providers/google_ai_studio_gemini.rs` | Compile fix only | Gemini format; ignore. |
| Dummy | `providers/dummy.rs` | Compile fix only | Test provider; ignore. |

Additionally, `From<ContentBlockChatOutput> for ContentBlock` in `tensorzero-inference-types/src/lib.rs` (line 490) and all Python/TS bindings that expose `ContentBlock` would need updates.

**Headline:** 3 providers actually need new logic (Anthropic, GCP Vertex Anthropic, AWS Bedrock); ~16 others need trivial compile-only match arm updates.

---

## 4. Proposal

**Approach B — Side-channel.** Add a `cache_control_spans: Vec<CacheControlSpan>` field to `ModelInferenceRequest` (or `InferenceParams`), populated only by the Anthropic ingress. Each span records `(message_idx, content_idx, cache_control)`. The outbound Anthropic and GCP Vertex Anthropic providers read this side-channel when building `AnthropicMessageContent` and inject the marker onto the corresponding block. AWS Bedrock can map the same spans to its `cachePoint` format.

This is preferable to Approach A because it avoids refactoring the widely-shared `ContentBlock` and `Text` structs, which would cascade into Python bindings, TS schemas, and ~20 provider files. It is also preferable to Approach C because it preserves **full** per-block fidelity — Anthropic clients can place `cache_control` on any text, image, document, tool_use, or tool_result block — rather than artificially limiting support to a single marker. The side-channel is read only by providers that understand explicit cache markers; all other providers compile and run unchanged.

---

## 5. Ship gate

**This does not block Phase 3.** The Anthropic ingress is fully functional without per-block cache control; the only loss is that Anthropic-SDK clients who send `cache_control` on individual blocks will not see those blocks cached by the upstream provider. Native TensorZero users can already achieve prompt caching via `extra_body` JSON pointers (as the E2E suite demonstrates). We should ship Phase 3 behind the existing route flag and implement the side-channel in the follow-up ticket. The side-channel can be added without breaking any public API.
