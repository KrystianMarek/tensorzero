//! Streaming types and state machine for Anthropic-compatible ingress.
//!
//! Converts TensorZero's flat `ProviderInferenceResponseStream` chunks into
//! Anthropic's block-oriented SSE event sequence:
//!
//! ```text
#![expect(clippy::expect_used)]
//! message_start
//! content_block_start (one per block, monotonic index)
//! content_block_delta (zero or more inside a block)
//! content_block_stop (one per block)
//! message_delta (stop_reason + final usage)
//! message_stop
//! ping (periodic keepalive)
//! ```

use axum::response::sse::{Event, Sse};
use futures::Stream;
use serde::Serialize;
use std::collections::HashMap;
use std::pin::Pin;
use std::task::{Context, Poll};

use crate::endpoints::inference::{InferenceResponseChunk, InferenceStream};
use crate::error::Error;
use crate::inference::types::ContentBlockChunk;
use tensorzero_types::Usage;
use tensorzero_types_providers::anthropic::AnthropicStopReason;

// =============================================================================
// Stream message types (Anthropic SSE events)
// =============================================================================

/// A single event emitted during a streaming response.
#[derive(Clone, Debug, Serialize)]
pub enum AnthropicStreamMessage {
    MessageStart {
        id: String,
        #[serde(rename = "type")]
        response_type: String,
        model: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        stop_sequence: Option<i32>,
        usage: AnthropicStreamUsage,
    },
    ContentBlockStart {
        index: usize,
        #[serde(rename = "type")]
        content_type: String,
        id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        thinking: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    ContentBlockDelta {
        index: usize,
        #[serde(rename = "type")]
        delta_type: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        text: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        tool_use_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        input: Option<serde_json::Value>,
    },
    ContentBlockStop {
        index: usize,
    },
    MessageDelta {
        #[serde(rename = "type")]
        delta_type: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        stop_reason: Option<AnthropicStopReason>,
        #[serde(skip_serializing_if = "Option::is_none")]
        stop_sequence: Option<i32>,
        usage: AnthropicStreamUsage,
    },
    MessageStop,
}

/// Usage data in Anthropic stream events.
#[derive(Clone, Debug, Default, Serialize)]
pub struct AnthropicStreamUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_input_tokens: Option<u32>,
}

impl From<&Usage> for AnthropicStreamUsage {
    fn from(usage: &Usage) -> Self {
        AnthropicStreamUsage {
            input_tokens: usage.input_tokens.unwrap_or(0),
            output_tokens: usage.output_tokens.unwrap_or(0),
            cache_creation_input_tokens: usage.provider_cache_write_input_tokens,
            cache_read_input_tokens: usage.provider_cache_read_input_tokens,
        }
    }
}

impl From<Usage> for AnthropicStreamUsage {
    fn from(usage: Usage) -> Self {
        (&usage).into()
    }
}

impl AnthropicStreamMessage {
    /// Serialize to an SSE Event with the correct `event:` field.
    #[expect(clippy::wrong_self_convention, clippy::unnecessary_wraps)]
    fn to_sse_event(self, _id: &str, event_id: usize) -> Result<Event, serde_json::Error> {
        let event_name = match &self {
            AnthropicStreamMessage::MessageStart { .. } => "message_start",
            AnthropicStreamMessage::ContentBlockStart { .. } => "content_block_start",
            AnthropicStreamMessage::ContentBlockDelta { .. } => "content_block_delta",
            AnthropicStreamMessage::ContentBlockStop { .. } => "content_block_stop",
            AnthropicStreamMessage::MessageDelta { .. } => "message_delta",
            AnthropicStreamMessage::MessageStop => "message_stop",
        };
        let json = match self {
            AnthropicStreamMessage::MessageStart {
                id,
                response_type,
                model,
                stop_sequence,
                usage,
            } => serde_json::json!({
                "id": id,
                "type": response_type,
                "model": model,
                "stop_sequence": stop_sequence,
                "usage": usage,
            }),
            AnthropicStreamMessage::ContentBlockStart {
                index,
                content_type,
                id,
                thinking,
                signature,
                name,
            } => serde_json::json!({
                "index": index,
                "type": content_type,
                "id": id,
                "thinking": thinking,
                "signature": signature,
                "name": name,
            }),
            AnthropicStreamMessage::ContentBlockDelta {
                index,
                delta_type,
                text,
                tool_use_id,
                name,
                input,
            } => serde_json::json!({
                "index": index,
                "type": delta_type,
                "text": text,
                "tool_use_id": tool_use_id,
                "name": name,
                "input": input,
            }),
            AnthropicStreamMessage::ContentBlockStop { index } => {
                serde_json::json!({"index": index})
            }
            AnthropicStreamMessage::MessageDelta {
                delta_type,
                stop_reason,
                stop_sequence,
                usage,
            } => serde_json::json!({
                "type": delta_type,
                "stop_reason": stop_reason,
                "stop_sequence": stop_sequence,
                "usage": usage,
            }),
            AnthropicStreamMessage::MessageStop => {
                serde_json::json!({})
            }
        };
        Ok(Event::default()
            .event(event_name)
            .id(event_id.to_string())
            .data(json.to_string()))
    }
}

// =============================================================================
// State machine
// =============================================================================

/// Tracks which block kind we are currently inside.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
enum BlockKind {
    #[default]
    None,
    Text,
    ToolCall,
    Thinking,
    RedactedThinking,
}

impl BlockKind {
    fn from_chunk(chunk: &ContentBlockChunk) -> BlockKind {
        match chunk {
            ContentBlockChunk::Text(_) => BlockKind::Text,
            ContentBlockChunk::ToolCall(_) => BlockKind::ToolCall,
            ContentBlockChunk::Thought(_) => BlockKind::Thinking,
            ContentBlockChunk::Unknown(_) => BlockKind::RedactedThinking,
        }
    }
}

/// State machine that restructures flat chunk streams into Anthropic's block-oriented events.
#[derive(Clone, Debug, Default)]
pub struct AnthropicStreamState {
    current_block_kind: BlockKind,
    /// Monotonically increasing block index across all emitted blocks.
    current_block_index: usize,
    /// Mapping from tool call ID to block index (for tool_use).
    tool_id_to_index: HashMap<String, usize>,
    /// Total number of distinct blocks emitted so far.
    total_blocks: usize,
    /// Aggregated usage accumulated across all chunks.
    aggregated_usage: Usage,
    /// Signature from the Thought chunk (emitted with content_block_stop for thinking blocks).
    thinking_signature: Option<String>,
    /// Finish reason from the last processed chunk.
    last_finish_reason: Option<tensorzero_types::FinishReason>,
}

impl AnthropicStreamState {
    /// Emits the trailing message_delta and message_stop events once the stream is complete.
    pub fn finalize(self) -> Vec<AnthropicStreamMessage> {
        let stop = finish_reason_to_anthropic_stop_reason(self.last_finish_reason);
        vec![
            AnthropicStreamMessage::MessageDelta {
                delta_type: "message_delta".to_string(),
                stop_reason: stop,
                stop_sequence: None,
                usage: self.aggregated_usage.into(),
            },
            AnthropicStreamMessage::MessageStop,
        ]
    }

    /// Processes a single chunk and returns the events to emit.
    pub fn process_chunk(
        &mut self,
        chunk: &InferenceResponseChunk,
        inference_id: &str,
        model: &str,
    ) -> Vec<AnthropicStreamMessage> {
        // Aggregate usage.
        if let Some(usage) = chunk.usage() {
            accumulate_usage(&mut self.aggregated_usage, usage);
        }
        // Track finish reason from the last chunk.
        if let Some(reason) = chunk.finish_reason() {
            self.last_finish_reason = Some(*reason);
        }

        let mut events = Vec::new();

        // Emit message_start on the very first chunk.
        if events.is_empty() {
            events.push(AnthropicStreamMessage::MessageStart {
                id: format!("msg_{inference_id}"),
                response_type: "message".to_string(),
                model: model.to_string(),
                stop_sequence: None,
                usage: self.aggregated_usage.into(),
            });
        }

        // Process each content block in the chunk.
        for block in chunk.content_blocks() {
            events.extend(self.ensure_block(block, inference_id, model));
        }

        events
    }

    /// Ensures the correct block is open for the given content block chunk.
    /// Emits block start/stop/delta events as needed.
    fn ensure_block(
        &mut self,
        block: &ContentBlockChunk,
        inference_id: &str,
        _model: &str,
    ) -> Vec<AnthropicStreamMessage> {
        let mut events = Vec::new();
        let new_kind = BlockKind::from_chunk(block);

        let kind_mismatch = match (self.current_block_kind, new_kind) {
            (BlockKind::None, _) => true,
            (old, new) => old != new,
        };

        // Close the current block if it's different from the new one.
        if kind_mismatch && !matches!(self.current_block_kind, BlockKind::None) {
            events.push(AnthropicStreamMessage::ContentBlockStop {
                index: self.current_block_index,
            });
        }

        // Open a new block if this is the first chunk for this block kind.
        if matches!(self.current_block_kind, BlockKind::None) {
            let block_id = block_id_from_chunk(block, inference_id);
            let index = self.total_blocks;
            self.total_blocks += 1;

            match new_kind {
                BlockKind::Thinking => {
                    self.current_block_kind = BlockKind::Thinking;
                    let (thinking, thinking_sig) = extract_thought_fields(block);
                    self.thinking_signature.clone_from(&thinking_sig);
                    events.push(AnthropicStreamMessage::ContentBlockStart {
                        index,
                        content_type: "thinking".to_string(),
                        id: block_id,
                        thinking,
                        signature: thinking_sig,
                        name: None,
                    });
                }
                BlockKind::RedactedThinking => {
                    self.current_block_kind = BlockKind::RedactedThinking;
                    let data = extract_unknown_data(block);
                    events.push(AnthropicStreamMessage::ContentBlockStart {
                        index,
                        content_type: "redacted_thinking".to_string(),
                        id: block_id,
                        thinking: Some(data),
                        signature: None,
                        name: None,
                    });
                }
                BlockKind::Text => {
                    self.current_block_kind = BlockKind::Text;
                    events.push(AnthropicStreamMessage::ContentBlockStart {
                        index,
                        content_type: "text".to_string(),
                        id: block_id,
                        thinking: None,
                        signature: None,
                        name: None,
                    });
                }
                BlockKind::ToolCall => {
                    self.current_block_kind = BlockKind::ToolCall;
                    let (id, name, input) = extract_toolcall_fields(block);
                    events.push(AnthropicStreamMessage::ContentBlockStart {
                        index,
                        content_type: "tool_use".to_string(),
                        id: id.clone(),
                        thinking: None,
                        signature: None,
                        name: Some(name.clone()),
                    });
                    self.tool_id_to_index.insert(id, index);
                    events.push(AnthropicStreamMessage::ContentBlockDelta {
                        index,
                        delta_type: "input_json_delta".to_string(),
                        text: None,
                        tool_use_id: None,
                        name: None,
                        input,
                    });
                    return events;
                }
                BlockKind::None => {}
            }
        }

        // Emit delta for the current open block.
        match new_kind {
            BlockKind::Thinking => {
                self.current_block_kind = BlockKind::Thinking;
                let (thinking, signature) = extract_thought_fields(block);
                if let Some(sig) = signature {
                    self.thinking_signature = Some(sig);
                }
                events.push(AnthropicStreamMessage::ContentBlockDelta {
                    index: self.current_block_index,
                    delta_type: "thinking_delta".to_string(),
                    text: thinking,
                    tool_use_id: None,
                    name: None,
                    input: None,
                });
            }
            BlockKind::RedactedThinking => {
                self.current_block_kind = BlockKind::RedactedThinking;
                let data = extract_unknown_data(block);
                events.push(AnthropicStreamMessage::ContentBlockDelta {
                    index: self.current_block_index,
                    delta_type: "thinking_delta".to_string(),
                    text: Some(data),
                    tool_use_id: None,
                    name: None,
                    input: None,
                });
            }
            BlockKind::Text => {
                self.current_block_kind = BlockKind::Text;
                let text = extract_text(block);
                events.push(AnthropicStreamMessage::ContentBlockDelta {
                    index: self.current_block_index,
                    delta_type: "text_delta".to_string(),
                    text,
                    tool_use_id: None,
                    name: None,
                    input: None,
                });
            }
            BlockKind::ToolCall => {
                self.current_block_kind = BlockKind::ToolCall;
                let (id, name, input) = extract_toolcall_fields(block);
                let index = match self.tool_id_to_index.get(&id) {
                    Some(&idx) => idx,
                    None => {
                        let idx = self.total_blocks;
                        self.total_blocks += 1;
                        self.current_block_index = idx;
                        self.tool_id_to_index.insert(id.clone(), idx);
                        idx
                    }
                };
                self.current_block_index = index;
                events.push(AnthropicStreamMessage::ContentBlockDelta {
                    index,
                    delta_type: "input_json_delta".to_string(),
                    text: None,
                    tool_use_id: None,
                    name: Some(name),
                    input,
                });
            }
            BlockKind::None => {}
        }

        events
    }
}

/// Accumulate usage from a new chunk into the accumulated usage.
fn accumulate_usage(accum: &mut Usage, other: &Usage) {
    accum.input_tokens = match (accum.input_tokens, other.input_tokens) {
        (Some(a), Some(b)) => Some(a + b),
        (a, b) => a.or(b),
    };
    accum.output_tokens = match (accum.output_tokens, other.output_tokens) {
        (Some(a), Some(b)) => Some(a + b),
        (a, b) => a.or(b),
    };
    accum.provider_cache_read_input_tokens = match (
        accum.provider_cache_read_input_tokens,
        other.provider_cache_read_input_tokens,
    ) {
        (Some(a), Some(b)) => Some(a + b),
        (a, b) => a.or(b),
    };
    accum.provider_cache_write_input_tokens = match (
        accum.provider_cache_write_input_tokens,
        other.provider_cache_write_input_tokens,
    ) {
        (Some(a), Some(b)) => Some(a + b),
        (a, b) => a.or(b),
    };
    // cost is not accumulated — keep the first non-None value.
    if accum.cost.is_none() {
        accum.cost = other.cost;
    }
}

/// Convert internal `FinishReason` to Anthropic `stop_reason`.
pub fn finish_reason_to_anthropic_stop_reason(
    reason: Option<tensorzero_types::FinishReason>,
) -> Option<AnthropicStopReason> {
    match reason {
        None => None,
        Some(tensorzero_types::FinishReason::Stop) => Some(AnthropicStopReason::EndTurn),
        Some(tensorzero_types::FinishReason::Length) => Some(AnthropicStopReason::MaxTokens),
        Some(tensorzero_types::FinishReason::StopSequence) => {
            Some(AnthropicStopReason::StopSequence)
        }
        Some(tensorzero_types::FinishReason::ToolCall) => Some(AnthropicStopReason::ToolUse),
        Some(tensorzero_types::FinishReason::ContentFilter)
        | Some(tensorzero_types::FinishReason::Unknown) => None,
    }
}

// =============================================================================
// Chunk extraction helpers
// =============================================================================

fn block_id_from_chunk(chunk: &ContentBlockChunk, inference_id: &str) -> String {
    match chunk {
        ContentBlockChunk::Text(t) => format!("{inference_id}-text-{}", t.id),
        ContentBlockChunk::ToolCall(t) => format!("{inference_id}-tool_use-{}", t.id),
        ContentBlockChunk::Thought(t) => format!("{inference_id}-thinking-{}", t.id),
        ContentBlockChunk::Unknown(t) => format!("{inference_id}-thinking-{}", t.id),
    }
}

fn extract_text(chunk: &ContentBlockChunk) -> Option<String> {
    match chunk {
        ContentBlockChunk::Text(t) => Some(t.text.clone()),
        _ => None,
    }
}

fn extract_thought_fields(chunk: &ContentBlockChunk) -> (Option<String>, Option<String>) {
    match chunk {
        ContentBlockChunk::Thought(t) => (
            t.text.clone().filter(|s| !s.is_empty()),
            t.signature.clone().filter(|s| !s.is_empty()),
        ),
        _ => (None, None),
    }
}

fn extract_unknown_data(chunk: &ContentBlockChunk) -> String {
    match chunk {
        ContentBlockChunk::Unknown(u) => match &u.data {
            serde_json::Value::String(s) => s.clone(),
            _ => u.data.to_string(),
        },
        _ => String::new(),
    }
}

fn extract_toolcall_fields(
    chunk: &ContentBlockChunk,
) -> (String, String, Option<serde_json::Value>) {
    match chunk {
        ContentBlockChunk::ToolCall(t) => (
            t.id.clone(),
            t.raw_name.clone().unwrap_or_default(),
            Some(serde_json::from_str(&t.raw_arguments).unwrap_or_else(|_| serde_json::json!({}))),
        ),
        _ => (String::new(), String::new(), None),
    }
}

// =============================================================================
// InferenceResponseChunk helpers
// =============================================================================

impl InferenceResponseChunk {
    fn content_blocks(&self) -> &[ContentBlockChunk] {
        match self {
            InferenceResponseChunk::Chat(c) => &c.content,
            InferenceResponseChunk::Json(_) => &[],
        }
    }

    fn usage(&self) -> Option<&Usage> {
        match self {
            InferenceResponseChunk::Chat(c) => c.usage.as_ref(),
            InferenceResponseChunk::Json(j) => j.usage.as_ref(),
        }
    }

    fn finish_reason(&self) -> Option<&tensorzero_types::FinishReason> {
        match self {
            InferenceResponseChunk::Chat(c) => c.finish_reason.as_ref(),
            InferenceResponseChunk::Json(j) => j.finish_reason.as_ref(),
        }
    }
}

// =============================================================================
// Stream conversion: InferenceStream → Sse
// =============================================================================

/// Converts an internal `InferenceStream` into an Anthropic-compatible SSE stream.
pub fn convert_to_anthropic_stream(
    stream: InferenceStream,
    inference_id: uuid::Uuid,
    model: String,
) -> Sse<impl Stream<Item = Result<Event, Error>>> {
    let inner = AnthropicStreamAdapter {
        inner: stream,
        state: AnthropicStreamState::default(),
        inference_id_str: inference_id.to_string(),
        model,
        is_first_chunk: true,
        event_id: 0,
        buffer: Vec::new(),
        json_text_id: None,
        finalized: false,
    };
    Sse::new(inner)
}

/// A stateful stream adapter that converts `InferenceStream` to Anthropic SSE events.
struct AnthropicStreamAdapter {
    inner: InferenceStream,
    state: AnthropicStreamState,
    inference_id_str: String,
    model: String,
    is_first_chunk: bool,
    event_id: usize,
    buffer: Vec<Result<Event, Error>>,
    json_text_id: Option<usize>,
    finalized: bool,
}

impl Stream for AnthropicStreamAdapter {
    type Item = Result<Event, Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // First, drain the buffer of pre-computed events.
        if let Some(event) = self.buffer.pop() {
            return Poll::Ready(Some(event));
        }

        // Check if we need to finalize (stream ended).
        if self.finalized {
            let events = self.state.clone().finalize();
            for event in events.into_iter().rev() {
                let result = event
                    .to_sse_event(&self.inference_id_str, self.event_id)
                    .map_err(|e| {
                        Error::new(crate::error::ErrorDetails::Inference {
                            message: format!("Failed to serialize stream event: {e}"),
                        })
                    });
                self.event_id += 1;
                if result.is_ok() {
                    self.buffer.push(result);
                }
            }
            if !self.buffer.is_empty() {
                return Poll::Ready(Some(self.buffer.pop().expect("buffer not empty")));
            }
            return Poll::Ready(None);
        }

        // Poll the inner stream for the next chunk.
        let chunk = match Pin::new(&mut self.inner).poll_next(cx) {
            Poll::Ready(Some(Ok(chunk))) => chunk,
            Poll::Ready(Some(Err(e))) => {
                // Stream error — emit error event and finalize.
                self.finalized = true;
                return Poll::Ready(Some(Ok(Event::default()
                    .event("error")
                    .id(self.event_id.to_string())
                    .data(
                        serde_json::json!({
                            "type": "error",
                            "error": {
                                "type": "api_error",
                                "message": e.to_string()
                            }
                        })
                        .to_string(),
                    ))));
            }
            Poll::Ready(None) => {
                // Stream ended — finalize.
                self.finalized = true;
                return self.poll_next(cx);
            }
            Poll::Pending => return Poll::Pending,
        };

        let is_json = matches!(&chunk, InferenceResponseChunk::Json(_));

        // Handle JSON chunks specially — no content blocks.
        if is_json {
            let inference_id = self.inference_id_str.clone();
            if self.is_first_chunk {
                // Emit message_start for JSON.
                let msg = AnthropicStreamMessage::MessageStart {
                    id: format!("msg_{inference_id}"),
                    response_type: "message".to_string(),
                    model: self.model.clone(),
                    stop_sequence: None,
                    usage: Default::default(),
                };
                if let Ok(event) = msg.to_sse_event(&inference_id, self.event_id) {
                    self.event_id += 1;
                    self.buffer.push(Ok(event));
                }
                self.is_first_chunk = false;
            }

            // Emit content block start if not already done.
            let text_block_id = self.json_text_id.unwrap_or(0);
            if self.json_text_id.is_none() {
                let ev_id = self.event_id;
                self.event_id += 1;
                let event = AnthropicStreamMessage::ContentBlockStart {
                    index: 0,
                    content_type: "text".to_string(),
                    id: format!("{inference_id}-text-0"),
                    thinking: None,
                    signature: None,
                    name: None,
                }
                .to_sse_event(&inference_id, ev_id)
                .map_err(|e| {
                    Error::new(crate::error::ErrorDetails::Inference {
                        message: format!("Failed to serialize stream event: {e}"),
                    })
                });
                self.buffer.push(event);
                self.json_text_id = Some(text_block_id);
            }

            // Emit delta with the raw JSON content.
            if let InferenceResponseChunk::Json(j) = &chunk {
                let ev_id = self.event_id;
                self.event_id += 1;
                let event = AnthropicStreamMessage::ContentBlockDelta {
                    index: 0,
                    delta_type: "text_delta".to_string(),
                    text: Some(j.raw.clone()),
                    tool_use_id: None,
                    name: None,
                    input: None,
                }
                .to_sse_event(&inference_id, ev_id)
                .map_err(|e| {
                    Error::new(crate::error::ErrorDetails::Inference {
                        message: format!("Failed to serialize stream event: {e}"),
                    })
                });
                self.buffer.push(event);
            }

            // Aggregate usage and finish reason.
            if let Some(usage) = chunk.usage() {
                accumulate_usage(&mut self.state.aggregated_usage, usage);
            }
            if let Some(reason) = chunk.finish_reason() {
                self.state.last_finish_reason = Some(*reason);
            }

            // Check buffer again.
            if !self.buffer.is_empty() {
                return Poll::Ready(Some(self.buffer.pop().expect("buffer not empty")));
            }
        }

        // Handle chat chunks via state machine.
        if self.is_first_chunk {
            self.is_first_chunk = false;
        }

        let inference_id_str = self.inference_id_str.clone();
        let model = self.model.clone();
        let events = self.state.process_chunk(&chunk, &inference_id_str, &model);

        // Push events in reverse order so they're yielded in sequence.
        for event in events.into_iter().rev() {
            let ev_id = self.event_id;
            self.event_id += 1;
            let result = event.to_sse_event(&inference_id_str, ev_id).map_err(|e| {
                Error::new(crate::error::ErrorDetails::Inference {
                    message: format!("Failed to serialize stream event: {e}"),
                })
            });
            self.event_id += 1;
            if result.is_ok() {
                self.buffer.push(result);
            }
        }

        // Check buffer again.
        if self.buffer.is_empty() {
            // No events emitted. This shouldn't happen for chat chunks, but poll again to be safe.
            self.poll_next(cx)
        } else {
            Poll::Ready(Some(self.buffer.pop().expect("buffer not empty")))
        }
    }
}
