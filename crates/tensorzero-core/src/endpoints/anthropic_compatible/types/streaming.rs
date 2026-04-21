#![expect(clippy::expect_used)]

//! Streaming types and state machine for Anthropic-compatible ingress.
//!
//! Converts TensorZero's flat `ProviderInferenceResponseStream` chunks into
//! Anthropic's block-oriented SSE event sequence:
//!
//! ```text
//! message_start
//! content_block_start (one per block, monotonic index)
//! content_block_delta (zero or more inside a block)
//! content_block_stop (one per block)
//! message_delta (stop_reason + final usage)
//! message_stop
//! ping (periodic keepalive)
//! ```

use axum::response::sse::Event;
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
    /// Key identifying the current open block (to handle parallel tool calls, multi-turn, etc.).
    /// `None` means text/thought/unknown blocks (a single global block per kind).
    current_block_key: Option<String>,
    /// Mapping from tool call ID to block index (for tool_use).
    tool_id_to_index: HashMap<String, usize>,
    /// Total number of distinct blocks emitted so far.
    total_blocks: usize,
    /// Counter for generating unique block IDs.
    block_id_counter: usize,
    /// Aggregated usage accumulated across all chunks.
    aggregated_usage: Usage,
    /// Signature from the Thought chunk (emitted with content_block_stop for thinking blocks).
    thinking_signature: Option<String>,
    /// Finish reason from the last processed chunk.
    last_finish_reason: Option<tensorzero_types::FinishReason>,
    /// Whether message_start has already been emitted.
    started: bool,
    /// Whether we currently have an open block.
    has_active_block: bool,
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
        if !self.started {
            self.started = true;
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

    /// Returns the block key for a given chunk.
    ///
    /// For tool calls, the key includes the tool ID so that parallel tool calls
    /// in the same response each get their own block. For text/thought/unknown
    /// blocks, we return None — these are tracked as a single global block per
    /// message (new blocks of the same kind just replace the previous one).
    fn block_key_for_chunk(chunk: &ContentBlockChunk) -> Option<String> {
        match chunk {
            ContentBlockChunk::ToolCall(t) => Some(format!("tool_{}", t.id)),
            ContentBlockChunk::Text(_) => None,
            ContentBlockChunk::Thought(_) => None,
            ContentBlockChunk::Unknown(_) => None,
        }
    }

    /// Ensures the correct block is open for the given content block chunk.
    /// Emits block start/stop/delta events as needed.
    fn ensure_block(
        &mut self,
        block: &ContentBlockChunk,
        _inference_id: &str,
        _model: &str,
    ) -> Vec<AnthropicStreamMessage> {
        let mut events = Vec::new();
        let new_kind = BlockKind::from_chunk(block);
        let new_key = Self::block_key_for_chunk(block);

        // Close the current block if the kind or key differs.
        // For non-tool blocks (None keys), only kind comparison matters.
        // For tool blocks, both kind and key must match to be the same block.
        let key_mismatch = match (&new_key, &self.current_block_key) {
            (Some(new_k), Some(cur_k)) => new_k != cur_k,
            _ => false, // Non-tool blocks only use kind comparison.
        };
        if self.has_active_block && (new_kind != self.current_block_kind || key_mismatch) {
            events.push(AnthropicStreamMessage::ContentBlockStop {
                index: self.current_block_index,
            });
            self.current_block_kind = BlockKind::None;
            self.has_active_block = false;
        }

        // Open a new block.
        if !self.has_active_block {
            match new_kind {
                BlockKind::Thinking => {
                    let (thinking, thinking_sig) = extract_thought_fields(block);
                    self.current_block_key = None;
                    self.thinking_signature.clone_from(&thinking_sig);
                    let idx = self.total_blocks;
                    self.total_blocks += 1;
                    let id = format!("block_{}", self.block_id_counter);
                    self.block_id_counter += 1;
                    self.current_block_index = idx;
                    self.current_block_kind = BlockKind::Thinking;
                    self.has_active_block = true;
                    events.push(AnthropicStreamMessage::ContentBlockStart {
                        index: idx,
                        content_type: "thinking".to_string(),
                        id,
                        thinking,
                        signature: thinking_sig,
                        name: None,
                    });
                }
                BlockKind::RedactedThinking => {
                    let data = extract_unknown_data(block);
                    self.current_block_key = None;
                    let idx = self.total_blocks;
                    self.total_blocks += 1;
                    let id = format!("block_{}", self.block_id_counter);
                    self.block_id_counter += 1;
                    self.current_block_index = idx;
                    self.current_block_kind = BlockKind::RedactedThinking;
                    self.has_active_block = true;
                    events.push(AnthropicStreamMessage::ContentBlockStart {
                        index: idx,
                        content_type: "redacted_thinking".to_string(),
                        id,
                        thinking: Some(data),
                        signature: None,
                        name: None,
                    });
                }
                BlockKind::Text => {
                    self.current_block_key = None;
                    let idx = self.total_blocks;
                    self.total_blocks += 1;
                    let id = format!("block_{}", self.block_id_counter);
                    self.block_id_counter += 1;
                    self.current_block_index = idx;
                    self.current_block_kind = BlockKind::Text;
                    self.has_active_block = true;
                    events.push(AnthropicStreamMessage::ContentBlockStart {
                        index: idx,
                        content_type: "text".to_string(),
                        id,
                        thinking: None,
                        signature: None,
                        name: None,
                    });
                }
                BlockKind::ToolCall => {
                    self.current_block_key = new_key;
                    let (id, name, input) = extract_toolcall_fields(block);
                    // Reuse existing index if the same tool_id was seen before,
                    // otherwise allocate a new one.
                    let idx = match self.tool_id_to_index.get(&id) {
                        Some(&i) => i,
                        None => {
                            let i = self.total_blocks;
                            self.total_blocks += 1;
                            self.tool_id_to_index.insert(id.clone(), i);
                            i
                        }
                    };
                    self.current_block_index = idx;
                    self.current_block_kind = BlockKind::ToolCall;
                    self.has_active_block = true;
                    let id = format!("block_{}", self.block_id_counter);
                    self.block_id_counter += 1;
                    events.push(AnthropicStreamMessage::ContentBlockStart {
                        index: idx,
                        content_type: "tool_use".to_string(),
                        id,
                        thinking: None,
                        signature: None,
                        name: Some(name.clone()),
                    });
                    // Tool call: emit start + delta together.
                    events.push(AnthropicStreamMessage::ContentBlockDelta {
                        index: idx,
                        delta_type: "input_json_delta".to_string(),
                        text: None,
                        tool_use_id: None,
                        name: Some(name),
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
    model: String,
) -> impl Stream<Item = Result<Event, Error>> {
    AnthropicStreamAdapter {
        inner: stream,
        state: AnthropicStreamState::default(),
        inference_id_str: String::new(),
        model,
        is_first_chunk: true,
        event_id: 0,
        buffer: Vec::new(),
        json_text_id: None,
        finalized: false,
    }
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

        // Extract inference_id from first chunk if not yet set.
        if self.is_first_chunk
            && self.inference_id_str.is_empty()
            && let InferenceResponseChunk::Chat(chat) = &chunk
        {
            self.inference_id_str = chat.inference_id.to_string();
        }

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

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use uuid::Uuid;

    use super::*;
    use crate::endpoints::inference::ChatInferenceResponseChunk;
    use crate::inference::types::streams::{ThoughtChunk, UnknownChunk};
    use crate::inference::types::usage::Usage;
    use crate::inference::types::{FinishReason, TextChunk};
    use crate::tool::ToolCallChunk;
    use googletest::prelude::*;
    use tensorzero_types_providers::anthropic::AnthropicStopReason;

    fn make_chat_chunk(
        inference_id: Uuid,
        episode_id: Uuid,
        content: Vec<ContentBlockChunk>,
        usage: Option<Usage>,
        finish_reason: Option<FinishReason>,
    ) -> InferenceResponseChunk {
        InferenceResponseChunk::Chat(ChatInferenceResponseChunk {
            inference_id,
            episode_id,
            variant_name: "test".to_string(),
            content,
            usage,
            raw_usage: None,
            raw_response: None,
            finish_reason,
            original_chunk: None,
            raw_chunk: None,
            aggregated_response: None,
        })
    }

    fn assert_event_type(events: &[AnthropicStreamMessage], idx: usize, expected_type: &str) {
        assert_json_type(events, idx, expected_type);
    }

    fn assert_json_type(
        events: &[AnthropicStreamMessage],
        idx: usize,
        expected_type: &str,
    ) -> serde_json::Value {
        let event = &events[idx];
        // Serialize to json via the same mechanism used in to_sse_event
        match event {
            AnthropicStreamMessage::MessageStart {
                id,
                response_type,
                model,
                stop_sequence,
                usage,
            } => {
                let json = serde_json::json!({
                    "id": id,
                    "type": response_type,
                    "model": model,
                    "stop_sequence": stop_sequence,
                    "usage": usage,
                });
                assert_eq!(
                    response_type, expected_type,
                    "event type mismatch at index {idx}"
                );
                json
            }
            AnthropicStreamMessage::ContentBlockStart {
                index,
                content_type,
                id,
                thinking,
                signature,
                name,
            } => {
                let json = serde_json::json!({
                    "index": index,
                    "type": content_type,
                    "id": id,
                    "thinking": thinking,
                    "signature": signature,
                    "name": name,
                });
                assert_eq!(
                    content_type, expected_type,
                    "event type mismatch at index {idx}"
                );
                json
            }
            AnthropicStreamMessage::ContentBlockDelta {
                index,
                delta_type,
                text,
                tool_use_id,
                name,
                input,
            } => {
                let json = serde_json::json!({
                    "index": index,
                    "type": delta_type,
                    "text": text,
                    "tool_use_id": tool_use_id,
                    "name": name,
                    "input": input,
                });
                assert_eq!(
                    delta_type, expected_type,
                    "event type mismatch at index {idx}"
                );
                json
            }
            AnthropicStreamMessage::ContentBlockStop { index } => {
                assert_eq!(
                    "content_block_stop", expected_type,
                    "event type mismatch at index {idx}"
                );
                serde_json::json!({"index": index, "type": expected_type})
            }
            AnthropicStreamMessage::MessageDelta {
                delta_type,
                stop_reason,
                stop_sequence,
                usage,
            } => {
                let json = serde_json::json!({
                    "type": delta_type,
                    "stop_reason": stop_reason,
                    "stop_sequence": stop_sequence,
                    "usage": usage,
                });
                assert_eq!(
                    delta_type, expected_type,
                    "event type mismatch at index {idx}"
                );
                json
            }
            AnthropicStreamMessage::MessageStop => {
                assert_eq!(
                    "message_stop", expected_type,
                    "event type mismatch at index {idx}"
                );
                serde_json::json!({})
            }
        }
    }

    // ---------------------------------------------------------------------------
    // Scenario 1: Text-only stream
    // ---------------------------------------------------------------------------
    #[gtest]
    fn test_text_only_stream() {
        let inference_id = Uuid::now_v7();
        let episode_id = Uuid::now_v7();
        let model = "test_model";
        let mut state = AnthropicStreamState::default();

        // Chunk 1: first text block starts.
        let chunk = make_chat_chunk(
            inference_id,
            episode_id,
            vec![ContentBlockChunk::Text(TextChunk {
                id: "1".to_string(),
                text: "Hello".to_string(),
            })],
            Some(Usage {
                input_tokens: Some(10),
                output_tokens: Some(5),
                provider_cache_read_input_tokens: None,
                provider_cache_write_input_tokens: None,
                cost: None,
            }),
            None,
        );

        let events = state.process_chunk(&chunk, &inference_id.to_string(), model);
        assert_eq!(
            events.len(),
            3,
            "should emit message_start + content_block_start + content_block_delta"
        );
        assert_event_type(&events, 0, "message");
        assert_event_type(&events, 1, "text");
        assert_event_type(&events, 2, "text_delta");

        // Chunk 2: text delta continues.
        let chunk2 = make_chat_chunk(
            inference_id,
            episode_id,
            vec![ContentBlockChunk::Text(TextChunk {
                id: "1".to_string(),
                text: " World".to_string(),
            })],
            Some(Usage {
                input_tokens: None,
                output_tokens: Some(3),
                provider_cache_read_input_tokens: None,
                provider_cache_write_input_tokens: None,
                cost: None,
            }),
            None,
        );

        let events2 = state.process_chunk(&chunk2, &inference_id.to_string(), model);
        assert_eq!(events2.len(), 1, "should emit only content_block_delta");
        assert_event_type(&events2, 0, "text_delta");

        // Finalize.
        let final_events = state.finalize();
        assert_eq!(
            final_events.len(),
            2,
            "should emit message_delta + message_stop"
        );
        assert_event_type(&final_events, 0, "message_delta");
        assert_event_type(&final_events, 1, "message_stop");

        // Check aggregated usage: 5 + 3 = 8 output tokens.
        let message_delta = &final_events[0];
        if let AnthropicStreamMessage::MessageDelta { usage, .. } = message_delta {
            assert_eq!(
                usage.output_tokens, 8,
                "output tokens should be aggregated: 5 + 3"
            );
            assert_eq!(
                usage.input_tokens, 10,
                "input tokens should be accumulated (only first chunk has it)"
            );
        } else {
            panic!("expected MessageDelta");
        }
    }

    // ---------------------------------------------------------------------------
    // Scenario 2: Tool-only stream (one tool call)
    // ---------------------------------------------------------------------------
    #[gtest]
    fn test_tool_only_stream() {
        let inference_id = Uuid::now_v7();
        let episode_id = Uuid::now_v7();
        let mut state = AnthropicStreamState::default();

        let chunk = make_chat_chunk(
            inference_id,
            episode_id,
            vec![ContentBlockChunk::ToolCall(ToolCallChunk {
                id: "tool_1".to_string(),
                raw_name: Some("get_weather".to_string()),
                raw_arguments: r#"{"location":"Tokyo"}"#.to_string(),
            })],
            None,
            None,
        );

        let events = state.process_chunk(&chunk, &inference_id.to_string(), "test_model");
        assert_eq!(
            events.len(),
            3,
            "should emit content_block_start + input_json_delta (+ message_start was emitted first)"
        );
        // message_start is only emitted on first chunk (events.is_empty check inside process_chunk)
        // Actually the first chunk always emits message_start, so:
        // message_start, content_block_start, content_block_delta(input_json_delta)
        assert_event_type(&events, 0, "message");
        assert_event_type(&events, 1, "tool_use");
        assert_event_type(&events, 2, "input_json_delta");

        // Finalize without finish reason (no ToolCall finish reason since model hasn't stopped).
        let final_events = state.finalize();
        assert_eq!(final_events.len(), 2);
        assert_event_type(&final_events, 0, "message_delta");
        assert_event_type(&final_events, 1, "message_stop");
    }

    // ---------------------------------------------------------------------------
    // Scenario 3: Mixed text → tool_use → text
    // ---------------------------------------------------------------------------
    #[gtest]
    fn test_mixed_text_tool_use_text() {
        let inference_id = Uuid::now_v7();
        let episode_id = Uuid::now_v7();
        let mut state = AnthropicStreamState::default();

        // Chunk 1: initial text.
        let chunk1 = make_chat_chunk(
            inference_id,
            episode_id,
            vec![ContentBlockChunk::Text(TextChunk {
                id: "1".to_string(),
                text: "Let me check.".to_string(),
            })],
            None,
            None,
        );
        let events1 = state.process_chunk(&chunk1, &inference_id.to_string(), "model");
        assert_eq!(events1.len(), 3); // message_start + block_start + delta
        assert_event_type(&events1, 1, "text");

        // Chunk 2: tool_use — closes text block, opens tool block.
        let chunk2 = make_chat_chunk(
            inference_id,
            episode_id,
            vec![ContentBlockChunk::ToolCall(ToolCallChunk {
                id: "tool_1".to_string(),
                raw_name: Some("search".to_string()),
                raw_arguments: r#"{"query":"weather"}"#.to_string(),
            })],
            None,
            None,
        );
        let events2 = state.process_chunk(&chunk2, &inference_id.to_string(), "model");
        assert!(
            events2.len() >= 2,
            "should close text block and open tool block, got {events2:?}"
        );
        // Find the content_block_stop and check ordering
        let has_stop = events2
            .iter()
            .any(|e| matches!(e, AnthropicStreamMessage::ContentBlockStop { .. }));
        assert!(
            has_stop,
            "should have a content_block_stop closing text block"
        );

        // Chunk 3: more text after tool_use — closes tool block, opens new text block.
        let chunk3 = make_chat_chunk(
            inference_id,
            episode_id,
            vec![ContentBlockChunk::Text(TextChunk {
                id: "2".to_string(),
                text: "Done!".to_string(),
            })],
            None,
            None,
        );
        let events3 = state.process_chunk(&chunk3, &inference_id.to_string(), "model");
        let has_tool_stop = events3
            .iter()
            .any(|e| matches!(e, AnthropicStreamMessage::ContentBlockStop { .. }));
        assert!(
            has_tool_stop,
            "should close tool block before opening text block"
        );

        // Finalize: verify total blocks.
        let final_events = state.finalize();
        assert_event_type(&final_events, 0, "message_delta");
        assert_event_type(&final_events, 1, "message_stop");
    }

    // ---------------------------------------------------------------------------
    // Scenario 4: Thinking + text (reasoning models)
    // ---------------------------------------------------------------------------
    #[gtest]
    fn test_thinking_then_text() {
        let inference_id = Uuid::now_v7();
        let episode_id = Uuid::now_v7();
        let mut state = AnthropicStreamState::default();

        // Chunk 1: thinking block.
        let chunk1 = make_chat_chunk(
            inference_id,
            episode_id,
            vec![ContentBlockChunk::Thought(ThoughtChunk {
                id: "1".to_string(),
                text: Some("Let me think about this...".to_string()),
                signature: Some("sig001".to_string()),
                summary_id: None,
                summary_text: None,
                provider_type: None,
                extra_data: None,
            })],
            None,
            None,
        );
        let events1 = state.process_chunk(&chunk1, &inference_id.to_string(), "model");
        assert!(events1.iter().any(|e| matches!(e, AnthropicStreamMessage::ContentBlockStart { content_type, .. } if content_type == "thinking")));

        // Chunk 2: thinking delta — same block, new signature.
        let chunk2 = make_chat_chunk(
            inference_id,
            episode_id,
            vec![ContentBlockChunk::Thought(ThoughtChunk {
                id: "1".to_string(),
                text: Some(" More thinking...".to_string()),
                signature: Some("sig002".to_string()),
                summary_id: None,
                summary_text: None,
                provider_type: None,
                extra_data: None,
            })],
            None,
            None,
        );
        let events2 = state.process_chunk(&chunk2, &inference_id.to_string(), "model");
        let thinking_deltas: Vec<_> = events2.iter()
            .filter(|e| matches!(e, AnthropicStreamMessage::ContentBlockDelta { delta_type, .. } if delta_type == "thinking_delta"))
            .collect();
        assert!(
            !thinking_deltas.is_empty(),
            "should emit thinking_delta for continuation"
        );

        // Chunk 3: text after thinking — should close thinking block.
        let chunk3 = make_chat_chunk(
            inference_id,
            episode_id,
            vec![ContentBlockChunk::Text(TextChunk {
                id: "2".to_string(),
                text: "Here is my answer.".to_string(),
            })],
            None,
            None,
        );
        let events3 = state.process_chunk(&chunk3, &inference_id.to_string(), "model");
        let has_thinking_stop = events3
            .iter()
            .any(|e| matches!(e, AnthropicStreamMessage::ContentBlockStop { .. }));
        assert!(
            has_thinking_stop,
            "should close thinking block before opening text block"
        );

        let text_starts = events3.iter()
            .filter(|e| matches!(e, AnthropicStreamMessage::ContentBlockStart { content_type, .. } if content_type == "text"));
        assert!(
            text_starts.count() >= 1,
            "should open text block after thinking"
        );

        // Finalize: finish reason should show up.
        let finalizer = state;
        let stop_events = finalizer.finalize();
        assert_eq!(stop_events.len(), 2);
    }

    // ---------------------------------------------------------------------------
    // Scenario 5: Empty content then finish
    // ---------------------------------------------------------------------------
    #[gtest]
    fn test_empty_content_then_finish() {
        let inference_id = Uuid::now_v7();
        let episode_id = Uuid::now_v7();
        let mut state = AnthropicStreamState::default();

        // Chunk with empty content blocks and a finish reason.
        let chunk = make_chat_chunk(
            inference_id,
            episode_id,
            vec![],
            None,
            Some(FinishReason::Stop),
        );
        let events = state.process_chunk(&chunk, &inference_id.to_string(), "model");
        assert!(!events.is_empty(), "should emit at least message_start");
        assert_event_type(&events, 0, "message");

        let final_events = state.finalize();
        assert_eq!(final_events.len(), 2);
        assert_event_type(&final_events, 0, "message_delta");
        assert_event_type(&final_events, 1, "message_stop");

        // Stop reason should be EndTurn.
        if let AnthropicStreamMessage::MessageDelta { stop_reason, .. } = &final_events[0] {
            assert_eq!(*stop_reason, Some(AnthropicStopReason::EndTurn));
        } else {
            panic!("expected MessageDelta");
        }
    }

    // ---------------------------------------------------------------------------
    // Scenario 6: Multiple tool_use blocks in parallel
    // ---------------------------------------------------------------------------
    #[gtest]
    fn test_multiple_tool_use_blocks() {
        let inference_id = Uuid::now_v7();
        let episode_id = Uuid::now_v7();
        let mut state = AnthropicStreamState::default();

        // Chunk with two parallel tool calls.
        let chunk = make_chat_chunk(
            inference_id,
            episode_id,
            vec![
                ContentBlockChunk::ToolCall(ToolCallChunk {
                    id: "tool_1".to_string(),
                    raw_name: Some("search_1".to_string()),
                    raw_arguments: r#"{"q":"weather"}"#.to_string(),
                }),
                ContentBlockChunk::ToolCall(ToolCallChunk {
                    id: "tool_2".to_string(),
                    raw_name: Some("search_2".to_string()),
                    raw_arguments: r#"{"q":"news"}"#.to_string(),
                }),
            ],
            None,
            None,
        );

        let events = state.process_chunk(&chunk, &inference_id.to_string(), "model");
        // message_start + tool_1 start + tool_1 delta + tool_1 stop + tool_2 start + tool_2 delta
        assert_eq!(
            events.len(),
            6,
            "should emit message_start + 2 tool_use blocks with deltas and block stop"
        );

        // Verify block indices are monotonic.
        let starts: Vec<_> = events
            .iter()
            .filter_map(|e| match e {
                AnthropicStreamMessage::ContentBlockStart { index, .. } => Some(*index),
                _ => None,
            })
            .collect();
        assert_eq!(starts, vec![0, 1], "indices should be monotonic: 0, 1");

        // Same tool ID should get same index.
        // Send another chunk with the same tool_1.
        let chunk2 = make_chat_chunk(
            inference_id,
            episode_id,
            vec![ContentBlockChunk::ToolCall(ToolCallChunk {
                id: "tool_1".to_string(),
                raw_name: Some("search_1".to_string()),
                raw_arguments: r#"{"q":"updated"}"#.to_string(),
            })],
            None,
            None,
        );
        let events2 = state.process_chunk(&chunk2, &inference_id.to_string(), "model");
        let delta_indices: Vec<_> = events2
            .iter()
            .filter_map(|e| match e {
                AnthropicStreamMessage::ContentBlockDelta { index, .. } => Some(*index),
                _ => None,
            })
            .collect();
        assert!(
            delta_indices.iter().all(|&i| i == 0),
            "same tool_id should reuse index 0, got deltas at indices {delta_indices:?}"
        );
    }

    // ---------------------------------------------------------------------------
    // Scenario 7: Signature delta arriving separately from thinking delta
    // ---------------------------------------------------------------------------
    #[gtest]
    fn test_separate_signature_delta() {
        let inference_id = Uuid::now_v7();
        let episode_id = Uuid::now_v7();
        let mut state = AnthropicStreamState::default();

        // Chunk 1: thinking without signature.
        let chunk1 = make_chat_chunk(
            inference_id,
            episode_id,
            vec![ContentBlockChunk::Thought(ThoughtChunk {
                id: "1".to_string(),
                text: Some("Thinking...".to_string()),
                signature: None,
                summary_id: None,
                summary_text: None,
                provider_type: None,
                extra_data: None,
            })],
            None,
            None,
        );
        let events1 = state.process_chunk(&chunk1, &inference_id.to_string(), "model");
        let start = events1.iter()
            .find(|e| matches!(e, AnthropicStreamMessage::ContentBlockStart { content_type, .. } if content_type == "thinking"));
        assert!(start.is_some(), "should start thinking block");

        // Chunk 2: thinking with signature delta.
        let chunk2 = make_chat_chunk(
            inference_id,
            episode_id,
            vec![ContentBlockChunk::Thought(ThoughtChunk {
                id: "1".to_string(),
                text: None, // no text, just signature.
                signature: Some("sig-abc".to_string()),
                summary_id: None,
                summary_text: None,
                provider_type: None,
                extra_data: None,
            })],
            None,
            None,
        );
        let events2 = state.process_chunk(&chunk2, &inference_id.to_string(), "model");
        let thinking_delta = events2.iter()
            .filter(|e| matches!(e, AnthropicStreamMessage::ContentBlockDelta { delta_type, .. } if delta_type == "thinking_delta"));
        assert!(
            thinking_delta.count() >= 1,
            "should emit thinking_delta for the signature-only chunk"
        );

        // Chunk 3: another text block — should close thinking.
        let chunk3 = make_chat_chunk(
            inference_id,
            episode_id,
            vec![ContentBlockChunk::Text(TextChunk {
                id: "2".to_string(),
                text: "Answer.".to_string(),
            })],
            None,
            None,
        );
        let events3 = state.process_chunk(&chunk3, &inference_id.to_string(), "model");
        assert!(
            events3
                .iter()
                .any(|e| matches!(e, AnthropicStreamMessage::ContentBlockStop { .. })),
            "should close thinking block"
        );
    }

    // ---------------------------------------------------------------------------
    // Scenario 8: message_delta carries final accumulated usage
    // ---------------------------------------------------------------------------
    #[gtest]
    fn test_accumulated_usage_in_message_delta() {
        let inference_id = Uuid::now_v7();
        let episode_id = Uuid::now_v7();
        let mut state = AnthropicStreamState::default();

        // Three chunks with increasing usage.
        for i in 0..3u32 {
            let chunk = make_chat_chunk(
                inference_id,
                episode_id,
                vec![ContentBlockChunk::Text(TextChunk {
                    id: "1".to_string(),
                    text: format!("chunk {i}"),
                })],
                Some(Usage {
                    input_tokens: Some(10),     // every chunk reports 10 input tokens
                    output_tokens: Some(i + 1), // 1, 2, 3 output tokens
                    provider_cache_read_input_tokens: Some(5),
                    provider_cache_write_input_tokens: None,
                    cost: None,
                }),
                Some(FinishReason::Stop),
            );
            state.process_chunk(&chunk, &inference_id.to_string(), "model");
        }

        let final_events = state.finalize();
        if let AnthropicStreamMessage::MessageDelta {
            usage, stop_reason, ..
        } = &final_events[0]
        {
            // 10 + 10 + 10 = 30 input tokens accumulated.
            assert_eq!(
                usage.input_tokens, 30,
                "input tokens should be fully accumulated"
            );
            // 1 + 2 + 3 = 6 output tokens.
            assert_eq!(
                usage.output_tokens, 6,
                "output tokens should be summed to 6"
            );
            // Cache read: 5 + 5 + 5 = 15.
            assert_eq!(
                usage.cache_creation_input_tokens, None,
                "no cache write tokens"
            );
            assert_eq!(
                usage.cache_read_input_tokens,
                Some(15),
                "cache read tokens should be accumulated"
            );
            assert_eq!(
                *stop_reason,
                Some(AnthropicStopReason::EndTurn),
                "stop_reason should be EndTurn"
            );
        } else {
            panic!("expected MessageDelta");
        }
    }

    // ---------------------------------------------------------------------------
    // Scenario 9: RedactedThinking (Unknown) block
    // ---------------------------------------------------------------------------
    #[gtest]
    fn test_redacted_thinking_block() {
        let inference_id = Uuid::now_v7();
        let episode_id = Uuid::now_v7();
        let mut state = AnthropicStreamState::default();

        let chunk = make_chat_chunk(
            inference_id,
            episode_id,
            vec![ContentBlockChunk::Unknown(UnknownChunk {
                id: "1".to_string(),
                data: serde_json::json!("redacted reasoning content".to_string()),
                model_name: None,
                provider_name: None,
            })],
            None,
            None,
        );

        let events = state.process_chunk(&chunk, &inference_id.to_string(), "model");
        let thinking_start = events.iter()
            .find(|e| matches!(e, AnthropicStreamMessage::ContentBlockStart { content_type, .. } if content_type == "redacted_thinking"));
        assert!(
            thinking_start.is_some(),
            "should open redacted_thinking block"
        );

        let has_delta = events.iter()
            .any(|e| matches!(e, AnthropicStreamMessage::ContentBlockDelta { delta_type, .. } if delta_type == "thinking_delta"));
        assert!(has_delta, "should emit thinking_delta for redacted content");
    }
}
