//! Owned-data types for deserializing incoming Anthropic `/v1/messages` requests
//! and serializing responses.
//!
//! These types mirror the outbound types in `providers/anthropic.rs` but use owned data
//! (`String` instead of `&'a str`) and are suitable for Axum request body deserialization.
//!
//! The request-to-internal translation logic lives in
//! `AnthropicMessagesParams::try_into_params()` (this file, `impl` block).
//! The response-to-Anthropic translation lives in `inference_response_to_anthropic()`.

use mime::MediaType;
use serde::{Deserialize, Serialize, de::Error as _};
use serde_json::Value;
use std::collections::HashMap;
use url::Url;
use uuid::Uuid;

use crate::inference::types::{Base64File, File, UrlFile};

use crate::cache::CacheParamsOptions;
use crate::config::Namespace;
use crate::endpoints::anthropic_compatible::types::streaming::finish_reason_to_anthropic_stop_reason;
use crate::endpoints::inference::{InferenceCredentials, InferenceParams};
use crate::error::{Error, ErrorDetails};
use crate::inference::types::Input;
use crate::inference::types::extra_body::UnfilteredInferenceExtraBody;
use crate::inference::types::extra_headers::UnfilteredInferenceExtraHeaders;
use tensorzero_inference_types::{CacheControlSpan, CacheControlTarget};
use tensorzero_types::content::{System, Text, Thought, Unknown};
use tensorzero_types::inference_params::ChatCompletionInferenceParams;
use tensorzero_types::message::InputMessage;
use tensorzero_types::message::InputMessageContent;
use tensorzero_types::role::Role;
use tensorzero_types::tool::{ToolCall, ToolCallWrapper, ToolResult};
use tensorzero_types::{ContentBlockChatOutput, InferenceResponse, JsonInferenceOutput, Usage};
use tensorzero_types_providers::anthropic::{AnthropicCacheControl, AnthropicStopReason};

// ============================================================================
// Top-level request body
// ============================================================================

/// The top-level request body for `POST /v1/messages`.
///
/// Mirrors Anthropic's `MessagesRequest` but with owned data.
#[derive(Clone, Debug, Deserialize)]
pub struct AnthropicMessagesParams {
    /// The model that will complete your messages.
    pub model: String,
    /// The messages to send to the model.
    pub messages: Vec<AnthropicMessageOwned>,
    /// System prompt (String or array of text blocks).
    #[serde(default, deserialize_with = "deserialize_system")]
    pub system: AnthropicSystem,
    /// Maximum tokens to generate.
    pub max_tokens: u32,
    /// Temperature.
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Nucleus sampling probability.
    #[serde(default)]
    pub top_p: Option<f32>,
    /// Top-K sampling.
    #[serde(default)]
    pub top_k: Option<i32>,
    /// Stop sequences.
    #[serde(default)]
    pub stop_sequences: Option<Vec<String>>,
    /// Whether to stream.
    #[serde(default)]
    pub stream: bool,
    /// Tools.
    #[serde(default)]
    pub tools: Option<Vec<AnthropicToolOwned>>,
    /// Tool choice.
    #[serde(default)]
    pub tool_choice: Option<AnthropicToolChoiceOwned>,
    /// Metadata (TensorZero extension fields live under `metadata.tensorzero`).
    #[serde(default)]
    pub metadata: Option<AnthropicMetadata>,
    /// Reserved: Anthropic service tier routing.
    #[serde(default)]
    pub service_tier: Option<String>,
    /// Reserved: Anthropic Code Execution container.
    #[serde(default)]
    pub container: Option<String>,
}

/// Deserializes the `system` field, which can be a `String` or a `Vec<SystemContentBlock>`.
fn deserialize_system<'de, D>(deserializer: D) -> Result<AnthropicSystem, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::Error;

    let value = Value::deserialize(deserializer)?;
    match value {
        Value::String(s) => Ok(AnthropicSystem::String(s)),
        Value::Array(items) => {
            let blocks: Result<Vec<_>, _> = items
                .into_iter()
                .map(AnthropicSystemContentBlockOwned::from_value)
                .collect();
            Ok(AnthropicSystem::Blocks(blocks.map_err(D::Error::custom)?))
        }
        _ => Err(D::Error::custom(
            "system must be a string or array of content blocks",
        )),
    }
}

/// System prompt — either a String or an array of content blocks.
#[derive(Clone, Debug, PartialEq)]
pub enum AnthropicSystem {
    String(String),
    Blocks(Vec<AnthropicSystemContentBlockOwned>),
}

impl Default for AnthropicSystem {
    fn default() -> Self {
        AnthropicSystem::String(String::default())
    }
}

/// Single content block within a system prompt.
#[derive(Clone, Debug, PartialEq)]
pub enum AnthropicSystemContentBlockOwned {
    Text {
        text: String,
        cache_control: Option<tensorzero_types_providers::anthropic::AnthropicCacheControl>,
    },
}

impl AnthropicSystemContentBlockOwned {
    fn from_value(value: Value) -> Result<Self, serde_json::Error> {
        #[derive(Deserialize)]
        struct Helper {
            #[serde(rename = "type")]
            r#type: String,
            #[serde(default)]
            text: String,
            #[serde(default)]
            cache_control: Option<tensorzero_types_providers::anthropic::AnthropicCacheControl>,
        }

        let helper: Helper = serde_json::from_value(value)?;
        match helper.r#type.as_str() {
            "text" => Ok(AnthropicSystemContentBlockOwned::Text {
                text: helper.text,
                cache_control: helper.cache_control,
            }),
            other => Err(serde_json::Error::custom(format!(
                "Unknown system content block type: {other}"
            ))),
        }
    }
}

// ============================================================================
// Message types
// ============================================================================

/// A single message in the conversation (owned).
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct AnthropicMessageOwned {
    pub role: AnthropicRoleOwned,
    #[serde(default, deserialize_with = "deserialize_message_content")]
    pub content: AnthropicMessageContentOwned,
}

/// Message content — either a string or a Vec of blocks.
#[derive(Clone, Debug, PartialEq)]
pub enum AnthropicMessageContentOwned {
    String(String),
    Blocks(Vec<AnthropicContentBlockOwned>),
}

impl Default for AnthropicMessageContentOwned {
    fn default() -> Self {
        AnthropicMessageContentOwned::String(String::default())
    }
}

fn deserialize_message_content<'de, D>(
    deserializer: D,
) -> Result<AnthropicMessageContentOwned, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::Error;

    let value = Value::deserialize(deserializer)?;
    match value {
        Value::String(s) => Ok(AnthropicMessageContentOwned::String(s)),
        Value::Array(items) => {
            let blocks: Result<Vec<_>, _> = items
                .into_iter()
                .map(AnthropicContentBlockOwned::from_value)
                .collect();
            Ok(AnthropicMessageContentOwned::Blocks(
                blocks.map_err(D::Error::custom)?,
            ))
        }
        _ => Err(D::Error::custom(
            "message content must be a string or array of content blocks",
        )),
    }
}

/// Message role.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum AnthropicRoleOwned {
    User,
    Assistant,
}

// ============================================================================
// Content blocks
// ============================================================================

/// A content block within a user or assistant message.
///
/// Owned equivalent of outbound `AnthropicContentBlock`, extended with
/// `Image`, `Document`, `ToolResult`, and per-block `cache_control`.
#[derive(Clone, Debug, PartialEq)]
pub enum AnthropicContentBlockOwned {
    Text {
        text: String,
        cache_control: Option<AnthropicCacheControl>,
    },
    Image {
        cache_control: Option<AnthropicCacheControl>,
        source: AnthropicImageSourceOwned,
    },
    Document {
        cache_control: Option<AnthropicCacheControl>,
        source: AnthropicDocumentSourceOwned,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
        cache_control: Option<AnthropicCacheControl>,
    },
    ToolResult {
        tool_use_id: String,
        content: Vec<AnthropicContentBlockOwned>,
        is_error: Option<bool>,
        cache_control: Option<AnthropicCacheControl>,
    },
    Thinking {
        thinking: String,
        signature: String,
    },
    RedactedThinking {
        data: String,
    },
}

impl AnthropicContentBlockOwned {
    fn from_value(value: Value) -> Result<Self, serde_json::Error> {
        #[derive(Deserialize)]
        struct BlockHeader {
            #[serde(rename = "type")]
            r#type: String,
        }

        let header: BlockHeader = serde_json::from_value(value.clone())?;
        match header.r#type.as_str() {
            "text" => {
                #[derive(Deserialize)]
                struct TextBlock {
                    text: String,
                    #[serde(default)]
                    cache_control: Option<AnthropicCacheControl>,
                }
                let block: TextBlock = serde_json::from_value(value)?;
                Ok(AnthropicContentBlockOwned::Text {
                    text: block.text,
                    cache_control: block.cache_control,
                })
            }
            "image" => {
                #[derive(Deserialize)]
                struct ImageBlock {
                    #[serde(default)]
                    cache_control: Option<AnthropicCacheControl>,
                    source: AnthropicImageSourceOwned,
                }
                let block: ImageBlock = serde_json::from_value(value)?;
                Ok(AnthropicContentBlockOwned::Image {
                    cache_control: block.cache_control,
                    source: block.source,
                })
            }
            "document" => {
                #[derive(Deserialize)]
                struct DocumentBlock {
                    #[serde(default)]
                    cache_control: Option<AnthropicCacheControl>,
                    source: AnthropicDocumentSourceOwned,
                }
                let block: DocumentBlock = serde_json::from_value(value)?;
                Ok(AnthropicContentBlockOwned::Document {
                    cache_control: block.cache_control,
                    source: block.source,
                })
            }
            "tool_use" => {
                #[derive(Deserialize)]
                struct ToolUseBlock {
                    id: String,
                    name: String,
                    input: Value,
                    #[serde(default)]
                    cache_control: Option<AnthropicCacheControl>,
                }
                let block: ToolUseBlock = serde_json::from_value(value)?;
                Ok(AnthropicContentBlockOwned::ToolUse {
                    id: block.id,
                    name: block.name,
                    input: block.input,
                    cache_control: block.cache_control,
                })
            }
            "tool_result" => {
                #[derive(Deserialize)]
                struct ToolResultBlock {
                    tool_use_id: String,
                    #[serde(default)]
                    content: Value,
                    #[serde(default)]
                    is_error: Option<bool>,
                    #[serde(default)]
                    cache_control: Option<AnthropicCacheControl>,
                }
                let block: ToolResultBlock = serde_json::from_value(value)?;
                let content = if let Value::Array(items) = block.content {
                    items
                        .into_iter()
                        .map(AnthropicContentBlockOwned::from_value)
                        .collect::<Result<_, _>>()?
                } else if let Value::String(s) = block.content {
                    vec![AnthropicContentBlockOwned::Text {
                        text: s,
                        cache_control: None,
                    }]
                } else {
                    return Err(serde_json::Error::custom(
                        "tool_result content must be string or array",
                    ));
                };
                Ok(AnthropicContentBlockOwned::ToolResult {
                    tool_use_id: block.tool_use_id,
                    content,
                    is_error: block.is_error,
                    cache_control: block.cache_control,
                })
            }
            "thinking" => {
                #[derive(Deserialize)]
                struct ThinkingBlock {
                    thinking: String,
                    signature: String,
                }
                let block: ThinkingBlock = serde_json::from_value(value)?;
                Ok(AnthropicContentBlockOwned::Thinking {
                    thinking: block.thinking,
                    signature: block.signature,
                })
            }
            "redacted_thinking" => {
                #[derive(Deserialize)]
                struct RedactedThinkingBlock {
                    data: String,
                }
                let block: RedactedThinkingBlock = serde_json::from_value(value)?;
                Ok(AnthropicContentBlockOwned::RedactedThinking { data: block.data })
            }
            other => Err(serde_json::Error::custom(format!(
                "Unknown content block type: {other}"
            ))),
        }
    }

    /// Returns the kind of this content block for error messages.
    pub fn kind(&self) -> &'static str {
        match self {
            AnthropicContentBlockOwned::Text { .. } => "text",
            AnthropicContentBlockOwned::Image { .. } => "image",
            AnthropicContentBlockOwned::Document { .. } => "document",
            AnthropicContentBlockOwned::ToolUse { .. } => "tool_use",
            AnthropicContentBlockOwned::ToolResult { .. } => "tool_result",
            AnthropicContentBlockOwned::Thinking { .. } => "thinking",
            AnthropicContentBlockOwned::RedactedThinking { .. } => "redacted_thinking",
        }
    }
}

/// Image source for Anthropic image content blocks.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicImageSourceOwned {
    Base64 { media_type: String, data: String },
    Url { url: String },
}

/// Document source for Anthropic document content blocks.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicDocumentSourceOwned {
    Base64 { media_type: String, data: String },
    Url { url: String },
}

// ============================================================================
// Tool types
// ============================================================================

/// An Anthropic tool definition.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct AnthropicToolOwned {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(rename = "input_schema")]
    pub parameters: Value,
    #[serde(default)]
    pub strict: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<AnthropicCacheControl>,
}

/// Anthropic tool choice.
#[derive(Clone, Debug, PartialEq)]
pub enum AnthropicToolChoiceOwned {
    Auto {
        disable_parallel_tool_use: Option<bool>,
    },
    Any {
        disable_parallel_tool_use: Option<bool>,
    },
    Tool {
        name: String,
        disable_parallel_tool_use: Option<bool>,
    },
    None {
        disable_parallel_tool_use: Option<bool>,
    },
}

impl<'de> Deserialize<'de> for AnthropicToolChoiceOwned {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::Error;
        use serde_json::Value;

        let value = Value::deserialize(deserializer)?;

        match &value {
            Value::String(s) => match s.as_str() {
                "auto" => Ok(AnthropicToolChoiceOwned::Auto {
                    disable_parallel_tool_use: None,
                }),
                "any" => Ok(AnthropicToolChoiceOwned::Any {
                    disable_parallel_tool_use: None,
                }),
                "none" => Ok(AnthropicToolChoiceOwned::None {
                    disable_parallel_tool_use: None,
                }),
                other => Err(D::Error::custom(format!(
                    "Invalid tool choice string: {other}. Expected 'auto', 'any', or 'none'."
                ))),
            },
            Value::Object(obj) => {
                if let Some(type_value) = obj.get("type") {
                    if let Some(type_str) = type_value.as_str() {
                        match type_str {
                            "auto" => {
                                #[derive(Deserialize)]
                                struct Helper {
                                    #[serde(default)]
                                    disable_parallel_tool_use: Option<bool>,
                                }
                                let h: Helper =
                                    serde_json::from_value(value).map_err(D::Error::custom)?;
                                Ok(AnthropicToolChoiceOwned::Auto {
                                    disable_parallel_tool_use: h.disable_parallel_tool_use,
                                })
                            }
                            "any" => {
                                #[derive(Deserialize)]
                                struct Helper {
                                    #[serde(default)]
                                    disable_parallel_tool_use: Option<bool>,
                                }
                                let h: Helper =
                                    serde_json::from_value(value).map_err(D::Error::custom)?;
                                Ok(AnthropicToolChoiceOwned::Any {
                                    disable_parallel_tool_use: h.disable_parallel_tool_use,
                                })
                            }
                            "tool" => {
                                #[derive(Deserialize)]
                                struct Helper {
                                    name: String,
                                    #[serde(default)]
                                    disable_parallel_tool_use: Option<bool>,
                                }
                                let h: Helper =
                                    serde_json::from_value(value).map_err(D::Error::custom)?;
                                Ok(AnthropicToolChoiceOwned::Tool {
                                    name: h.name,
                                    disable_parallel_tool_use: h.disable_parallel_tool_use,
                                })
                            }
                            "none" => {
                                #[derive(Deserialize)]
                                struct Helper {
                                    #[serde(default)]
                                    disable_parallel_tool_use: Option<bool>,
                                }
                                let h: Helper =
                                    serde_json::from_value(value).map_err(D::Error::custom)?;
                                Ok(AnthropicToolChoiceOwned::None {
                                    disable_parallel_tool_use: h.disable_parallel_tool_use,
                                })
                            }
                            other => Err(D::Error::custom(format!(
                                "Invalid tool choice type: {other}"
                            ))),
                        }
                    } else {
                        Err(D::Error::custom(
                            "Tool choice 'type' field must be a string",
                        ))
                    }
                } else {
                    Err(D::Error::custom(
                        "Tool choice must have a 'type' field if it is an object",
                    ))
                }
            }
            _ => Err(D::Error::custom("Tool choice must be a string or object")),
        }
    }
}

/// Extract `disable_parallel_tool_use` from any variant.
#[expect(
    dead_code,
    reason = "used by full Params translation in task tensorzero-s4w"
)]
fn get_disable_parallel(choice: &AnthropicToolChoiceOwned) -> Option<bool> {
    match choice {
        AnthropicToolChoiceOwned::Auto {
            disable_parallel_tool_use,
        }
        | AnthropicToolChoiceOwned::Any {
            disable_parallel_tool_use,
        }
        | AnthropicToolChoiceOwned::Tool {
            disable_parallel_tool_use,
            ..
        }
        | AnthropicToolChoiceOwned::None {
            disable_parallel_tool_use,
        } => *disable_parallel_tool_use,
    }
}

// ============================================================================
// Metadata
// ============================================================================

/// Top-level `metadata` field in Anthropic requests.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct AnthropicMetadata {
    /// User ID (lifted into TensorZero tags).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
    /// TensorZero extension fields.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub tensorzero: Option<AnthropicTensorZeroMetadata>,
}

/// TensorZero-specific extension fields, namespaced under `metadata.tensorzero`.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct AnthropicTensorZeroMetadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub episode_id: Option<Uuid>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub variant_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub tags: HashMap<String, String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_options: Option<CacheParamsOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub extra_body: UnfilteredInferenceExtraBody,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub extra_headers: UnfilteredInferenceExtraHeaders,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub dryrun: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub namespace: Option<Namespace>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub credentials: InferenceCredentials,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub params: Option<InferenceParams>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub include_raw_usage: Option<bool>,
}

// ============================================================================
// Response types
// ============================================================================

/// Top-level response for non-streaming `POST /v1/messages`.
#[derive(Debug, Serialize)]
pub struct AnthropicResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub response_type: String,
    pub role: String,
    pub content: Vec<AnthropicResponseContentBlock>,
    pub model: String,
    pub stop_reason: Option<AnthropicStopReason>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequence: Option<i32>,
    pub usage: AnthropicResponseUsage,
}

/// Content block in Anthropic response.
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicResponseContentBlock {
    Text {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<AnthropicCacheControl>,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<AnthropicCacheControl>,
    },
    Thinking {
        thinking: String,
        signature: String,
    },
    RedactedThinking {
        data: String,
    },
}

/// Usage in Anthropic response.
#[derive(Debug, Serialize)]
pub struct AnthropicResponseUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_input_tokens: Option<u32>,
}

// ============================================================================
// Response translation: inference result → Anthropic response
// ============================================================================

/// Converts an `InferenceOutput` into an `AnthropicResponse`.
///
/// Converts both `Chat` and `Json` inference responses to Anthropic format.
/// JSON responses are wrapped as a single text content block with the serialized JSON.
pub fn inference_response_to_anthropic(
    output: crate::endpoints::inference::InferenceOutput,
    model_prefix: &str,
) -> Result<AnthropicResponse, Error> {
    let (inference_id, variant_name, content_blocks, usage, finish_reason) = match output {
        crate::endpoints::inference::InferenceOutput::NonStreaming(response) => {
            let (inference_id, variant_name, content_blocks, usage, finish_reason) = match response
            {
                InferenceResponse::Chat(chat) => (
                    chat.inference_id,
                    chat.variant_name,
                    chat.content,
                    chat.usage,
                    chat.finish_reason,
                ),
                InferenceResponse::Json(json) => {
                    let text = match &json.output {
                        JsonInferenceOutput { raw: Some(raw), .. } => raw.clone(),
                        JsonInferenceOutput {
                            parsed: Some(parsed),
                            ..
                        } => serde_json::to_string(parsed).unwrap_or_else(|_| parsed.to_string()),
                        JsonInferenceOutput {
                            raw: None,
                            parsed: None,
                        } => "null".to_string(),
                    };
                    return Ok(AnthropicResponse {
                        id: format!("msg_{}", json.inference_id),
                        response_type: "message".to_string(),
                        role: "assistant".to_string(),
                        content: vec![AnthropicResponseContentBlock::Text {
                            text,
                            cache_control: None,
                        }],
                        model: if model_prefix.is_empty() {
                            json.variant_name.clone()
                        } else {
                            format!("{model_prefix}{}", json.variant_name)
                        },
                        stop_reason: None,
                        stop_sequence: None,
                        usage: usage_to_anthropic(&json.usage),
                    });
                }
            };
            (
                inference_id,
                variant_name,
                content_blocks,
                usage,
                finish_reason,
            )
        }
        crate::endpoints::inference::InferenceOutput::Streaming(_) => {
            return Err(Error::new(ErrorDetails::InvalidRequest {
                message: "Streaming responses are not expected for non-streaming requests."
                    .to_string(),
            }));
        }
    };

    let content: Vec<AnthropicResponseContentBlock> = content_blocks
        .into_iter()
        .map(chat_output_to_anthropic_block)
        .collect();

    let usage = usage_to_anthropic(&usage);

    let stop_reason = finish_reason_to_anthropic_stop_reason(finish_reason);

    let model = if model_prefix.is_empty() {
        variant_name
    } else {
        format!("{model_prefix}{variant_name}")
    };

    Ok(AnthropicResponse {
        id: format!("msg_{inference_id}"),
        response_type: "message".to_string(),
        role: "assistant".to_string(),
        content,
        model,
        stop_reason,
        stop_sequence: None,
        usage,
    })
}

/// Converts a `ContentBlockChatOutput` into an `AnthropicResponseContentBlock`.
fn chat_output_to_anthropic_block(block: ContentBlockChatOutput) -> AnthropicResponseContentBlock {
    match block {
        ContentBlockChatOutput::Text(text) => AnthropicResponseContentBlock::Text {
            text: text.text,
            cache_control: None,
        },
        ContentBlockChatOutput::ToolCall(tc) => {
            // Parse the arguments JSON from raw string
            let input: Value = serde_json::from_str(&tc.raw_arguments)
                .unwrap_or_else(|_| Value::String(tc.raw_arguments.clone()));
            AnthropicResponseContentBlock::ToolUse {
                id: tc.id,
                name: tc.raw_name,
                input,
                cache_control: None,
            }
        }
        ContentBlockChatOutput::Thought(thought) => AnthropicResponseContentBlock::Thinking {
            thinking: thought.text.unwrap_or_default(),
            signature: thought.signature.unwrap_or_default(),
        },
        ContentBlockChatOutput::Unknown(unknown) => {
            // RedactedThinking from Anthropic arrives as a JSON string.
            // For other Unknown data shapes (e.g. future tool types), emit as
            // Text so the data is visible instead of being misclassified.
            if let Value::String(data) = &unknown.data {
                AnthropicResponseContentBlock::RedactedThinking { data: data.clone() }
            } else {
                AnthropicResponseContentBlock::Text {
                    text: unknown.data.to_string(),
                    cache_control: None,
                }
            }
        }
    }
}

/// Converts internal `Usage` to Anthropic `AnthropicResponseUsage`.
fn usage_to_anthropic(usage: &Usage) -> AnthropicResponseUsage {
    AnthropicResponseUsage {
        input_tokens: usage.input_tokens.unwrap_or(0),
        output_tokens: usage.output_tokens.unwrap_or(0),
        cache_creation_input_tokens: usage.provider_cache_write_input_tokens,
        cache_read_input_tokens: usage.provider_cache_read_input_tokens,
    }
}

// ============================================================================
// Request → internal Params translation
// ============================================================================

impl AnthropicMessagesParams {
    /// Converts an Anthropic `/v1/messages` request body into TensorZero's internal `Params`.
    pub fn try_into_params(self) -> Result<crate::endpoints::inference::Params, Error> {
        // Validate required fields
        if self.model.is_empty() {
            return Err(Error::new(ErrorDetails::InvalidRequest {
                message: "model is required".to_string(),
            }));
        }
        if self.messages.is_empty() {
            return Err(Error::new(ErrorDetails::InvalidRequest {
                message: "messages is required".to_string(),
            }));
        }

        // --- Model name parsing ---
        let (function_name, model_name) =
            if let Some(rest) = self.model.strip_prefix("tensorzero::function_name::") {
                (Some(rest.to_string()), None)
            } else if let Some(rest) = self.model.strip_prefix("tensorzero::model_name::") {
                (None, Some(rest.to_string()))
            } else {
                // Bare model/function name defaults to function_name
                (Some(self.model.clone()), None)
            };

        // --- System prompt ---
        let mut system_spans: Vec<(usize, AnthropicCacheControl)> = Vec::new();
        let system = match self.system {
            AnthropicSystem::String(s) if !s.is_empty() => Some(System::Text(s)),
            AnthropicSystem::Blocks(blocks) => {
                if blocks.is_empty() {
                    None
                } else {
                    let combined: String = blocks
                        .into_iter()
                        .enumerate()
                        .map(|(block_idx, b)| match b {
                            AnthropicSystemContentBlockOwned::Text {
                                text,
                                cache_control,
                            } => {
                                if let Some(marker) = cache_control {
                                    system_spans.push((block_idx, marker));
                                }
                                text
                            }
                        })
                        .collect::<Vec<_>>()
                        .join("\n");
                    if combined.is_empty() {
                        None
                    } else {
                        Some(System::Text(combined))
                    }
                }
            }
            AnthropicSystem::String(_) => None,
        };

        // --- Input messages ---
        // Build tool_use_id → name mapping from all messages first, so tool_result
        // blocks can resolve their tool names (Anthropic wire only carries tool_use_id).
        let mut tool_use_name_map: HashMap<String, String> = HashMap::new();
        for msg in &self.messages {
            if let AnthropicMessageContentOwned::Blocks(ref blocks) = msg.content {
                for block in blocks {
                    if let AnthropicContentBlockOwned::ToolUse { id, name, .. } = block {
                        tool_use_name_map.insert(id.clone(), name.clone());
                    }
                }
            }
        }

        let mut message_spans: Vec<(usize, usize, AnthropicCacheControl)> = Vec::new();
        let mut input_messages: Vec<InputMessage> = Vec::with_capacity(self.messages.len());
        for (message_idx, msg) in self.messages.into_iter().enumerate() {
            let role = match msg.role {
                AnthropicRoleOwned::User => Role::User,
                AnthropicRoleOwned::Assistant => Role::Assistant,
            };

            let content = match msg.content {
                AnthropicMessageContentOwned::String(s) => {
                    vec![InputMessageContent::Text(Text { text: s })]
                }
                AnthropicMessageContentOwned::Blocks(blocks) => {
                    let (content, spans) = blocks_into_input_content(blocks, &tool_use_name_map)?;
                    for (content_idx, marker) in spans {
                        message_spans.push((message_idx, content_idx, marker));
                    }
                    content
                }
            };

            input_messages.push(InputMessage { role, content });
        }

        // --- Dynamic tool params ---
        let mut tool_spans: Vec<(usize, AnthropicCacheControl)> = Vec::new();
        let additional_tools = self.tools.map(|tools| {
            tools
                .into_iter()
                .enumerate()
                .map(|(tool_idx, t)| {
                    if let Some(marker) = t.cache_control {
                        tool_spans.push((tool_idx, marker));
                    }
                    crate::tool::Tool::Function(crate::tool::FunctionTool {
                        name: t.name,
                        description: t.description.unwrap_or_default(),
                        parameters: t.parameters,
                        strict: t.strict.unwrap_or(false),
                    })
                })
                .collect()
        });

        let parallel_tool_calls = match &self.tool_choice {
            Some(AnthropicToolChoiceOwned::Any { .. }) => Some(true),
            _ => None,
        };

        let tool_choice = self.tool_choice.map(tool_choice_owned_to_internal);

        // --- Inference params ---
        let temperature = self.temperature;
        let max_tokens = self.max_tokens;
        let top_p = self.top_p;
        let stop_sequences = self.stop_sequences.clone();

        let chat_params = ChatCompletionInferenceParams {
            temperature,
            max_tokens: if max_tokens > 0 {
                Some(max_tokens)
            } else {
                None
            },
            top_p,
            stop_sequences: stop_sequences.clone(),
            ..Default::default()
        };

        // --- Metadata / TensorZero extension fields ---
        let meta = self.metadata;
        let tags = meta
            .clone()
            .and_then(|m| {
                let mut tags = if let Some(tz) = m.tensorzero {
                    tz.tags
                } else {
                    HashMap::new()
                };
                if let Some(user_id) = m.user_id {
                    tags.insert("tensorzero::user".to_string(), user_id);
                }
                if tags.is_empty() { None } else { Some(tags) }
            })
            .unwrap_or_default();

        let (
            episode_id,
            dryrun,
            cache_options,
            extra_body,
            extra_headers,
            namespace,
            credentials,
            params,
            include_raw_usage,
        ) = meta
            .and_then(|m| m.tensorzero)
            .map(|tz| {
                (
                    tz.episode_id,
                    tz.dryrun.unwrap_or(false),
                    tz.cache_options,
                    tz.extra_body,
                    tz.extra_headers,
                    tz.namespace,
                    tz.credentials,
                    tz.params,
                    tz.include_raw_usage.unwrap_or(false),
                )
            })
            .unwrap_or_else(|| {
                (
                    None,
                    false,
                    None,
                    UnfilteredInferenceExtraBody::default(),
                    UnfilteredInferenceExtraHeaders::default(),
                    None,
                    InferenceCredentials::default(),
                    None,
                    false,
                )
            });

        // --- Build internal Params ---
        let input = Input {
            system,
            messages: input_messages,
        };

        let dynamic_tool_params = crate::tool::DynamicToolParams {
            allowed_tools: None,
            additional_tools,
            tool_choice,
            parallel_tool_calls,
            provider_tools: Vec::new(),
        };

        Ok(crate::endpoints::inference::Params {
            function_name,
            model_name,
            episode_id,
            namespace,
            input,
            stream: if self.stream { Some(true) } else { None },
            params: params.map_or(
                InferenceParams {
                    chat_completion: chat_params,
                },
                |p| InferenceParams {
                    chat_completion: ChatCompletionInferenceParams {
                        temperature: p.chat_completion.temperature.or(temperature),
                        max_tokens: p.chat_completion.max_tokens.or(if max_tokens > 0 {
                            Some(max_tokens)
                        } else {
                            None
                        }),
                        top_p: p.chat_completion.top_p.or(top_p),
                        stop_sequences: p.chat_completion.stop_sequences.or(stop_sequences),
                        ..p.chat_completion
                    },
                },
            ),
            // variant_name is not set by the Anthropic-compatible ingress.
            // Tool choice does not map to variant name.
            variant_name: None,
            dryrun: if dryrun { Some(true) } else { None },
            internal: false,
            tags,
            dynamic_tool_params,
            output_schema: None,
            cache_options: cache_options.unwrap_or_default(),
            credentials,
            include_original_response: false,
            include_raw_response: false,
            include_raw_usage,
            include_aggregated_response: false,
            extra_body,
            extra_headers,
            internal_dynamic_variant_config: None,
            cache_control_spans: system_spans
                .into_iter()
                .map(|(block_idx, marker)| CacheControlSpan {
                    target: CacheControlTarget::SystemBlock { block_idx },
                    marker,
                })
                .chain(
                    message_spans
                        .into_iter()
                        .map(|(message_idx, content_idx, marker)| CacheControlSpan {
                            target: CacheControlTarget::MessageContent {
                                message_idx,
                                content_idx,
                            },
                            marker,
                        }),
                )
                .chain(
                    tool_spans
                        .into_iter()
                        .map(|(tool_idx, marker)| CacheControlSpan {
                            target: CacheControlTarget::Tool { tool_idx },
                            marker,
                        }),
                )
                .collect(),
        })
    }
}

/// Converts a list of Anthropic content blocks to TensorZero `InputMessageContent` items.
///
/// Also returns any per-block `cache_control` spans encountered, as `(content_idx, marker)`
/// pairs so the caller can build `CacheControlSpan` values with the correct `message_idx`.
#[expect(clippy::type_complexity)]
fn blocks_into_input_content(
    blocks: Vec<AnthropicContentBlockOwned>,
    tool_use_name_map: &HashMap<String, String>,
) -> Result<
    (
        Vec<InputMessageContent>,
        Vec<(usize, AnthropicCacheControl)>,
    ),
    Error,
> {
    let mut result = Vec::with_capacity(blocks.len());
    let mut spans = Vec::new();
    for (content_idx, block) in blocks.into_iter().enumerate() {
        match block {
            AnthropicContentBlockOwned::Text {
                text,
                cache_control,
            } => {
                if let Some(marker) = cache_control {
                    spans.push((content_idx, marker));
                }
                result.push(InputMessageContent::Text(Text { text }));
            }
            AnthropicContentBlockOwned::Thinking {
                thinking,
                signature,
            } => {
                result.push(InputMessageContent::Thought(Thought {
                    text: if thinking.is_empty() {
                        None
                    } else {
                        Some(thinking)
                    },
                    signature: if signature.is_empty() {
                        None
                    } else {
                        Some(signature)
                    },
                    summary: None,
                    provider_type: None,
                    extra_data: None,
                }));
            }
            AnthropicContentBlockOwned::RedactedThinking { data } => {
                // Pass through as unknown content block for provider-specific handling
                result.push(InputMessageContent::Unknown(Unknown {
                    data: Value::String(data),
                    model_name: None,
                    provider_name: None,
                }));
            }
            AnthropicContentBlockOwned::ToolUse {
                id,
                name,
                input,
                cache_control,
            } => {
                if let Some(marker) = cache_control {
                    spans.push((content_idx, marker));
                }
                result.push(InputMessageContent::ToolCall(ToolCallWrapper::ToolCall(
                    ToolCall {
                        id,
                        name,
                        arguments: input.to_string(),
                    },
                )));
            }
            AnthropicContentBlockOwned::ToolResult {
                tool_use_id,
                content,
                is_error: _,
                cache_control,
            } => {
                if let Some(marker) = cache_control {
                    spans.push((content_idx, marker));
                }
                // Concatenate tool result content blocks into a single string.
                // Error on unsupported sub-block types instead of silently dropping.
                let mut result_text = String::new();
                for block in content {
                    match block {
                        AnthropicContentBlockOwned::Text { text, .. } => {
                            if !result_text.is_empty() {
                                result_text.push('\n');
                            }
                            result_text.push_str(&text);
                        }
                        other => {
                            return Err(Error::new(ErrorDetails::InvalidRequest {
                                message: format!(
                                    "Unsupported content block type '{0}' inside tool_result; only Text blocks are supported",
                                    other.kind()
                                ),
                            }));
                        }
                    }
                }
                // Look up tool name from earlier tool_use blocks in the conversation
                let name = tool_use_name_map
                    .get(tool_use_id.as_str())
                    .cloned()
                    .unwrap_or_default();
                result.push(InputMessageContent::ToolResult(ToolResult {
                    id: tool_use_id.clone(),
                    name,
                    result: result_text,
                }));
            }
            AnthropicContentBlockOwned::Image {
                source,
                cache_control,
            } => {
                if let Some(marker) = cache_control {
                    spans.push((content_idx, marker));
                }
                let file = match source {
                    AnthropicImageSourceOwned::Base64 { media_type, data } => {
                        let mime_type: MediaType = media_type.parse().map_err(|_| {
                            Error::new(ErrorDetails::InvalidRequest {
                                message: format!(
                                    "Invalid MIME type in base64 image: `{media_type}`"
                                ),
                            })
                        })?;
                        File::Base64(
                            Base64File::new(None, Some(mime_type), data.clone(), None, None)
                                .map_err(|e| {
                                    Error::new(ErrorDetails::InvalidRequest {
                                        message: format!("Invalid base64 image data: {e}"),
                                    })
                                })?,
                        )
                    }
                    AnthropicImageSourceOwned::Url { url } => {
                        let parsed_url = Url::parse(&url).map_err(|_| {
                            Error::new(ErrorDetails::InvalidRequest {
                                message: format!("Invalid URL in image content block: `{url}`"),
                            })
                        })?;
                        File::Url(UrlFile {
                            url: parsed_url,
                            mime_type: None,
                            detail: None,
                            filename: None,
                        })
                    }
                };
                result.push(InputMessageContent::File(file));
            }
            AnthropicContentBlockOwned::Document {
                source,
                cache_control,
            } => {
                if let Some(marker) = cache_control {
                    spans.push((content_idx, marker));
                }
                let file = match source {
                    AnthropicDocumentSourceOwned::Base64 { media_type, data } => {
                        let mime_type: MediaType = media_type.parse().map_err(|_| {
                            Error::new(ErrorDetails::InvalidRequest {
                                message: format!(
                                    "Invalid MIME type in base64 document: `{media_type}`"
                                ),
                            })
                        })?;
                        File::Base64(
                            Base64File::new(None, Some(mime_type), data.clone(), None, None)
                                .map_err(|e| {
                                    Error::new(ErrorDetails::InvalidRequest {
                                        message: format!("Invalid base64 document data: {e}"),
                                    })
                                })?,
                        )
                    }
                    AnthropicDocumentSourceOwned::Url { url } => {
                        let parsed_url = Url::parse(&url).map_err(|_| {
                            Error::new(ErrorDetails::InvalidRequest {
                                message: format!("Invalid URL in document content block: `{url}`"),
                            })
                        })?;
                        File::Url(UrlFile {
                            url: parsed_url,
                            mime_type: None,
                            detail: None,
                            filename: None,
                        })
                    }
                };
                result.push(InputMessageContent::File(file));
            }
        }
    }
    Ok((result, spans))
}

/// Converts an owned Anthropic tool choice to the internal `ToolChoice`.
fn tool_choice_owned_to_internal(choice: AnthropicToolChoiceOwned) -> crate::tool::ToolChoice {
    match choice {
        AnthropicToolChoiceOwned::Auto { .. } => crate::tool::ToolChoice::Auto,
        AnthropicToolChoiceOwned::Any { .. } => crate::tool::ToolChoice::Required,
        AnthropicToolChoiceOwned::Tool { name, .. } => crate::tool::ToolChoice::Specific(name),
        AnthropicToolChoiceOwned::None { .. } => crate::tool::ToolChoice::None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensorzero_types::FinishReason;

    #[test]
    fn test_deserialize_simple_request() {
        let json = r#"{
            "model": "tensorzero::function_name::test_model",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100
        }"#;
        let params: AnthropicMessagesParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.model, "tensorzero::function_name::test_model");
        assert_eq!(params.messages.len(), 1);
        assert_eq!(params.max_tokens, 100);
    }

    #[test]
    fn test_deserialize_system_as_string() {
        let json = r#"{
            "model": "test",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 100,
            "system": "You are a helpful assistant"
        }"#;
        let params: AnthropicMessagesParams = serde_json::from_str(json).unwrap();
        match params.system {
            AnthropicSystem::String(s) => assert_eq!(s, "You are a helpful assistant"),
            AnthropicSystem::Blocks(_) => {
                panic!("Expected system string")
            }
        }
    }

    #[test]
    fn test_deserialize_content_blocks() {
        let json = r#"{
            "model": "test",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "tool_use", "id": "tu1", "name": "get_weather", "input": {"location": "NYC"}}
                ]
            }],
            "max_tokens": 100
        }"#;
        let params: AnthropicMessagesParams = serde_json::from_str(json).unwrap();
        match &params.messages[0].content {
            AnthropicMessageContentOwned::Blocks(blocks) => {
                assert_eq!(blocks.len(), 2);
                match &blocks[0] {
                    AnthropicContentBlockOwned::Text { text, .. } => assert_eq!(text, "Hello"),
                    other => panic!("Expected Text, got {}", other.kind()),
                }
                match &blocks[1] {
                    AnthropicContentBlockOwned::ToolUse { name, .. } => {
                        assert_eq!(name, "get_weather");
                    }
                    other => panic!("Expected ToolUse, got {}", other.kind()),
                }
            }
            AnthropicMessageContentOwned::String(_) => {
                panic!("Expected Blocks")
            }
        }
    }

    #[test]
    fn test_deserialize_tool_choice() {
        let json = r#"{
            "model": "test",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 100,
            "tool_choice": {"type": "tool", "name": "my_tool"}
        }"#;
        let params: AnthropicMessagesParams = serde_json::from_str(json).unwrap();
        match params.tool_choice {
            Some(AnthropicToolChoiceOwned::Tool { name, .. }) => assert_eq!(name, "my_tool"),
            _ => panic!("Expected Tool variant"),
        }
    }

    #[test]
    fn test_deserialize_thinking_block() {
        let json = r#"{
            "model": "test",
            "messages": [{
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Let me think", "signature": "abc123"},
                    {"type": "text", "text": "The answer is 42"}
                ]
            }],
            "max_tokens": 100
        }"#;
        let params: AnthropicMessagesParams = serde_json::from_str(json).unwrap();
        match &params.messages[0].content {
            AnthropicMessageContentOwned::Blocks(blocks) => {
                assert_eq!(blocks.len(), 2);
                match &blocks[0] {
                    AnthropicContentBlockOwned::Thinking {
                        thinking,
                        signature,
                    } => {
                        assert_eq!(thinking, "Let me think");
                        assert_eq!(signature, "abc123");
                    }
                    other => panic!("Expected Thinking, got {}", other.kind()),
                }
            }
            AnthropicMessageContentOwned::String(_) => {
                panic!("Expected Blocks")
            }
        }
    }

    #[test]
    fn test_deserialize_redacted_thinking() {
        let json = r#"{
            "model": "test",
            "messages": [{
                "role": "assistant",
                "content": [
                    {"type": "redacted_thinking", "data": "encrypted"}
                ]
            }],
            "max_tokens": 100
        }"#;
        let params: AnthropicMessagesParams = serde_json::from_str(json).unwrap();
        match &params.messages[0].content {
            AnthropicMessageContentOwned::Blocks(blocks) => match &blocks[0] {
                AnthropicContentBlockOwned::RedactedThinking { data } => {
                    assert_eq!(data, "encrypted");
                }
                other => panic!("Expected RedactedThinking, got {}", other.kind()),
            },
            AnthropicMessageContentOwned::String(_) => {
                panic!("Expected Blocks")
            }
        }
    }

    #[test]
    fn test_deserialize_metadata() {
        let json = r#"{
            "model": "test",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 100,
            "metadata": {
                "user_id": "user_42",
                "tensorzero": {
                    "episode_id": "00000000-0000-0000-0000-000000000001",
                    "tags": {"team": "infra"}
                }
            }
        }"#;
        let params: AnthropicMessagesParams = serde_json::from_str(json).unwrap();
        let meta = params.metadata.unwrap();
        assert_eq!(meta.user_id, Some("user_42".to_string()));
        let tz = meta.tensorzero.unwrap();
        assert_eq!(tz.tags.get("team"), Some(&"infra".to_string()));
    }

    #[test]
    fn test_tool_choice_string_deserialize() {
        let json = r#""auto""#;
        let choice: AnthropicToolChoiceOwned = serde_json::from_str(json).unwrap();
        assert!(matches!(choice, AnthropicToolChoiceOwned::Auto { .. }));

        let json = r#""any""#;
        let choice: AnthropicToolChoiceOwned = serde_json::from_str(json).unwrap();
        assert!(matches!(choice, AnthropicToolChoiceOwned::Any { .. }));

        let json = r#""none""#;
        let choice: AnthropicToolChoiceOwned = serde_json::from_str(json).unwrap();
        assert!(matches!(choice, AnthropicToolChoiceOwned::None { .. }));
    }

    #[test]
    fn test_tool_choice_object_deserialize() {
        let json = r#"{"type": "auto"}"#;
        let choice: AnthropicToolChoiceOwned = serde_json::from_str(json).unwrap();
        assert!(matches!(choice, AnthropicToolChoiceOwned::Auto { .. }));

        let json = r#"{"type": "any", "disable_parallel_tool_use": true}"#;
        let choice: AnthropicToolChoiceOwned = serde_json::from_str(json).unwrap();
        assert!(matches!(
            choice,
            AnthropicToolChoiceOwned::Any {
                disable_parallel_tool_use: Some(true)
            }
        ));
    }

    #[test]
    fn test_try_into_params_basic() {
        let json = r#"{
            "model": "tensorzero::function_name::my_function",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100
        }"#;
        let params: AnthropicMessagesParams = serde_json::from_str(json).unwrap();
        let internal = params.try_into_params().unwrap();

        assert_eq!(internal.function_name, Some("my_function".to_string()));
        assert!(internal.model_name.is_none());
        assert_eq!(internal.input.messages.len(), 1);
        assert!(matches!(internal.input.messages[0].role, Role::User));
        assert_eq!(internal.input.messages[0].content.len(), 1);
    }

    #[test]
    fn test_try_into_params_model_name() {
        let json = r#"{
            "model": "tensorzero::model_name::my_model",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100
        }"#;
        let params: AnthropicMessagesParams = serde_json::from_str(json).unwrap();
        let internal = params.try_into_params().unwrap();

        assert!(internal.function_name.is_none());
        assert_eq!(internal.model_name, Some("my_model".to_string()));
    }

    #[test]
    fn test_try_into_params_bare_model() {
        let json = r#"{
            "model": "my_function",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100
        }"#;
        let params: AnthropicMessagesParams = serde_json::from_str(json).unwrap();
        let internal = params.try_into_params().unwrap();

        assert_eq!(internal.function_name, Some("my_function".to_string()));
    }

    #[test]
    fn test_try_into_params_system_prompt() {
        let json = r#"{
            "model": "my_function",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
            "system": "You are helpful"
        }"#;
        let params: AnthropicMessagesParams = serde_json::from_str(json).unwrap();
        let internal = params.try_into_params().unwrap();

        match &internal.input.system {
            Some(System::Text(s)) => assert_eq!(s, "You are helpful"),
            _ => panic!("Expected text system prompt"),
        }
    }

    #[test]
    fn test_try_into_params_tools() {
        let json = r#"{
            "model": "my_function",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
            "tools": [{"name": "get_weather", "description": "Get weather", "input_schema": {"type": "object"}}]
        }"#;
        let params: AnthropicMessagesParams = serde_json::from_str(json).unwrap();
        let internal = params.try_into_params().unwrap();

        match &internal.dynamic_tool_params.additional_tools {
            Some(tools) => {
                assert_eq!(tools.len(), 1);
                match &tools[0] {
                    crate::tool::Tool::Function(f) => {
                        assert_eq!(f.name, "get_weather");
                    }
                    crate::tool::Tool::OpenAICustom(_) => {
                        panic!("Expected Function tool")
                    }
                }
            }
            None => panic!("Expected additional tools"),
        }
    }

    #[test]
    fn test_try_into_params_tool_choice() {
        let json = r#"{
            "model": "my_function",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
            "tool_choice": {"type": "tool", "name": "get_weather"}
        }"#;
        let params: AnthropicMessagesParams = serde_json::from_str(json).unwrap();
        let internal = params.try_into_params().unwrap();

        match &internal.dynamic_tool_params.tool_choice {
            Some(crate::tool::ToolChoice::Specific(name)) => assert_eq!(name, "get_weather"),
            _ => panic!("Expected Specific tool choice"),
        }
    }

    #[test]
    fn test_try_into_params_metadata_tags() {
        let json = r#"{
            "model": "my_function",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
            "metadata": {
                "user_id": "user_42",
                "tensorzero": {
                    "tags": {"team": "infra"},
                    "episode_id": "00000000-0000-0000-0000-000000000001"
                }
            }
        }"#;
        let params: AnthropicMessagesParams = serde_json::from_str(json).unwrap();
        let internal = params.try_into_params().unwrap();

        assert_eq!(
            internal.tags.get("tensorzero::user"),
            Some(&"user_42".to_string())
        );
        assert_eq!(internal.tags.get("team"), Some(&"infra".to_string()));
        assert_eq!(internal.episode_id, Some(uuid::Uuid::from_u128(1)));
    }

    #[test]
    fn test_try_into_params_empty_model() {
        let json = r#"{
            "model": "",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100
        }"#;
        let params: AnthropicMessagesParams = serde_json::from_str(json).unwrap();
        let result = params.try_into_params();
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("model is required")
        );
    }

    #[test]
    fn test_try_into_params_empty_messages() {
        let json = r#"{
            "model": "my_function",
            "messages": [],
            "max_tokens": 100
        }"#;
        let params: AnthropicMessagesParams = serde_json::from_str(json).unwrap();
        let result = params.try_into_params();
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("messages is required")
        );
    }

    #[test]
    fn test_try_into_params_temperature_and_max_tokens() {
        let json = r#"{
            "model": "my_function",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 200,
            "temperature": 0.7,
            "top_p": 0.9
        }"#;
        let params: AnthropicMessagesParams = serde_json::from_str(json).unwrap();
        let internal = params.try_into_params().unwrap();

        assert_eq!(internal.params.chat_completion.temperature, Some(0.7));
        assert_eq!(internal.params.chat_completion.max_tokens, Some(200));
        assert_eq!(internal.params.chat_completion.top_p, Some(0.9));
    }

    #[test]
    fn test_try_into_params_thinking_blocks() {
        let json = r#"{
            "model": "my_function",
            "messages": [{
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Let me think", "signature": "sig123"}
                ]
            }],
            "max_tokens": 100
        }"#;
        let params: AnthropicMessagesParams = serde_json::from_str(json).unwrap();
        let internal = params.try_into_params().unwrap();

        assert_eq!(internal.input.messages.len(), 1);
        assert!(matches!(internal.input.messages[0].role, Role::Assistant));
        match &internal.input.messages[0].content[0] {
            InputMessageContent::Thought(thought) => {
                assert_eq!(thought.text, Some("Let me think".to_string()));
                assert_eq!(thought.signature, Some("sig123".to_string()));
            }
            other => panic!("Expected Thought, got {other:?}"),
        }
    }

    #[test]
    fn test_try_into_params_redacted_thinking() {
        let json = r#"{
            "model": "my_function",
            "messages": [{
                "role": "assistant",
                "content": [
                    {"type": "redacted_thinking", "data": "encrypted"}
                ]
            }],
            "max_tokens": 100
        }"#;
        let params: AnthropicMessagesParams = serde_json::from_str(json).unwrap();
        let internal = params.try_into_params().unwrap();

        match &internal.input.messages[0].content[0] {
            InputMessageContent::Unknown(u) => {
                assert_eq!(u.data, serde_json::Value::String("encrypted".to_string()));
            }
            other => panic!("Expected Unknown, got {other:?}"),
        }
    }

    #[test]
    fn test_try_into_params_tool_use_blocks() {
        let json = r#"{
            "model": "my_function",
            "messages": [{
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me call a tool"},
                    {"type": "tool_use", "id": "tu1", "name": "get_weather", "input": {"location": "NYC"}}
                ]
            }],
            "max_tokens": 100
        }"#;
        let params: AnthropicMessagesParams = serde_json::from_str(json).unwrap();
        let internal = params.try_into_params().unwrap();

        assert_eq!(internal.input.messages[0].content.len(), 2);
        match &internal.input.messages[0].content[0] {
            InputMessageContent::Text(t) => assert_eq!(t.text, "Let me call a tool"),
            other => panic!("Expected Text, got {other:?}"),
        }
        match &internal.input.messages[0].content[1] {
            InputMessageContent::ToolCall(wrapper) => {
                let tc = wrapper.clone().into_tool_call();
                assert_eq!(tc.id, "tu1");
                assert_eq!(tc.name, "get_weather");
                assert_eq!(tc.arguments, r#"{"location":"NYC"}"#);
            }
            other => panic!("Expected ToolCall, got {other:?}"),
        }
    }

    #[test]
    fn test_try_into_params_tool_result_resolves_name() {
        // Simulates a full conversation: assistant calls tool, user returns result.
        // The tool_use_id in the result should resolve to the tool name from the
        // earlier tool_use block.
        let json = r#"{
            "model": "my_function",
            "messages": [
                {"role": "user", "content": "What's the weather in NYC?"},
                {"role": "assistant", "content": [{"type": "tool_use", "id": "tu1", "name": "get_weather", "input": {"location": "NYC"}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu1", "content": "Sunny, 72F"}]}
            ],
            "max_tokens": 100
        }"#;
        let params: AnthropicMessagesParams = serde_json::from_str(json).unwrap();
        let internal = params.try_into_params().unwrap();

        // Message 0: user text
        assert_eq!(internal.input.messages[0].content.len(), 1);
        // Message 1: assistant tool_use
        assert_eq!(internal.input.messages[1].content.len(), 1);
        // Message 2: user tool_result — should have resolved name
        match &internal.input.messages[2].content[0] {
            InputMessageContent::ToolResult(tr) => {
                assert_eq!(tr.id, "tu1");
                assert_eq!(tr.name, "get_weather");
                assert_eq!(tr.result, "Sunny, 72F");
            }
            other => panic!("Expected ToolResult, got {other:?}"),
        }
    }

    #[test]
    fn test_try_into_params_stream_flag() {
        let json = r#"{
            "model": "my_function",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
            "stream": true
        }"#;
        let params: AnthropicMessagesParams = serde_json::from_str(json).unwrap();
        let internal = params.try_into_params().unwrap();

        assert_eq!(internal.stream, Some(true));
    }

    #[test]
    fn test_try_into_params_tool_result_non_text_error() {
        // Non-text sub-blocks inside tool_result should return an error,
        // not be silently dropped.
        let json = r#"{
            "model": "my_function",
            "messages": [
                {"role": "user", "content": "Check this"},
                {"role": "assistant", "content": [{"type": "tool_use", "id": "tu1", "name": "analyze", "input": {"field": "value"}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu1", "content": [{"type": "image", "source": {"type": "base64", "data": "abc", "media_type": "image/png"}}]}]}
            ],
            "max_tokens": 100
        }"#;
        let params: AnthropicMessagesParams = serde_json::from_str(json).unwrap();
        let result = params.try_into_params();
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Unsupported content block type 'image' inside tool_result")
        );
    }

    #[test]
    fn test_try_into_params_stop_sequences() {
        let json = r#"{
            "model": "my_function",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
            "stop_sequences": ["\n\n", "END"]
        }"#;
        let params: AnthropicMessagesParams = serde_json::from_str(json).unwrap();
        let internal = params.try_into_params().unwrap();

        assert_eq!(
            internal.params.chat_completion.stop_sequences,
            Some(vec!["\n\n".to_string(), "END".to_string()])
        );
    }

    #[test]
    fn test_try_into_params_image_content_block_base64() {
        let json = r#"{
            "model": "my_function",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "data": "abc", "media_type": "image/png"}}
                ]
            }],
            "max_tokens": 100
        }"#;
        let params: AnthropicMessagesParams = serde_json::from_str(json).unwrap();
        let result = params.try_into_params().unwrap();
        let msg = &result.input.messages[0];
        assert_eq!(msg.content.len(), 1);
        match &msg.content[0] {
            InputMessageContent::File(file) => match file {
                File::Base64(b64) => {
                    assert_eq!(b64.mime_type.to_string(), "image/png");
                    assert_eq!(b64.data(), "abc");
                }
                other => panic!("Expected File::Base64, got {other:?}"),
            },
            other => panic!("Expected File variant, got {other:?}"),
        }
    }

    #[test]
    fn test_try_into_params_image_content_block_url() {
        let json = r#"{
            "model": "my_function",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "url", "url": "https://example.com/image.png"}}
                ]
            }],
            "max_tokens": 100
        }"#;
        let params: AnthropicMessagesParams = serde_json::from_str(json).unwrap();
        let result = params.try_into_params().unwrap();
        let msg = &result.input.messages[0];
        assert_eq!(msg.content.len(), 1);
        match &msg.content[0] {
            InputMessageContent::File(file) => match file {
                File::Url(url_file) => {
                    assert_eq!(url_file.url.as_str(), "https://example.com/image.png");
                }
                other => panic!("Expected File::Url, got {other:?}"),
            },
            other => panic!("Expected File variant, got {other:?}"),
        }
    }

    #[test]
    fn test_try_into_params_document_content_block_base64() {
        let json = r#"{
            "model": "my_function",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "document", "source": {"type": "base64", "data": "def", "media_type": "application/pdf"}}
                ]
            }],
            "max_tokens": 100
        }"#;
        let params: AnthropicMessagesParams = serde_json::from_str(json).unwrap();
        let result = params.try_into_params().unwrap();
        let msg = &result.input.messages[0];
        assert_eq!(msg.content.len(), 1);
        match &msg.content[0] {
            InputMessageContent::File(file) => match file {
                File::Base64(b64) => {
                    assert_eq!(b64.mime_type.to_string(), "application/pdf");
                    assert_eq!(b64.data(), "def");
                }
                other => panic!("Expected File::Base64, got {other:?}"),
            },
            other => panic!("Expected File variant, got {other:?}"),
        }
    }

    #[test]
    fn test_try_into_params_document_content_block_url() {
        let json = r#"{
            "model": "my_function",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "document", "source": {"type": "url", "url": "https://example.com/doc.pdf"}}
                ]
            }],
            "max_tokens": 100
        }"#;
        let params: AnthropicMessagesParams = serde_json::from_str(json).unwrap();
        let result = params.try_into_params().unwrap();
        let msg = &result.input.messages[0];
        assert_eq!(msg.content.len(), 1);
        match &msg.content[0] {
            InputMessageContent::File(file) => match file {
                File::Url(url_file) => {
                    assert_eq!(url_file.url.as_str(), "https://example.com/doc.pdf");
                }
                other => panic!("Expected File::Url, got {other:?}"),
            },
            other => panic!("Expected File variant, got {other:?}"),
        }
    }

    #[test]
    fn test_inference_response_to_anthropic_stop_reason() {
        use crate::endpoints::inference::{InferenceOutput, InferenceResponse};
        use tensorzero_types::{
            ChatInferenceResponse, ContentBlockChatOutput, FinishReason, Text, Usage,
        };
        use tensorzero_types_providers::anthropic::AnthropicStopReason;

        let build_output = |finish_reason: Option<FinishReason>| {
            InferenceOutput::NonStreaming(InferenceResponse::Chat(ChatInferenceResponse {
                inference_id: uuid::Uuid::default(),
                episode_id: uuid::Uuid::default(),
                variant_name: "test_variant".to_string(),
                content: vec![ContentBlockChatOutput::Text(Text {
                    text: "Hello".to_string(),
                })],
                usage: Usage::default(),
                raw_usage: None,
                original_response: None,
                raw_response: None,
                finish_reason,
            }))
        };

        let model_prefix = "tensorzero::function_name::my_function::variant_name::";

        // Stop → EndTurn
        let resp =
            inference_response_to_anthropic(build_output(Some(FinishReason::Stop)), model_prefix)
                .unwrap();
        assert_eq!(resp.stop_reason, Some(AnthropicStopReason::EndTurn));

        // Length → MaxTokens
        let resp =
            inference_response_to_anthropic(build_output(Some(FinishReason::Length)), model_prefix)
                .unwrap();
        assert_eq!(resp.stop_reason, Some(AnthropicStopReason::MaxTokens));

        // ToolCall → ToolUse
        let resp = inference_response_to_anthropic(
            build_output(Some(FinishReason::ToolCall)),
            model_prefix,
        )
        .unwrap();
        assert_eq!(resp.stop_reason, Some(AnthropicStopReason::ToolUse));

        // StopSequence → StopSequence
        let resp = inference_response_to_anthropic(
            build_output(Some(FinishReason::StopSequence)),
            model_prefix,
        )
        .unwrap();
        assert_eq!(resp.stop_reason, Some(AnthropicStopReason::StopSequence));

        // ContentFilter → None
        let resp = inference_response_to_anthropic(
            build_output(Some(FinishReason::ContentFilter)),
            model_prefix,
        )
        .unwrap();
        assert_eq!(resp.stop_reason, None);

        // Unknown → None
        let resp = inference_response_to_anthropic(
            build_output(Some(FinishReason::Unknown)),
            model_prefix,
        )
        .unwrap();
        assert_eq!(resp.stop_reason, None);

        // None → None
        let resp = inference_response_to_anthropic(build_output(None), model_prefix).unwrap();
        assert_eq!(resp.stop_reason, None);
    }

    #[test]
    fn test_inference_response_to_anthropic_unknown_block_handling() {
        use crate::endpoints::inference::{InferenceOutput, InferenceResponse};
        use tensorzero_types::{ChatInferenceResponse, Unknown};

        let build_output = |content: Vec<ContentBlockChatOutput>| {
            InferenceOutput::NonStreaming(InferenceResponse::Chat(ChatInferenceResponse {
                inference_id: uuid::Uuid::default(),
                episode_id: uuid::Uuid::default(),
                variant_name: "test_variant".to_string(),
                content,
                usage: Usage::default(),
                raw_usage: None,
                original_response: None,
                raw_response: None,
                finish_reason: Some(FinishReason::Stop),
            }))
        };

        let model_prefix = "tensorzero::function_name::my_function::variant_name::";

        // String data Unknown → RedactedThinking
        let resp = inference_response_to_anthropic(
            build_output(vec![ContentBlockChatOutput::Unknown(Unknown {
                data: Value::String("encrypted-data-here".to_string()),
                model_name: None,
                provider_name: None,
            })]),
            model_prefix,
        )
        .unwrap();
        assert_eq!(resp.content.len(), 1);
        match &resp.content[0] {
            AnthropicResponseContentBlock::RedactedThinking { data } => {
                assert_eq!(data, "encrypted-data-here");
            }
            other => panic!("Expected RedactedThinking, got {other:?}"),
        }

        // Object data Unknown → Text (not RedactedThinking)
        let resp = inference_response_to_anthropic(
            build_output(vec![ContentBlockChatOutput::Unknown(Unknown {
                data: Value::Object(serde_json::Map::from_iter([
                    ("type".to_string(), Value::String("tool_use".to_string())),
                    ("id".to_string(), Value::String("tool_123".to_string())),
                ])),
                model_name: None,
                provider_name: None,
            })]),
            model_prefix,
        )
        .unwrap();
        assert_eq!(resp.content.len(), 1);
        match &resp.content[0] {
            AnthropicResponseContentBlock::Text { text, .. } => {
                assert!(text.contains("tool_use"));
            }
            other => panic!("Expected Text, got {other:?}"),
        }
    }

    #[test]
    fn test_inference_response_to_anthropic_json_with_raw() {
        use crate::endpoints::inference::{InferenceOutput, InferenceResponse};
        use tensorzero_types::{JsonInferenceOutput, JsonInferenceResponse, Usage};

        let response = InferenceResponse::Json(JsonInferenceResponse {
            inference_id: Uuid::nil(),
            episode_id: Uuid::nil(),
            variant_name: "test_variant".to_string(),
            output: JsonInferenceOutput {
                raw: Some(r#"{"key":"value"}"#.to_string()),
                parsed: Some(serde_json::json!({"key": "value"})),
            },
            usage: Usage::default(),
            raw_usage: None,
            original_response: None,
            raw_response: None,
            finish_reason: None,
        });
        let output = InferenceOutput::NonStreaming(response);
        let resp = inference_response_to_anthropic(output, "tensorzero::").unwrap();

        assert_eq!(resp.content.len(), 1);
        match &resp.content[0] {
            AnthropicResponseContentBlock::Text { text, .. } => {
                assert_eq!(text, r#"{"key":"value"}"#);
            }
            other => panic!("Expected Text, got {other:?}"),
        }
        assert_eq!(resp.model, "tensorzero::test_variant");
        assert!(resp.stop_reason.is_none());
    }

    #[test]
    fn test_try_into_params_cache_control_spans() {
        let json = r#"{
            "model": "my_function",
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "Hello", "cache_control": {"type": "ephemeral"}},
                    {"type": "image", "source": {"type": "base64", "data": "abc", "media_type": "image/png"}, "cache_control": {"type": "ephemeral"}}
                ]},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "tu1", "name": "get_weather", "input": {}, "cache_control": {"type": "ephemeral"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "tu1", "content": "sunny", "cache_control": {"type": "ephemeral"}}
                ]}
            ],
            "max_tokens": 100,
            "system": [
                {"type": "text", "text": "Sys1", "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "Sys2"}
            ],
            "tools": [
                {"name": "get_weather", "description": "Get weather", "input_schema": {"type": "object"}, "cache_control": {"type": "ephemeral"}}
            ]
        }"#;
        let params: AnthropicMessagesParams = serde_json::from_str(json).unwrap();
        let internal = params.try_into_params().unwrap();

        let spans = internal.cache_control_spans;
        assert_eq!(
            spans.len(),
            6,
            "expected 6 cache_control spans, got {spans:?}"
        );

        // system block 0
        assert!(
            spans
                .iter()
                .any(|s| matches!(s.target, CacheControlTarget::SystemBlock { block_idx: 0 }))
        );
        // message 0, content 0 (text)
        assert!(spans.iter().any(|s| matches!(
            s.target,
            CacheControlTarget::MessageContent {
                message_idx: 0,
                content_idx: 0
            }
        )));
        // message 0, content 1 (image)
        assert!(spans.iter().any(|s| matches!(
            s.target,
            CacheControlTarget::MessageContent {
                message_idx: 0,
                content_idx: 1
            }
        )));
        // message 1, content 0 (tool_use)
        assert!(spans.iter().any(|s| matches!(
            s.target,
            CacheControlTarget::MessageContent {
                message_idx: 1,
                content_idx: 0
            }
        )));
        // message 2, content 0 (tool_result)
        assert!(spans.iter().any(|s| matches!(
            s.target,
            CacheControlTarget::MessageContent {
                message_idx: 2,
                content_idx: 0
            }
        )));
        // tool 0
        assert!(
            spans
                .iter()
                .any(|s| matches!(s.target, CacheControlTarget::Tool { tool_idx: 0 }))
        );
    }

    #[test]
    fn test_inference_response_to_anthropic_json_with_parsed() {
        use crate::endpoints::inference::{InferenceOutput, InferenceResponse};
        use tensorzero_types::{JsonInferenceOutput, JsonInferenceResponse, Usage};

        let response = InferenceResponse::Json(JsonInferenceResponse {
            inference_id: Uuid::nil(),
            episode_id: Uuid::nil(),
            variant_name: "json_func".to_string(),
            output: JsonInferenceOutput {
                raw: None,
                parsed: Some(serde_json::json!({"count": 42})),
            },
            usage: Usage::default(),
            raw_usage: None,
            original_response: None,
            raw_response: None,
            finish_reason: None,
        });
        let output = InferenceOutput::NonStreaming(response);
        let resp = inference_response_to_anthropic(output, "").unwrap();

        assert_eq!(resp.content.len(), 1);
        match &resp.content[0] {
            AnthropicResponseContentBlock::Text { text, .. } => {
                assert!(text.contains("42"));
            }
            other => panic!("Expected Text, got {other:?}"),
        }
        assert_eq!(resp.model, "json_func");
    }
}
