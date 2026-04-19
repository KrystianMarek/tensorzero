//! Wire format types for the Anthropic API.
//!
//! This module contains both types shared between the outbound Anthropic provider
//! and the new inbound (ingress) Anthropic-compatible endpoint.
//!
//! Types without lifetimes that are used by both outbound and inbound code live here.
//! Types with borrowed lifetimes that are outbound-only remain in `providers/anthropic.rs`.

use serde::{Deserialize, Serialize};

/// Usage statistics for an Anthropic API response.
///
/// Anthropic reports cache tokens separately from input_tokens.
/// When converting to TensorZero's internal `Usage`, add cache tokens back
/// to get the total input token count.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct AnthropicUsage {
    #[serde(default)]
    pub input_tokens: Option<u32>,
    #[serde(default)]
    pub output_tokens: Option<u32>,
    /// Number of input tokens used to create a new cache entry
    #[serde(default)]
    pub cache_creation_input_tokens: Option<u32>,
    /// Number of input tokens read from cache
    #[serde(default)]
    pub cache_read_input_tokens: Option<u32>,
}

impl AnthropicUsage {
    /// Converts Anthropic usage to TensorZero's internal `Usage`.
    ///
    /// Anthropic reports cache tokens separately from input_tokens.
    /// We add them back to get the total input token count.
    pub fn into_usage(self) -> tensorzero_types::Usage {
        let total_input_tokens = match (
            self.input_tokens,
            self.cache_creation_input_tokens,
            self.cache_read_input_tokens,
        ) {
            (Some(input), Some(cache_write), Some(cache_read)) => {
                Some(input + cache_write + cache_read)
            }
            (Some(input), Some(cache_write), None) => Some(input + cache_write),
            (Some(input), None, Some(cache_read)) => Some(input + cache_read),
            (Some(input), None, None) => Some(input),
            (None, Some(cache_write), Some(cache_read)) => Some(cache_write + cache_read),
            (None, Some(cache_write), None) => Some(cache_write),
            (None, None, Some(cache_read)) => Some(cache_read),
            (None, None, None) => None,
        };

        tensorzero_types::Usage {
            input_tokens: total_input_tokens.or(self.input_tokens),
            output_tokens: self.output_tokens,
            provider_cache_read_input_tokens: self.cache_read_input_tokens,
            provider_cache_write_input_tokens: self.cache_creation_input_tokens,
            cost: None,
        }
    }
}

/// The reason the model stopped generating.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AnthropicStopReason {
    EndTurn,
    MaxTokens,
    StopSequence,
    ToolUse,
    #[serde(other)]
    Unknown,
}

impl AnthropicStopReason {
    /// Converts to TensorZero's internal `FinishReason`.
    pub fn into_finish_reason(self) -> tensorzero_types::FinishReason {
        match self {
            AnthropicStopReason::EndTurn => tensorzero_types::FinishReason::Stop,
            AnthropicStopReason::MaxTokens => tensorzero_types::FinishReason::Length,
            AnthropicStopReason::StopSequence => tensorzero_types::FinishReason::StopSequence,
            AnthropicStopReason::ToolUse => tensorzero_types::FinishReason::ToolCall,
            AnthropicStopReason::Unknown => tensorzero_types::FinishReason::Unknown,
        }
    }
}
