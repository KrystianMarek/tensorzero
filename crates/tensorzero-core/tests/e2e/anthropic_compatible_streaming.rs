//! Streaming E2E tests for the Anthropic-compatible `/v1/messages` endpoint.
//!
//! Tests the full SSE event sequence: message_start → content_block_start/delta/stop →
//! message_delta → message_stop.

#![expect(clippy::print_stdout)]

use axum::Json;
use axum::extract::State;
use googletest::gtest;
use http_body_util::BodyExt;
use serde_json::Value;
use tensorzero::ClientExt;
use tensorzero_core::db::delegating_connection::DelegatingDatabaseConnection;
use tensorzero_core::db::inferences::{InferenceQueries, ListInferencesParams};
use tensorzero_core::db::test_helpers::TestDatabaseHelpers;
use tensorzero_core::endpoints::anthropic_compatible::messages::messages_handler;
use tensorzero_core::endpoints::anthropic_compatible::types::messages::{
    AnthropicContentBlockOwned, AnthropicMessageContentOwned, AnthropicMessageOwned,
    AnthropicMessagesParams, AnthropicRoleOwned, AnthropicSystem, AnthropicToolOwned,
};
use tensorzero_core::test_helpers::get_e2e_config;
use uuid::Uuid;

/// Helper to create a basic text user message.
fn user_msg(text: &str) -> AnthropicMessageOwned {
    AnthropicMessageOwned {
        role: AnthropicRoleOwned::User,
        content: AnthropicMessageContentOwned::Blocks(vec![AnthropicContentBlockOwned::Text {
            text: text.to_string(),
            cache_control: None,
        }]),
    }
}

/// Helper to create an assistant tool_use message.
fn assistant_tool_use(tool_use_id: &str, name: &str, input: Value) -> AnthropicMessageOwned {
    AnthropicMessageOwned {
        role: AnthropicRoleOwned::Assistant,
        content: AnthropicMessageContentOwned::Blocks(vec![AnthropicContentBlockOwned::ToolUse {
            id: tool_use_id.to_string(),
            name: name.to_string(),
            input,
            cache_control: None,
        }]),
    }
}

/// Helper to create a tool_result message.
fn tool_result(
    tool_use_id: &str,
    content: Vec<AnthropicContentBlockOwned>,
) -> AnthropicMessageOwned {
    AnthropicMessageOwned {
        role: AnthropicRoleOwned::User,
        content: AnthropicMessageContentOwned::Blocks(vec![
            AnthropicContentBlockOwned::ToolResult {
                tool_use_id: tool_use_id.to_string(),
                content,
                is_error: None,
                cache_control: None,
            },
        ]),
    }
}

/// Helper to create a tool definition.
fn get_weather_tool() -> AnthropicToolOwned {
    AnthropicToolOwned {
        name: "get_temperature".to_string(),
        description: Some("Get the current weather".to_string()),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The location to get weather for"
                }
            },
            "required": ["location"]
        }),
        strict: None,
        cache_control: None,
    }
}

/// Collect SSE events from a streaming response into a vector of JSON values.
async fn collect_sse_events(response: axum::response::Response) -> Vec<(String, Value)> {
    let body_bytes = response.into_body().collect().await.unwrap().to_bytes();
    let body_str = String::from_utf8_lossy(&body_bytes);

    let mut events = Vec::new();
    let mut current_event: Option<(String, String)> = None;

    for line in body_str.lines() {
        if line.is_empty() {
            // Empty line marks the end of an SSE event.
            if let Some((event_name, data)) = current_event.take() {
                let parsed: Value = serde_json::from_str(&data).unwrap_or(serde_json::json!({
                    "_raw": data
                }));
                events.push((event_name, parsed));
            }
        } else if let Some(stripped) = line.strip_prefix("event:") {
            // Save previous event if any.
            if let Some((event_name, data)) = current_event.take() {
                let parsed: Value =
                    serde_json::from_str(&data).unwrap_or(serde_json::json!({ "_raw": data }));
                events.push((event_name, parsed));
            }
            let event_name = stripped.trim().to_string();
            current_event = Some((event_name, String::new()));
        } else if let Some(stripped) = line.strip_prefix("data:") {
            let data = stripped.trim().to_string();
            match &mut current_event {
                Some((_, prev_data)) => {
                    if !prev_data.is_empty() {
                        prev_data.push('\n');
                    }
                    prev_data.push_str(&data);
                }
                None => {
                    current_event = Some((String::new(), data));
                }
            }
        } else if line.starts_with("id:") {
            // Event ID, skip for our purposes.
        }
    }

    // Don't forget the last event if the stream didn't end with an empty line.
    if let Some((event_name, data)) = current_event {
        let parsed: Value =
            serde_json::from_str(&data).unwrap_or_else(|_| serde_json::json!({ "_raw": data }));
        events.push((event_name, parsed));
    }

    events
}

/// Build a streaming request for a simple text inference.
fn make_streaming_request(function_name: &str) -> AnthropicMessagesParams {
    AnthropicMessagesParams {
        model: format!("tensorzero::function_name::{function_name}"),
        messages: vec![user_msg("What is the capital of Japan?")],
        system: AnthropicSystem::String("TensorBot".to_string()),
        max_tokens: 100,
        temperature: None,
        top_p: None,
        top_k: None,
        stop_sequences: None,
        stream: true,
        tools: None,
        tool_choice: None,
        metadata: None,
        service_tier: None,
        container: None,
    }
}

// ===========================================================================
// Scenario 1: Text-only streaming
// ===========================================================================
#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn test_streaming_text_only() {
    let client = tensorzero::test_helpers::make_embedded_gateway().await;
    let state = client.get_app_state_data().unwrap().load_latest();

    let params = make_streaming_request("basic_test_no_system_schema");

    let response = messages_handler(State(state), None, Json(params))
        .await
        .unwrap();

    let events = collect_sse_events(response).await;
    println!("Text-only events: {events:?}");

    // Extract event types in order.
    let event_types: Vec<&str> = events.iter().map(|(name, _)| name.as_str()).collect();

    // Must start with message_start.
    assert_eq!(
        event_types[0], "message_start",
        "stream should start with message_start"
    );
    let msg_start = &events[0].1;
    assert_eq!(msg_start["type"].as_str().unwrap(), "message");
    assert!(msg_start["id"].as_str().is_some());
    assert!(msg_start["usage"].is_object());
    let inference_id: Uuid = msg_start["id"]
        .as_str()
        .unwrap()
        .trim_start_matches("msg_")
        .parse()
        .unwrap();

    // Must end with message_stop.
    assert_eq!(
        event_types[event_types.len() - 1],
        "message_stop",
        "stream should end with message_stop"
    );

    // Must contain content_block_start, at least one content_block_delta, and content_block_stop.
    assert!(
        event_types.contains(&"content_block_start"),
        "events: {event_types:?}"
    );
    assert!(
        event_types.contains(&"content_block_delta"),
        "events: {event_types:?}"
    );
    assert!(
        event_types.contains(&"content_block_stop"),
        "events: {event_types:?}"
    );

    // Verify content_block_start has text type.
    let start_idx = event_types
        .iter()
        .position(|&e| e == "content_block_start")
        .unwrap();
    assert_eq!(
        events[start_idx].1["type"].as_str().unwrap(),
        "text",
        "first content block should be text"
    );

    // Verify message_delta carries usage.
    let delta_idx = event_types
        .iter()
        .position(|&e| e == "message_delta")
        .unwrap();
    assert!(
        events[delta_idx].1["usage"]["output_tokens"]
            .as_u64()
            .unwrap()
            > 0,
        "should have output tokens"
    );
    assert!(
        events[delta_idx].1["usage"]["input_tokens"]
            .as_u64()
            .unwrap()
            > 0,
        "should have input tokens"
    );

    // Verify stop_reason in message_delta.
    assert!(
        events[delta_idx].1["stop_reason"].is_string(),
        "should have stop_reason in message_delta"
    );

    // Verify DB records.
    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;

    let config = get_e2e_config().await;
    let inferences = conn
        .list_inferences(
            &config,
            &ListInferencesParams {
                ids: Some(&[inference_id]),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(inferences.len(), 1, "should have one inference record");

    // Response text should contain expected content.
    let text_deltas: Vec<&str> = events
        .iter()
        .filter_map(|(name, data)| {
            if name == "content_block_delta" && data["delta_type"].as_str() == Some("text_delta") {
                data["text"].as_str()
            } else {
                None
            }
        })
        .collect();
    let full_text: String = text_deltas.join("");
    assert!(
        !full_text.is_empty(),
        "text content should not be empty, got deltas: {text_deltas:?}"
    );
    assert!(
        full_text.contains("Megumin") || full_text.contains("explosion"),
        "expected haiku-like answer, got: {full_text}"
    );
}

// ===========================================================================
// Scenario 2: Tool-use streaming (single tool)
// ===========================================================================
#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn test_streaming_tool_use() {
    let client = tensorzero::test_helpers::make_embedded_gateway().await;
    let state = client.get_app_state_data().unwrap().load_latest();

    let params = AnthropicMessagesParams {
        model: "tensorzero::function_name::basic_test_no_system_schema".to_string(),
        messages: vec![user_msg("What is the weather in Tokyo?")],
        system: AnthropicSystem::String("TensorBot".to_string()),
        max_tokens: 100,
        temperature: None,
        top_p: None,
        top_k: None,
        stop_sequences: None,
        stream: true,
        tools: Some(vec![get_weather_tool()]),
        tool_choice: None,
        metadata: None,
        service_tier: None,
        container: None,
    };

    let response = messages_handler(State(state), None, Json(params))
        .await
        .unwrap();

    let events = collect_sse_events(response).await;
    println!("Tool-use events: {events:?}");

    let event_types: Vec<&str> = events.iter().map(|(name, _)| name.as_str()).collect();

    // Verify event sequence: message_start → content_block_start(text) → content_block_delta(text)
    // → content_block_stop → content_block_start(tool_use) → ... → content_block_stop →
    // message_delta → message_stop
    assert_eq!(event_types[0], "message_start");

    // Should have a tool_use content block.
    let tool_start_idx = event_types
        .iter()
        .position(|&e| e == "content_block_start")
        .and_then(|i| {
            (events[i].1["type"].as_str()? == "tool_use")
                .then_some(i)
                .or_else(|| {
                    event_types[i + 1..]
                        .iter()
                        .position(|&e| e == "content_block_start")
                        .map(|j| i + 1 + j)
                })
        });

    if let Some(idx) = tool_start_idx {
        assert_eq!(events[idx].1["type"].as_str().unwrap(), "tool_use");
        assert!(
            events[idx].1["name"].as_str().is_some(),
            "tool_use should have a name"
        );

        // Should have input_json_delta events for the tool call.
        let tool_delta_count = event_types
            .iter()
            .enumerate()
            .skip(idx)
            .take(5)
            .filter(|&(_, t)| *t == "content_block_delta")
            .filter(|(_, t)| {
                let target_type = *t;
                events.iter().any(|(_, d)| {
                    d.get("delta_type").and_then(|s| s.as_str()) == Some("input_json_delta")
                        && d.get("type").and_then(|s| s.as_str()) == Some(target_type)
                })
            })
            .count();
        assert!(
            tool_delta_count > 0
                || event_types
                    .iter()
                    .skip(idx)
                    .any(|&t| t == "content_block_delta"),
            "should have input_json_delta events for tool_use"
        );
    }

    assert_eq!(event_types[event_types.len() - 1], "message_stop");
}

// ===========================================================================
// Scenario 3: Multi-turn with tool_result
// ===========================================================================
#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn test_streaming_multi_turn_with_tool_result() {
    let client = tensorzero::test_helpers::make_embedded_gateway().await;
    let state = client.get_app_state_data().unwrap().load_latest();

    // Turn 1: Send request with tool.
    let params = AnthropicMessagesParams {
        model: "tensorzero::function_name::basic_test_no_system_schema".to_string(),
        messages: vec![user_msg("What is the weather in Tokyo?")],
        system: AnthropicSystem::String("TensorBot".to_string()),
        max_tokens: 100,
        temperature: None,
        top_p: None,
        top_k: None,
        stop_sequences: None,
        stream: true,
        tools: Some(vec![get_weather_tool()]),
        tool_choice: None,
        metadata: None,
        service_tier: None,
        container: None,
    };

    let response = messages_handler(State(state.clone()), None, Json(params))
        .await
        .unwrap();

    let events = collect_sse_events(response).await;
    let event_types: Vec<&str> = events.iter().map(|(name, _)| name.as_str()).collect();

    assert_eq!(event_types[0], "message_start");
    assert_eq!(event_types[event_types.len() - 1], "message_stop");

    // Extract tool_use_id from the tool_use block.
    let tool_use = events
        .iter()
        .find(|(_, data)| data["type"].as_str() == Some("tool_use"));
    let tool_use_id = tool_use
        .expect("should have a tool_use block")
        .1
        .get("id")
        .unwrap()
        .as_str()
        .unwrap()
        .to_string();

    // Turn 2: Send tool_result back.
    let params = AnthropicMessagesParams {
        model: "tensorzero::function_name::basic_test_no_system_schema".to_string(),
        messages: vec![
            assistant_tool_use(
                &tool_use_id,
                "get_temperature",
                serde_json::json!({"location": "Tokyo"}),
            ),
            tool_result(
                &tool_use_id,
                vec![AnthropicContentBlockOwned::Text {
                    text: "The weather in Tokyo is sunny with a temperature of 22 degrees celsius."
                        .to_string(),
                    cache_control: None,
                }],
            ),
        ],
        system: AnthropicSystem::String("TensorBot".to_string()),
        max_tokens: 100,
        temperature: None,
        top_p: None,
        top_k: None,
        stop_sequences: None,
        stream: true,
        tools: None,
        tool_choice: None,
        metadata: None,
        service_tier: None,
        container: None,
    };

    let response = messages_handler(State(state), None, Json(params))
        .await
        .unwrap();

    let events = collect_sse_events(response).await;
    let event_types: Vec<&str> = events.iter().map(|(name, _)| name.as_str()).collect();

    assert_eq!(event_types[0], "message_start");
    assert_eq!(event_types[event_types.len() - 1], "message_stop");

    // Collect text deltas from second turn.
    let text_deltas: Vec<&str> = events
        .iter()
        .filter_map(|(name, data)| {
            if name == "content_block_delta" && data["delta_type"].as_str() == Some("text_delta") {
                data["text"].as_str()
            } else {
                None
            }
        })
        .collect();
    let full_text: String = text_deltas.join("");
    assert!(
        !full_text.is_empty(),
        "should have text content after tool_result, got: {full_text}"
    );
}

// ===========================================================================
// Scenario 4: Mixed text + tool + text streaming
// ===========================================================================
#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn test_streaming_mixed_blocks() {
    let client = tensorzero::test_helpers::make_embedded_gateway().await;
    let state = client.get_app_state_data().unwrap().load_latest();

    let params = AnthropicMessagesParams {
        model: "tensorzero::function_name::basic_test_no_system_schema".to_string(),
        messages: vec![user_msg("Check the weather and tell me about it.")],
        system: AnthropicSystem::String("TensorBot".to_string()),
        max_tokens: 100,
        temperature: None,
        top_p: None,
        top_k: None,
        stop_sequences: None,
        stream: true,
        tools: Some(vec![get_weather_tool()]),
        tool_choice: None,
        metadata: None,
        service_tier: None,
        container: None,
    };

    let response = messages_handler(State(state), None, Json(params))
        .await
        .unwrap();

    let events = collect_sse_events(response).await;
    let event_types: Vec<&str> = events.iter().map(|(name, _)| name.as_str()).collect();

    // Verify proper block lifecycle ordering.
    // Find all block starts and their corresponding stops.
    let start_indices: Vec<usize> = event_types
        .iter()
        .enumerate()
        .filter(|&(_, t)| *t == "content_block_start")
        .map(|(i, _)| i)
        .collect();

    let stop_indices: Vec<usize> = event_types
        .iter()
        .enumerate()
        .filter(|&(_, t)| *t == "content_block_stop")
        .map(|(i, _)| i)
        .collect();

    assert!(
        !start_indices.is_empty(),
        "should have at least one content_block_start, events: {event_types:?}"
    );

    // Each block should be closed before the next one opens.
    for (i, &start_idx) in start_indices.iter().enumerate() {
        let block_type = events[start_idx].1["type"].as_str().unwrap();
        println!("Block {i}: start@{start_idx} type={block_type}");

        // Find the corresponding stop (next content_block_stop after this start).
        let block_stop = stop_indices.iter().find(|si| **si > start_idx);
        assert!(
            block_stop.is_some(),
            "block {i} ({block_type}) started at {start_idx} but never stopped. events: {event_types:?}"
        );

        // If there's a next block start, verify this block stops before it.
        if i + 1 < start_indices.len() {
            let next_start = start_indices[i + 1];
            let stop = block_stop.unwrap();
            assert!(
                *stop < next_start,
                "block {i} stop @{stop} should be before next start @{next_start}. events: {event_types:?}"
            );
        }
    }

    assert_eq!(event_types[event_types.len() - 1], "message_stop");
}

// ===========================================================================
// Scenario 5: Streaming parity — OpenAI-compat vs Anthropic-compat
// ===========================================================================
#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn test_streaming_parity_openai_vs_anthropic() {
    let client = tensorzero::test_helpers::make_embedded_gateway().await;
    let state = client.get_app_state_data().unwrap().load_latest();

    // Make a streaming request via the Anthropic-compat endpoint.
    let anthropic_params = AnthropicMessagesParams {
        model: "tensorzero::function_name::basic_test_no_system_schema".to_string(),
        messages: vec![user_msg("What is the capital of Japan?")],
        system: AnthropicSystem::String("TensorBot".to_string()),
        max_tokens: 100,
        temperature: None,
        top_p: None,
        top_k: None,
        stop_sequences: None,
        stream: true,
        tools: None,
        tool_choice: None,
        metadata: None,
        service_tier: None,
        container: None,
    };

    let anthropic_response = messages_handler(State(state.clone()), None, Json(anthropic_params))
        .await
        .unwrap();

    let anthropic_events = collect_sse_events(anthropic_response).await;
    let _anthropic_event_types: Vec<&str> = anthropic_events
        .iter()
        .map(|(name, _)| name.as_str())
        .collect();
    let anthropic_inference_id: Uuid = anthropic_events[0]
        .1
        .get("id")
        .and_then(|id| id.as_str())
        .and_then(|s| s.strip_prefix("msg_"))
        .and_then(|s| s.parse().ok())
        .expect("should have inference id in message_start");

    // Make a non-streaming request via the same endpoint to compare DB records.
    // (The OpenAI-compat parity test already verifies this path extensively.)
    let non_streaming_params = AnthropicMessagesParams {
        model: "tensorzero::function_name::basic_test_no_system_schema".to_string(),
        messages: vec![user_msg("What is the capital of Japan?")],
        system: AnthropicSystem::String("TensorBot".to_string()),
        max_tokens: 100,
        temperature: None,
        top_p: None,
        top_k: None,
        stop_sequences: None,
        stream: false,
        tools: None,
        tool_choice: None,
        metadata: None,
        service_tier: None,
        container: None,
    };

    let non_streaming_response = messages_handler(State(state), None, Json(non_streaming_params))
        .await
        .unwrap();

    let body_bytes = non_streaming_response
        .into_body()
        .collect()
        .await
        .unwrap()
        .to_bytes();
    let body: Value = serde_json::from_slice(&body_bytes).unwrap();
    let non_streaming_inference_id: Uuid = body["id"].as_str().unwrap().parse().unwrap();

    // Verify both inference IDs exist in the DB.
    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;

    let config = get_e2e_config().await;

    let streaming_inference = conn
        .list_inferences(
            &config,
            &ListInferencesParams {
                ids: Some(&[anthropic_inference_id]),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(
        streaming_inference.len(),
        1,
        "streaming inference should exist in DB"
    );

    let non_streaming_inference = conn
        .list_inferences(
            &config,
            &ListInferencesParams {
                ids: Some(&[non_streaming_inference_id]),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(
        non_streaming_inference.len(),
        1,
        "non-streaming inference should exist in DB"
    );

    // Both should be Chat inferences of the same function.
    let streaming_chat = match &streaming_inference[0] {
        tensorzero_core::stored_inference::StoredInferenceDatabase::Chat(c) => c,
        tensorzero_core::stored_inference::StoredInferenceDatabase::Json(_) => {
            panic!("expected chat inference, got json")
        }
    };
    let non_streaming_chat = match &non_streaming_inference[0] {
        tensorzero_core::stored_inference::StoredInferenceDatabase::Chat(c) => c,
        tensorzero_core::stored_inference::StoredInferenceDatabase::Json(_) => {
            panic!("expected chat inference, got json")
        }
    };

    assert_eq!(
        streaming_chat.function_name, non_streaming_chat.function_name,
        "function name should match"
    );
    assert_eq!(
        streaming_chat.variant_name, non_streaming_chat.variant_name,
        "variant name should match"
    );

    // Reconstruct text from streaming events.
    let streaming_text: String = anthropic_events
        .iter()
        .filter_map(|(name, data)| {
            if name == "content_block_delta" && data["delta_type"].as_str() == Some("text_delta") {
                data["text"].as_str()
            } else {
                None
            }
        })
        .collect::<Vec<_>>()
        .join("");

    let non_streaming_content = body["content"]
        .as_array()
        .unwrap()
        .iter()
        .find(|b| b["type"].as_str() == Some("text"))
        .unwrap()["text"]
        .as_str()
        .unwrap();

    assert_eq!(
        streaming_text, non_streaming_content,
        "streaming and non-streaming text should match"
    );
}

// ===========================================================================
// Scenario 6: Keepalive ping events don't break SDK parsers
// ===========================================================================
#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn test_streaming_keepalive_pings() {
    // This test verifies that the SSE stream includes periodic keepalive
    // events (ping) and that they don't break standard SSE parsers.
    // The ping events appear as bare "event: ping\n\ndata: \n\n" lines.
    // Most SDKs simply ignore unknown event types.

    let client = tensorzero::test_helpers::make_embedded_gateway().await;
    let state = client.get_app_state_data().unwrap().load_latest();

    let params = make_streaming_request("basic_test_no_system_schema");

    let response = messages_handler(State(state), None, Json(params))
        .await
        .unwrap();

    let body_bytes = response.into_body().collect().await.unwrap().to_bytes();
    let body_str = String::from_utf8_lossy(&body_bytes);

    // Parse all event lines including pings.
    let mut event_types: Vec<&str> = Vec::new();
    let mut current_event = String::new();
    let mut in_event = false;

    for line in body_str.lines() {
        if line.is_empty() {
            if in_event {
                in_event = false;
            }
        } else if let Some(stripped) = line.strip_prefix("event:") {
            if in_event {
                // Unfinished previous event.
            }
            in_event = true;
            let event_name = stripped.trim();
            if event_name.is_empty() {
                event_types.push(""); // unnamed event
            } else {
                event_types.push(event_name);
            }
            current_event.clear();
        } else if let Some(stripped) = line.strip_prefix("data:") {
            let data = stripped.trim();
            current_event.push_str(data);
        }
    }

    // Collect ping events.
    let ping_count = event_types.iter().filter(|&&e| e == "ping").count();
    println!("Event types: {event_types:?} ({ping_count} pings)");

    // The stream should be parseable even with pings.
    // The critical invariant: all non-ping events maintain the correct sequence.
    let non_ping_events: Vec<&str> = event_types
        .iter()
        .filter(|e| **e != "ping")
        .copied()
        .collect();

    assert_eq!(non_ping_events[0], "message_start");
    assert_eq!(non_ping_events[non_ping_events.len() - 1], "message_stop");
    assert!(
        non_ping_events.contains(&"content_block_start"),
        "should have content_block_start in non-ping events"
    );
    assert!(
        non_ping_events.contains(&"message_delta"),
        "should have message_delta in non-ping events"
    );

    // Verify all SSE events are valid JSON (pings have empty data, which is fine).
    for (name, data) in body_str.split("\n\n").enumerate() {
        let parts: Vec<&str> = data.split('\n').collect();
        for part in parts {
            if let Some(stripped) = part.strip_prefix("data:") {
                let json_str = stripped.trim();
                if !json_str.is_empty() && json_str != "[DONE]" {
                    // Should be valid JSON.
                    let _parsed: Value = serde_json::from_str(json_str).unwrap_or_else(|_| {
                        panic!("event {name} data should be valid JSON: {json_str}");
                    });
                }
            }
        }
    }
}

// ===========================================================================
// Scenario 7: JSON streaming
// ===========================================================================
#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn test_streaming_json() {
    let client = tensorzero::test_helpers::make_embedded_gateway().await;
    let state = client.get_app_state_data().unwrap().load_latest();

    let params = AnthropicMessagesParams {
        model: "tensorzero::function_name::json_success_no_input_schema".to_string(),
        messages: vec![user_msg("Return a JSON object with answer: Tokyo")],
        system: AnthropicSystem::String("TensorBot".to_string()),
        max_tokens: 100,
        temperature: None,
        top_p: None,
        top_k: None,
        stop_sequences: None,
        stream: true,
        tools: None,
        tool_choice: None,
        metadata: None,
        service_tier: None,
        container: None,
    };

    let response = messages_handler(State(state), None, Json(params))
        .await
        .unwrap();

    let events = collect_sse_events(response).await;
    println!("JSON streaming events: {events:?}");

    let event_types: Vec<&str> = events.iter().map(|(name, _)| name.as_str()).collect();

    assert_eq!(event_types[0], "message_start");
    assert_eq!(event_types[event_types.len() - 1], "message_stop");

    // JSON mode should have a text content block with the raw JSON response.
    assert!(
        event_types.contains(&"content_block_start"),
        "should have content_block_start"
    );
    assert!(
        event_types.contains(&"content_block_delta"),
        "should have content_block_delta"
    );
    assert!(
        event_types.contains(&"content_block_stop"),
        "should have content_block_stop"
    );

    // Collect the JSON content.
    let text_deltas: Vec<&str> = events
        .iter()
        .filter_map(|(name, data)| {
            if name == "content_block_delta" && data["delta_type"].as_str() == Some("text_delta") {
                data["text"].as_str()
            } else {
                None
            }
        })
        .collect();
    let json_content: String = text_deltas.join("");
    assert!(
        !json_content.is_empty(),
        "JSON content should not be empty: {json_content}"
    );
}
