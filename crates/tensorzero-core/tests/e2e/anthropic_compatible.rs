#![expect(clippy::print_stdout)]

use axum::Json;
use axum::extract::State;
use googletest::prelude::*;
use http_body_util::BodyExt;
use serde_json::Value;
use tensorzero::ClientExt;
use uuid::Uuid;

use tensorzero_core::db::delegating_connection::DelegatingDatabaseConnection;
use tensorzero_core::db::inferences::{InferenceQueries, ListInferencesParams};
use tensorzero_core::db::model_inferences::ModelInferenceQueries;
use tensorzero_core::db::test_helpers::TestDatabaseHelpers;
use tensorzero_core::endpoints::anthropic_compatible::messages::messages_handler;
use tensorzero_core::endpoints::anthropic_compatible::types::messages::AnthropicContentBlockOwned;
use tensorzero_core::endpoints::anthropic_compatible::types::messages::{
    AnthropicMessageContentOwned, AnthropicMessageOwned, AnthropicMessagesParams,
    AnthropicRoleOwned, AnthropicSystem, AnthropicSystemContentBlockOwned, AnthropicToolOwned,
};
use tensorzero_core::stored_inference::StoredInferenceDatabase;
use tensorzero_core::test_helpers::get_e2e_config;

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

/// Run a basic text inference and verify response + DB records.
async fn test_anthropic_basic_text_with_function_name(function_name: &str) {
    let client = tensorzero::test_helpers::make_embedded_gateway().await;
    let state = client.get_app_state_data().unwrap().load_latest();
    let episode_id = Uuid::now_v7();

    let params = AnthropicMessagesParams {
        model: format!("tensorzero::function_name::{function_name}"),
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

    let response = messages_handler(State(state), None, Json(params))
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let body_bytes = response.into_body().collect().await.unwrap().to_bytes();
    let body: Value = serde_json::from_slice(&body_bytes).unwrap();
    println!("Anthropic response: {body:?}");

    // Anthropic response shape: type, role, content[], id, model, usage, stop_reason
    assert_eq!(body.get("type").unwrap().as_str().unwrap(), "message");
    assert_eq!(body.get("role").unwrap().as_str().unwrap(), "assistant");
    let content = body.get("content").unwrap().as_array().unwrap();
    assert!(!content.is_empty());
    assert_eq!(content[0].get("type").unwrap().as_str().unwrap(), "text");
    let text = content[0].get("text").unwrap().as_str().unwrap();
    assert_eq!(
        text,
        "Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake."
    );
    assert!(body.get("id").unwrap().as_str().is_some());
    let inference_id: Uuid = body.get("id").unwrap().as_str().unwrap().parse().unwrap();

    // Verify DB records
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
    assert_eq!(inferences.len(), 1);
    let chat = match &inferences[0] {
        StoredInferenceDatabase::Chat(c) => c,
        StoredInferenceDatabase::Json(_) => panic!("Expected chat inference, got json"),
    };
    assert_eq!(chat.episode_id, episode_id);
    assert_eq!(chat.variant_name, "test");
    let output = chat.output.as_ref().unwrap();
    assert_eq!(output.len(), 1);

    let model_inferences = conn
        .get_model_inferences_by_inference_id(inference_id)
        .await
        .unwrap();
    assert_eq!(model_inferences.len(), 1);
    assert_eq!(model_inferences[0].model_name, "test");
    assert_eq!(model_inferences[0].model_provider_name, "good");
}

/// Test basic text over OpenAI-underlying (dummy::good via "test" model).
#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn test_anthropic_basic_text_openai_underlying() {
    test_anthropic_basic_text_with_function_name("basic_test_no_system_schema").await;
}

/// Test multi-turn with tool_result.
#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn test_anthropic_tool_use_multi_turn() {
    let client = tensorzero::test_helpers::make_embedded_gateway().await;
    let state = client.get_app_state_data().unwrap().load_latest();
    let tool = AnthropicToolOwned {
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
    };

    // Turn 1: Send initial request with tool
    let params = AnthropicMessagesParams {
        model: "tensorzero::function_name::basic_test_no_system_schema".to_string(),
        messages: vec![user_msg("What is the weather in Tokyo?")],
        system: AnthropicSystem::String("TensorBot".to_string()),
        max_tokens: 100,
        temperature: None,
        top_p: None,
        top_k: None,
        stop_sequences: None,
        stream: false,
        tools: Some(vec![tool]),
        tool_choice: None,
        metadata: None,
        service_tier: None,
        container: None,
    };

    let response = messages_handler(State(state.clone()), None, Json(params))
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let body_bytes = response.into_body().collect().await.unwrap().to_bytes();
    let body: Value = serde_json::from_slice(&body_bytes).unwrap();
    println!("Tool use response: {body:?}");

    let content = body.get("content").unwrap().as_array().unwrap();
    let has_tool_use = content
        .iter()
        .any(|b| b.get("type").and_then(|t| t.as_str()) == Some("tool_use"));
    assert!(
        has_tool_use,
        "Expected at least one tool_use block in response"
    );

    // Turn 2: Send tool_result back
    let tool_use_id = body
        .get("content")
        .unwrap()
        .as_array()
        .unwrap()
        .iter()
        .find(|b| b.get("type").and_then(|t| t.as_str()) == Some("tool_use"))
        .unwrap()
        .get("id")
        .unwrap()
        .as_str()
        .unwrap()
        .to_string();

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
        stream: false,
        tools: None,
        tool_choice: None,
        metadata: None,
        service_tier: None,
        container: None,
    };

    let response = messages_handler(State(state), None, Json(params))
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let body_bytes = response.into_body().collect().await.unwrap().to_bytes();
    let body: Value = serde_json::from_slice(&body_bytes).unwrap();

    let content = body.get("content").unwrap().as_array().unwrap();
    let text_block = content
        .iter()
        .find(|b| b.get("type").and_then(|t| t.as_str()) == Some("text"))
        .unwrap();
    assert!(!text_block.get("text").unwrap().as_str().unwrap().is_empty());
}

/// Test system message as string.
#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn test_anthropic_system_message_string() {
    let client = tensorzero::test_helpers::make_embedded_gateway().await;
    let state = client.get_app_state_data().unwrap().load_latest();

    let params = AnthropicMessagesParams {
        model: "tensorzero::function_name::basic_test_no_system_schema".to_string(),
        messages: vec![user_msg("Hello")],
        system: AnthropicSystem::String("You are a helpful assistant named TensorBot".to_string()),
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

    let response = messages_handler(State(state), None, Json(params))
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let body_bytes = response.into_body().collect().await.unwrap().to_bytes();
    let body: Value = serde_json::from_slice(&body_bytes).unwrap();
    let text = body.get("content").unwrap().as_array().unwrap()[0]
        .get("text")
        .unwrap()
        .as_str()
        .unwrap();
    assert!(text.contains("Megumin") || text.contains("explosion"));
}

/// Test system message as blocks array.
#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn test_anthropic_system_message_blocks() {
    let client = tensorzero::test_helpers::make_embedded_gateway().await;
    let state = client.get_app_state_data().unwrap().load_latest();

    let params = AnthropicMessagesParams {
        model: "tensorzero::function_name::basic_test_no_system_schema".to_string(),
        messages: vec![user_msg("Hello")],
        system: AnthropicSystem::Blocks(vec![AnthropicSystemContentBlockOwned::Text {
            text: "TensorBot".to_string(),
            cache_control: None,
        }]),
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

    let response = messages_handler(State(state), None, Json(params))
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
}

/// Test parity: same logical call via OpenAI-compat and Anthropic-compat.
#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn test_anthropic_compatible_parity() {
    use tensorzero_core::endpoints::openai_compatible::OpenAIStructuredJson;
    use tensorzero_core::endpoints::openai_compatible::chat_completions::chat_completions_handler;

    let client = tensorzero::test_helpers::make_embedded_gateway().await;
    let state = client.get_app_state_data().unwrap().load_latest();

    // Send via Anthropic-compat
    let anthropic_params = AnthropicMessagesParams {
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

    let anthropic_response = messages_handler(State(state.clone()), None, Json(anthropic_params))
        .await
        .unwrap();
    let anthropic_body_bytes = anthropic_response
        .into_body()
        .collect()
        .await
        .unwrap()
        .to_bytes();
    let anthropic_body: Value = serde_json::from_slice(&anthropic_body_bytes).unwrap();
    let anthropic_inference_id: Uuid = anthropic_body
        .get("id")
        .unwrap()
        .as_str()
        .unwrap()
        .parse()
        .unwrap();

    // Send via OpenAI-compat with same logical call
    let openai_params = OpenAIStructuredJson(
        serde_json::from_value(serde_json::json!({
            "model": "tensorzero::function_name::basic_test_no_system_schema",
            "messages": [
                {"role": "system", "content": "TensorBot"},
                {"role": "user", "content": "What is the capital of Japan?"}
            ],
            "stream": false,
        }))
        .unwrap(),
    );

    let openai_response = chat_completions_handler(State(state.clone()), None, openai_params)
        .await
        .unwrap();
    let openai_body_bytes = openai_response
        .into_body()
        .collect()
        .await
        .unwrap()
        .to_bytes();
    let openai_body: Value = serde_json::from_slice(&openai_body_bytes).unwrap();
    let openai_inference_id: Uuid = openai_body
        .get("id")
        .unwrap()
        .as_str()
        .unwrap()
        .parse()
        .unwrap();

    // Verify DB records are equivalent
    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;

    let config = get_e2e_config().await;

    let anthropic_inference = conn
        .list_inferences(
            &config,
            &ListInferencesParams {
                ids: Some(&[anthropic_inference_id]),
                ..Default::default()
            },
        )
        .await
        .unwrap()[0]
        .clone();

    let openai_inference = conn
        .list_inferences(
            &config,
            &ListInferencesParams {
                ids: Some(&[openai_inference_id]),
                ..Default::default()
            },
        )
        .await
        .unwrap()[0]
        .clone();

    // Both should be Chat inferences of the same function
    let anthropic_chat = match &anthropic_inference {
        StoredInferenceDatabase::Chat(c) => c,
        StoredInferenceDatabase::Json(_) => panic!("Expected chat inference"),
    };
    let openai_chat = match &openai_inference {
        StoredInferenceDatabase::Chat(c) => c,
        StoredInferenceDatabase::Json(_) => panic!("Expected chat inference"),
    };

    assert_eq!(anthropic_chat.function_name, openai_chat.function_name);
    assert_eq!(anthropic_chat.variant_name, openai_chat.variant_name);

    // Both should have same input
    let anthropic_input = serde_json::to_value(&anthropic_chat.input).unwrap();
    let openai_input = serde_json::to_value(&openai_chat.input).unwrap();
    assert_eq!(anthropic_input, openai_input);

    // Both should have same output content
    let anthropic_output = serde_json::to_value(anthropic_chat.output.as_ref().unwrap()).unwrap();
    let openai_output = serde_json::to_value(openai_chat.output.as_ref().unwrap()).unwrap();
    assert_eq!(anthropic_output, openai_output);

    // ModelInference should be equivalent (same outbound provider request)
    let anthropic_model_inf = conn
        .get_model_inferences_by_inference_id(anthropic_inference_id)
        .await
        .unwrap();
    let openai_model_inf = conn
        .get_model_inferences_by_inference_id(openai_inference_id)
        .await
        .unwrap();

    assert_eq!(anthropic_model_inf.len(), 1);
    assert_eq!(openai_model_inf.len(), 1);
    assert_eq!(
        anthropic_model_inf[0].model_name,
        openai_model_inf[0].model_name
    );
    assert_eq!(
        anthropic_model_inf[0].model_provider_name,
        openai_model_inf[0].model_provider_name
    );
    // Raw requests should be identical since both go through the same provider
    assert_eq!(
        anthropic_model_inf[0].raw_request,
        openai_model_inf[0].raw_request
    );
}

/// Test response includes Anthropic-specific fields.
#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn test_anthropic_response_shape() {
    let client = tensorzero::test_helpers::make_embedded_gateway().await;
    let state = client.get_app_state_data().unwrap().load_latest();

    let params = AnthropicMessagesParams {
        model: "tensorzero::function_name::basic_test_no_system_schema".to_string(),
        messages: vec![user_msg("Hello")],
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

    let response = messages_handler(State(state), None, Json(params))
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let body_bytes = response.into_body().collect().await.unwrap().to_bytes();
    let body: Value = serde_json::from_slice(&body_bytes).unwrap();

    // Verify required Anthropic fields
    assert!(body.get("id").is_some());
    assert_eq!(body.get("type").unwrap().as_str().unwrap(), "message");
    assert_eq!(body.get("role").unwrap().as_str().unwrap(), "assistant");
    assert!(!body.get("content").unwrap().as_array().unwrap().is_empty());
    assert!(body.get("model").is_some());
    assert!(body.get("usage").is_some());

    // Usage should have input and output tokens
    let usage = body.get("usage").unwrap();
    assert!(usage.get("input_tokens").is_some());
    assert!(usage.get("output_tokens").is_some());
}
