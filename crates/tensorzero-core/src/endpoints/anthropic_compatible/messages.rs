//! Anthropic `/v1/messages` handler.
//!
//! See [`super::error`] for error formatting.
//! See [`super::types`] for request/response types.

use axum::Extension;
use axum::Json;
use axum::body::Body;
use axum::extract::State;
use axum::response::sse::Sse;
use axum::response::{IntoResponse, Response};
use secrecy::ExposeSecret;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::endpoints::anthropic_compatible::error::{AnthropicErrorBody, AnthropicErrorResponse};
use crate::endpoints::anthropic_compatible::types::messages::{
    AnthropicCountTokensParams, AnthropicMessagesParams, inference_response_to_anthropic,
};
use crate::endpoints::anthropic_compatible::types::streaming::convert_to_anthropic_stream;

use crate::endpoints::inference::inference;
use crate::model::ProviderConfig;
use crate::utils::gateway::{AppState, AppStateData};
use tensorzero_auth::middleware::RequestApiKeyExtension;

/// Handles `POST /v1/messages`.
pub async fn messages_handler(
    State(AppStateData {
        config,
        http_client,
        clickhouse_connection_info,
        postgres_connection_info,
        cache_manager,
        deferred_tasks,
        rate_limiting_manager,
        primary_datastore,
        ..
    }): AppState,
    api_key_ext: Option<Extension<RequestApiKeyExtension>>,
    Json(anthropic_params): Json<AnthropicMessagesParams>,
) -> Result<Response<Body>, AnthropicErrorResponse> {
    let params = match anthropic_params.clone().try_into_params() {
        Ok(p) => p,
        Err(e) => return Err(AnthropicErrorResponse::from(e)),
    };

    // Determine the response's model prefix
    let response_model_prefix = match (&params.function_name, &params.model_name) {
        (Some(function_name), None) => {
            format!("tensorzero::function_name::{function_name}::variant_name::",)
        }
        (None, Some(_model_name)) => "tensorzero::model_name::".to_string(),
        (Some(_), Some(_)) => {
            return Err(AnthropicErrorResponse::bad_request_msg(
                "Only one of `function_name` or `model_name` can be provided",
            ));
        }
        (None, None) => {
            return Err(AnthropicErrorResponse::bad_request_msg(
                "Either `function_name` or `model_name` must be provided",
            ));
        }
    };

    let inference_result = Box::pin(inference(
        config,
        &http_client,
        clickhouse_connection_info,
        postgres_connection_info,
        cache_manager,
        deferred_tasks,
        rate_limiting_manager,
        primary_datastore,
        params,
        api_key_ext,
    ))
    .await;

    let data = match inference_result {
        Ok(data) => data,
        Err(e) => return Err(AnthropicErrorResponse::from(e)),
    };

    if anthropic_params.stream {
        match data.output {
            crate::endpoints::inference::InferenceOutput::Streaming(stream) => {
                let sse = convert_to_anthropic_stream(stream, response_model_prefix);
                Ok(Sse::new(sse)
                    .keep_alive(axum::response::sse::KeepAlive::new())
                    .into_response())
            }
            crate::endpoints::inference::InferenceOutput::NonStreaming(_) => Err(
                AnthropicErrorResponse::internal_error(AnthropicErrorBody::api_error(
                    "Streaming was requested but the inference returned non-streaming output"
                        .to_string(),
                )),
            ),
        }
    } else {
        let response = data.output;
        let anthropic_response =
            match inference_response_to_anthropic(response, &response_model_prefix) {
                Ok(r) => r,
                Err(e) => {
                    return Err(AnthropicErrorResponse::internal_error(
                        AnthropicErrorBody::api_error(e.to_string()),
                    ));
                }
            };
        Ok(Json(anthropic_response).into_response())
    }
}

/// Upstream response shape for `POST /v1/messages/count_tokens`.
#[derive(Debug, Deserialize, Serialize)]
struct AnthropicCountTokensResponse {
    input_tokens: u32,
    #[serde(default)]
    cache_creation_input_tokens: u32,
    #[serde(default)]
    cache_read_input_tokens: u32,
}

/// Handles `POST /v1/messages/count_tokens`.
pub async fn count_tokens_handler(
    State(AppStateData {
        config,
        http_client,
        ..
    }): AppState,
    body: axum::body::Bytes,
) -> Result<Response<Body>, AnthropicErrorResponse> {
    // Parse the request body for validation and model resolution.
    let params: AnthropicCountTokensParams = serde_json::from_slice(&body).map_err(|e| {
        AnthropicErrorResponse::bad_request_msg(format!("Invalid request body: {e}"))
    })?;

    // Also keep the raw body as JSON so we can forward the relevant subset
    // to the upstream Anthropic count_tokens endpoint without needing Serialize
    // on every inbound type.
    let body_value: Value = serde_json::from_slice(&body).map_err(|e| {
        AnthropicErrorResponse::internal_error(AnthropicErrorBody::api_error(e.to_string()))
    })?;

    // Resolve the model name.
    let model_name = if let Some(rest) = params.model.strip_prefix("tensorzero::model_name::") {
        rest.to_string()
    } else if params.model.starts_with("tensorzero::function_name::") {
        return Err(AnthropicErrorResponse::unimplemented(
            AnthropicErrorBody::unimplemented(
                "count_tokens is not supported for function routes in Phase 3",
            ),
        ));
    } else {
        params.model.clone()
    };

    // Look up the model in the config.
    let model_config = config
        .models
        .table
        .get(model_name.as_str())
        .ok_or_else(|| {
            AnthropicErrorResponse::bad_request_msg(format!("Model '{model_name}' not found"))
        })?;

    let provider_name = model_config.routing.first().ok_or_else(|| {
        AnthropicErrorResponse::internal_error(AnthropicErrorBody::api_error(
            "Model has no providers configured".to_string(),
        ))
    })?;

    let provider = model_config
        .providers
        .get(provider_name.as_ref())
        .ok_or_else(|| {
            AnthropicErrorResponse::internal_error(AnthropicErrorBody::api_error(format!(
                "Provider '{provider_name}' not found for model '{model_name}'"
            )))
        })?;

    // Extract any per-request dynamic credentials from the metadata extension.
    let dynamic_api_keys = params
        .metadata
        .and_then(|m| m.tensorzero)
        .map(|tz| tz.credentials)
        .unwrap_or_default();

    match &provider.config {
        ProviderConfig::Anthropic(anthropic_provider) => {
            let api_key = anthropic_provider
                .credentials()
                .get_api_key(&dynamic_api_keys)
                .map_err(|e| {
                    AnthropicErrorResponse::internal_error(AnthropicErrorBody::api_error(
                        e.to_string(),
                    ))
                })?;

            let mut request_url = anthropic_provider.base_url().clone();
            if !request_url.path().ends_with('/') {
                request_url.set_path(&format!("{}/", request_url.path()));
            }
            let request_url = request_url.join("messages/count_tokens").map_err(|e| {
                AnthropicErrorResponse::internal_error(AnthropicErrorBody::api_error(e.to_string()))
            })?;

            // Build upstream body: keep only the fields Anthropic needs.
            let mut upstream_body = serde_json::Map::new();
            upstream_body.insert(
                "model".to_string(),
                Value::String(anthropic_provider.model_name().to_string()),
            );
            if let Some(messages) = body_value.get("messages") {
                upstream_body.insert("messages".to_string(), messages.clone());
            }
            if let Some(system) = body_value.get("system") {
                upstream_body.insert("system".to_string(), system.clone());
            }
            if let Some(tools) = body_value.get("tools") {
                upstream_body.insert("tools".to_string(), tools.clone());
            }
            let upstream_body = Value::Object(upstream_body);

            let res = http_client
                .post(request_url.as_ref())
                .header(
                    "anthropic-version",
                    crate::providers::anthropic::ANTHROPIC_API_VERSION,
                )
                .header("x-api-key", api_key.expose_secret())
                .json(&upstream_body)
                .send()
                .await
                .map_err(|e| {
                    AnthropicErrorResponse::internal_error(AnthropicErrorBody::api_error(
                        e.to_string(),
                    ))
                })?;

            if res.status().is_success() {
                let bytes = res.bytes().await.map_err(|e| {
                    AnthropicErrorResponse::internal_error(AnthropicErrorBody::api_error(
                        e.to_string(),
                    ))
                })?;
                let response: AnthropicCountTokensResponse = serde_json::from_slice(&bytes)
                    .map_err(|e| {
                        AnthropicErrorResponse::internal_error(AnthropicErrorBody::api_error(
                            e.to_string(),
                        ))
                    })?;
                Ok(Json(response).into_response())
            } else {
                let status = res.status();
                let text = res.text().await.unwrap_or_default();
                if status.is_client_error() {
                    Err(AnthropicErrorResponse::bad_request_msg(text))
                } else {
                    Err(AnthropicErrorResponse::internal_error(
                        AnthropicErrorBody::api_error(text),
                    ))
                }
            }
        }
        ProviderConfig::GCPVertexAnthropic(_) => Err(AnthropicErrorResponse::unimplemented(
            AnthropicErrorBody::unimplemented(
                "count_tokens is not supported for GCP Vertex Anthropic in Phase 3",
            ),
        )),
        _other => {
            let provider_type = provider.provider_type();
            Err(AnthropicErrorResponse::unimplemented(
                AnthropicErrorBody::unimplemented(format!(
                    "count_tokens is not supported for provider '{provider_type}'; only Anthropic-family providers support count_tokens"
                )),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use http_body_util::BodyExt;
    use serde_json::json;
    use std::collections::HashMap;

    /// Builds a minimal TOML config with an Anthropic provider pointing at `api_base`.
    fn anthropic_config(api_base: &str) -> String {
        format!(
            r#"
[models.test-model]
routing = ["test-provider"]

[models.test-model.providers.test-provider]
type = "anthropic"
model_name = "claude-test"
api_base = "{api_base}"
api_key_location = "dynamic::test_key"
"#
        )
    }

    /// Builds a minimal TOML config with an OpenAI provider.
    fn openai_config() -> String {
        r#"
[models.test-model]
routing = ["test-provider"]

[models.test-model.providers.test-provider]
type = "openai"
model_name = "gpt-4"
api_key_location = "dynamic::test_key"
"#
        .to_string()
    }

    async fn load_test_config(toml: &str) -> std::sync::Arc<crate::config::Config> {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), toml).unwrap();
        let glob = crate::config::ConfigFileGlob::new_from_path(tmp.path()).unwrap();
        let unwritten =
            crate::config::Config::load_from_path_optional_verify_credentials(&glob, false)
                .await
                .unwrap();
        std::sync::Arc::new(unwritten.dangerous_into_config_without_writing())
    }

    fn test_app_state(config: std::sync::Arc<crate::config::Config>) -> AppStateData {
        let mut mock_clickhouse =
            crate::db::clickhouse::clickhouse_client::MockClickHouseClient::new();
        mock_clickhouse
            .expect_batcher_join_handle()
            .returning(|| None);
        mock_clickhouse
            .expect_client_type()
            .return_const(crate::db::clickhouse::clickhouse_client::ClickHouseClientType::Disabled);
        let options = crate::utils::gateway::GatewayHandleTestOptions {
            clickhouse_client: std::sync::Arc::new(mock_clickhouse),
            postgres_healthy: true,
        };
        let handle = crate::utils::gateway::GatewayHandle::new_unit_test_data(config, options);
        handle.app_state.load_latest()
    }

    #[tokio::test]
    async fn test_count_tokens_anthropic_happy_path() {
        // Start a mock Anthropic upstream server.
        let app = axum::Router::new().route(
            "/v1/messages/count_tokens",
            axum::routing::post(
                |axum::Json(body): axum::Json<HashMap<String, Value>>| async move {
                    // Verify the upstream request has the provider model name.
                    assert_eq!(body.get("model").unwrap().as_str().unwrap(), "claude-test");
                    assert!(body.contains_key("messages"));
                    Json(json!({
                        "input_tokens": 42,
                        "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": 0,
                    }))
                },
            ),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        #[expect(clippy::disallowed_methods)]
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let config_str = anthropic_config(&format!("http://{addr}/v1/"));
        let config = load_test_config(&config_str).await;
        let state = test_app_state(config);

        let body = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "metadata": {
                "tensorzero": {
                    "credentials": {"test_key": "fake-key"}
                }
            }
        })
        .to_string();

        let response = count_tokens_handler(State(state), axum::body::Bytes::from(body))
            .await
            .unwrap();

        assert_eq!(response.status(), 200);
        let body_bytes = response.into_body().collect().await.unwrap().to_bytes();
        let body: Value = serde_json::from_slice(&body_bytes).unwrap();
        assert_eq!(body["input_tokens"], 42);
        assert_eq!(body["cache_creation_input_tokens"], 0);
        assert_eq!(body["cache_read_input_tokens"], 0);
    }

    #[tokio::test]
    async fn test_count_tokens_non_anthropic_501() {
        let config = load_test_config(&openai_config()).await;
        let state = test_app_state(config);

        let body = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
        })
        .to_string();

        let err = count_tokens_handler(State(state), axum::body::Bytes::from(body))
            .await
            .unwrap_err();

        assert_eq!(err.1, axum::http::StatusCode::NOT_IMPLEMENTED);
        let msg = err.0.error.message;
        assert!(
            msg.contains("openai"),
            "Expected error to name the provider, got: {msg}"
        );
    }

    #[tokio::test]
    async fn test_count_tokens_function_route_501() {
        let config_str = anthropic_config("http://localhost:9999/v1/");
        let config = load_test_config(&config_str).await;
        let state = test_app_state(config);

        let body = json!({
            "model": "tensorzero::function_name::my_func",
            "messages": [{"role": "user", "content": "Hello"}],
        })
        .to_string();

        let err = count_tokens_handler(State(state), axum::body::Bytes::from(body))
            .await
            .unwrap_err();

        assert_eq!(err.1, axum::http::StatusCode::NOT_IMPLEMENTED);
        assert!(err.0.error.message.contains("function routes"));
    }
}
