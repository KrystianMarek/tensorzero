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

use crate::endpoints::anthropic_compatible::error::{AnthropicErrorBody, AnthropicErrorResponse};
use crate::endpoints::anthropic_compatible::types::messages::{
    AnthropicMessagesParams, inference_response_to_anthropic,
};
use crate::endpoints::anthropic_compatible::types::streaming::convert_to_anthropic_stream;
use crate::endpoints::inference::inference;
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

/// Handles `POST /v1/messages/count_tokens`.
pub async fn count_tokens_handler() -> AnthropicErrorResponse {
    AnthropicErrorResponse::unimplemented(AnthropicErrorBody::unimplemented(
        "count_tokens handler not yet implemented",
    ))
}
