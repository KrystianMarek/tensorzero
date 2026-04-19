//! Anthropic-compatible API endpoints.
//!
//! This module provides an Anthropic-compatible ingress (`/v1/messages`) so clients
//! using the Anthropic SDK can route through the TensorZero gateway without switching to the OpenAI SDK.
//!
//! # Architecture
//!
//! The module mirrors the structure of [`super::openai_compatible`]:
//!
//! - [`messages`] — `POST /messages` handler (non-streaming + streaming)
//! - [`error`] — Anthropic-shaped error responses
//! - [`types`] — wire format types for deserializing requests and serializing responses
//!
//! # Route paths
//!
//! Both paths point to the same handler:
//!
//! - `/anthropic/v1/messages` — canonical path
//! - `/v1/messages` — alias (keeps Anthropic SDK zero-config with default `base_url`)

pub mod error;
pub mod messages;
pub mod types;

use crate::endpoints::RouteHandlers;
use axum::routing::post;

/// Builds the Anthropic-compatible route handlers.
///
/// Routes:
/// - `POST /anthropic/v1/messages`
/// - `POST /anthropic/v1/messages/count_tokens`
/// - `POST /v1/messages`
/// - `POST /v1/messages/count_tokens`
///
/// Both paths share the same handler function.
pub fn build_anthropic_compatible_routes() -> RouteHandlers {
    RouteHandlers {
        routes: vec![
            ("/anthropic/v1/messages", post(messages::messages_handler)),
            (
                "/anthropic/v1/messages/count_tokens",
                post(messages::count_tokens_handler),
            ),
            ("/v1/messages", post(messages::messages_handler)),
            (
                "/v1/messages/count_tokens",
                post(messages::count_tokens_handler),
            ),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_route_count() {
        let handlers = build_anthropic_compatible_routes();
        assert_eq!(handlers.routes.len(), 4);
    }
}
