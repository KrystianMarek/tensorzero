//! Anthropic-shaped error handling.
//!
//! This module provides error types and helpers that format responses according to
//! Anthropic's API specification.
//!
//! # Error format
//!
//! Anthropic errors follow this shape:
//!
//! ```json
//! {
//!   "type": "error",
//!   "error": {
//!     "type": "invalid_request_error",
//!     "message": "..."
//!   }
//! }
//! ```
//!
//! # Mapping internal errors to Anthropic error categories
//!
//! | Internal | Anthropic `error.type` | HTTP |
//! |----------|------------------------|------|
//! | Authentication | `authentication_error` | 401 |
//! | Permission / missing scope | `permission_error` | 403 |
//! | Validation / bad input | `invalid_request_error` | 400 |
//! | Rate limit | `rate_limit_error` | 429 |
//! | Provider overloaded | `overloaded_error` | 529 |
//! | Timeout | `api_error` | 504 |
//! | Unexpected | `api_error` | 500 |
//! | Unimplemented | `501` | 501 |

use axum::Json;
use axum::http::StatusCode;
use serde::Serialize;

use crate::error::Error;

/// The top-level error response body for Anthropic-compatible endpoints.
#[derive(Debug, Serialize)]
pub struct AnthropicErrorBody {
    pub r#type: &'static str,
    pub error: AnthropicError,
}

/// The inner error object.
#[derive(Debug, Serialize)]
pub struct AnthropicError {
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub r#type: Option<&'static str>,
}

impl AnthropicErrorBody {
    /// Creates a 400 Bad Request error for invalid client input.
    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self {
            r#type: "error",
            error: AnthropicError {
                message: message.into(),
                r#type: Some("invalid_request_error"),
            },
        }
    }

    /// Creates a 401 Unauthorized error for authentication failures.
    pub fn authentication_error(message: impl Into<String>) -> Self {
        Self {
            r#type: "error",
            error: AnthropicError {
                message: message.into(),
                r#type: Some("authentication_error"),
            },
        }
    }

    /// Creates a 403 Forbidden error for permission/scope failures.
    pub fn permission_error(message: impl Into<String>) -> Self {
        Self {
            r#type: "error",
            error: AnthropicError {
                message: message.into(),
                r#type: Some("permission_error"),
            },
        }
    }

    /// Creates a 429 Too Many Requests error for rate limiting.
    pub fn rate_limit_error(message: impl Into<String>) -> Self {
        Self {
            r#type: "error",
            error: AnthropicError {
                message: message.into(),
                r#type: Some("rate_limit_error"),
            },
        }
    }

    /// Creates a 529 Overloaded error when the provider is at capacity.
    pub fn overloaded_error(message: impl Into<String>) -> Self {
        Self {
            r#type: "error",
            error: AnthropicError {
                message: message.into(),
                r#type: Some("overloaded_error"),
            },
        }
    }

    /// Creates a generic 500 Internal Server Error.
    pub fn api_error(message: impl Into<String>) -> Self {
        Self {
            r#type: "error",
            error: AnthropicError {
                message: message.into(),
                r#type: Some("api_error"),
            },
        }
    }

    /// Creates a 501 Not Implemented error.
    pub fn unimplemented(message: impl Into<String>) -> Self {
        Self {
            r#type: "error",
            error: AnthropicError {
                message: message.into(),
                r#type: Some("not_implemented_error"),
            },
        }
    }
}

/// A wrapper around `AnthropicErrorBody` that implements `axum::IntoResponse`.
#[derive(Debug)]
pub struct AnthropicErrorResponse(pub AnthropicErrorBody, pub StatusCode);

impl AnthropicErrorResponse {
    /// Convenience constructor: returns 400 with the given error body.
    pub fn bad_request(body: AnthropicErrorBody) -> Self {
        Self(body, StatusCode::BAD_REQUEST)
    }

    /// Convenience constructor: returns 401.
    pub fn unauthorized(body: AnthropicErrorBody) -> Self {
        Self(body, StatusCode::UNAUTHORIZED)
    }

    /// Convenience constructor: returns 403.
    pub fn forbidden(body: AnthropicErrorBody) -> Self {
        Self(body, StatusCode::FORBIDDEN)
    }

    /// Convenience constructor: returns 429.
    pub fn rate_limited(body: AnthropicErrorBody) -> Self {
        Self(body, StatusCode::TOO_MANY_REQUESTS)
    }

    /// Convenience constructor: returns 500.
    pub fn internal_error(body: AnthropicErrorBody) -> Self {
        Self(body, StatusCode::INTERNAL_SERVER_ERROR)
    }

    /// Convenience constructor: returns 501.
    pub fn unimplemented(body: AnthropicErrorBody) -> Self {
        Self(body, StatusCode::NOT_IMPLEMENTED)
    }

    /// Convenience constructor: returns 529.
    pub fn overloaded(body: AnthropicErrorBody) -> Self {
        Self(body, StatusCode::SERVICE_UNAVAILABLE)
    }

    /// Convenience constructor: creates a bad_request from a string message.
    pub fn bad_request_msg(msg: impl Into<String>) -> Self {
        Self::bad_request(AnthropicErrorBody::invalid_request(msg))
    }
}

impl From<Error> for AnthropicErrorResponse {
    fn from(error: Error) -> Self {
        let status = error.status_code();
        let message = error.to_string();
        match status {
            StatusCode::UNAUTHORIZED => {
                Self::unauthorized(AnthropicErrorBody::authentication_error(message))
            }
            StatusCode::FORBIDDEN => Self::forbidden(AnthropicErrorBody::permission_error(message)),
            StatusCode::TOO_MANY_REQUESTS => {
                Self::rate_limited(AnthropicErrorBody::rate_limit_error(message))
            }
            StatusCode::SERVICE_UNAVAILABLE => {
                Self::overloaded(AnthropicErrorBody::overloaded_error(message))
            }
            StatusCode::NOT_IMPLEMENTED => {
                Self::unimplemented(AnthropicErrorBody::unimplemented(message))
            }
            // Server errors — map to internal_error / api_error so SDKs retry appropriately
            StatusCode::INTERNAL_SERVER_ERROR
            | StatusCode::BAD_GATEWAY
            | StatusCode::REQUEST_TIMEOUT => {
                Self::internal_error(AnthropicErrorBody::api_error(message))
            }
            // Everything else — client input errors
            _ => Self::bad_request(AnthropicErrorBody::invalid_request(message)),
        }
    }
}

impl axum::response::IntoResponse for AnthropicErrorResponse {
    fn into_response(self) -> axum::response::Response {
        (self.1, Json(self.0)).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ErrorDetails;
    use serde_json::json;

    #[test]
    fn test_invalid_request_error() {
        let body = AnthropicErrorBody::invalid_request("bad input");
        let serialized = serde_json::to_string(&body).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&serialized).unwrap();
        assert_eq!(
            parsed,
            json!({
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": "bad input"
                }
            })
        );
    }

    #[test]
    fn test_unauthorized_error() {
        let body = AnthropicErrorBody::authentication_error("no token");
        assert_eq!(body.error.r#type, Some("authentication_error"));
    }

    #[test]
    fn test_unimplemented_error() {
        let body = AnthropicErrorBody::unimplemented("not yet");
        assert_eq!(body.error.r#type, Some("not_implemented_error"));
    }

    #[test]
    fn test_from_error_server_error_maps_to_500() {
        let error = Error::new(ErrorDetails::InternalError {
            message: "something broke".to_string(),
        });
        let response: AnthropicErrorResponse = error.into();
        assert_eq!(response.1, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(response.0.error.r#type, Some("api_error"));
    }

    #[test]
    fn test_from_error_not_implemented_maps_to_501() {
        let error = Error::new(ErrorDetails::NotImplemented {
            message: "not yet done".to_string(),
        });
        let response: AnthropicErrorResponse = error.into();
        assert_eq!(response.1, StatusCode::NOT_IMPLEMENTED);
        assert_eq!(response.0.error.r#type, Some("not_implemented_error"));
    }

    #[test]
    fn test_from_error_client_error_remains_400() {
        let error = Error::new(ErrorDetails::InvalidRequest {
            message: "bad field".to_string(),
        });
        let response: AnthropicErrorResponse = error.into();
        assert_eq!(response.1, StatusCode::BAD_REQUEST);
        assert_eq!(response.0.error.r#type, Some("invalid_request_error"));
    }
}
