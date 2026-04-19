//! Type definitions for Anthropic-compatible API.
//!
//! This module contains all the wire format types used by the Anthropic-compatible endpoints,
//! organized into submodules for messages, streaming, tools, and usage.

pub mod messages;
pub mod streaming;
pub mod tool;
pub mod usage;
