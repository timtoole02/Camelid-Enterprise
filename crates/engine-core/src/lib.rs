//! Platform-neutral engine core.
//!
//! This crate holds everything the engine needs that does not depend on the
//! host CPU or GPU: the GGUF container format, model configuration, and the
//! types shared with the per-platform kernel crates (`engine-macos`,
//! `engine-linux`, `engine-windows`). Anything that dispatches on detected
//! hardware features lives in those crates, never here.

pub mod error;
pub mod forward;
pub mod gguf;
pub mod host;
pub mod model;
pub mod tensor;
pub mod tokenizer;

pub use error::{EngineError, Result};
