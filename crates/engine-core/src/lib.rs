//! Platform-neutral engine core.
//!
//! This crate holds everything the engine needs that does not depend on the
//! host CPU or GPU: the GGUF container format, model configuration, and the
//! types shared with the per-platform kernel crates (`engine-macos`,
//! `engine-linux`, `engine-windows`) — the kernel seams themselves, the host
//! capability description those crates fill in, and the [`posture`] vocabulary
//! they declare their kernels' numeric contract in. Anything that dispatches on
//! detected hardware features lives in those crates, never here.

pub mod error;
pub mod forward;
pub mod gguf;
pub mod host;
pub mod model;
pub mod posture;
pub mod runtime;
pub mod tensor;
pub mod tokenizer;

pub use error::{EngineError, Result};
