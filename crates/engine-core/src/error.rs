use std::path::PathBuf;

pub type Result<T> = std::result::Result<T, EngineError>;

#[derive(Debug)]
pub enum EngineError {
    /// The file violates the GGUF container format.
    InvalidGguf(String),
    /// The file is valid GGUF but uses a feature this engine does not support.
    UnsupportedGguf(String),
    /// The loaded model carries no tokenizer metadata at all. Distinct from a
    /// format violation: the container is fine, the tokenizer just isn't there.
    TokenizerNotAvailable,
    /// Tokenizer metadata is present but malformed or self-inconsistent.
    InvalidTokenizerMetadata(String),
    /// Tokenizer metadata is valid but describes a scheme this engine does not
    /// support.
    UnsupportedTokenizer(String),
    /// Tensor payload bytes are malformed for their declared type: misaligned
    /// block lengths, byte counts that disagree with the element count, or
    /// values the format forbids. Distinct from a container violation: the
    /// GGUF framing is fine, the tensor data itself is not usable.
    InvalidTensorData(String),
    /// A tensor was requested by a name the loaded model does not contain.
    /// Distinct from malformed data: the store is fine, the name misses.
    TensorNotFound(String),
    /// The tensor's storage type parsed fine but the requested load path has
    /// no decoder for it. Distinct from a container violation: the type id is
    /// legal GGUF, this engine just cannot decode it here.
    UnsupportedTensorType(String),
    /// Runtime operands do not fit together: wrong rank, disagreeing
    /// dimensions, or data lengths that do not match a declared shape.
    /// Distinct from InvalidTensorData: each tensor is internally fine, the
    /// combination requested of them is not.
    ShapeMismatch(String),
    Io { path: PathBuf, source: std::io::Error },
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidGguf(msg) => write!(f, "invalid GGUF: {msg}"),
            Self::UnsupportedGguf(msg) => write!(f, "unsupported GGUF: {msg}"),
            Self::TokenizerNotAvailable => {
                write!(f, "tokenizer metadata is not available in the loaded model")
            }
            Self::InvalidTokenizerMetadata(msg) => write!(f, "invalid tokenizer metadata: {msg}"),
            Self::UnsupportedTokenizer(msg) => write!(f, "unsupported tokenizer: {msg}"),
            Self::InvalidTensorData(msg) => write!(f, "invalid tensor data: {msg}"),
            Self::TensorNotFound(name) => write!(f, "tensor not found: {name}"),
            Self::UnsupportedTensorType(msg) => write!(f, "unsupported tensor type: {msg}"),
            Self::ShapeMismatch(msg) => write!(f, "shape mismatch: {msg}"),
            Self::Io { path, source } => write!(f, "io error on {}: {source}", path.display()),
        }
    }
}

impl std::error::Error for EngineError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}
