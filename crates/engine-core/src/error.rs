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
