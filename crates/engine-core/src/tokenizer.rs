//! Tokenizer built from GGUF metadata.
//!
//! Two vocabulary families are supported: SentencePiece (`tokenizer.ggml.model
//! = "llama"` or `"gemma4"`) and byte-level GPT-2 BPE (`"gpt2"`). Everything —
//! vocabulary, merge ranks, scores, token kinds, special ids, and behavior
//! flags — is read from the file's metadata; nothing is inferred from the
//! tensor data.

use std::collections::{BTreeSet, BinaryHeap, HashMap};

use crate::{gguf::GgufFile, EngineError, Result};

/// Index of a token in the vocabulary.
pub type TokenId = u32;

/// The SentencePiece meta-space character (U+2581): encodes ASCII space.
const SPM_SPACE: char = '▁';

/// Vocabulary family, from `tokenizer.ggml.model`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerModel {
    /// SentencePiece (unigram) vocabulary: `"llama"` and `"gemma4"`.
    LlamaSpm,
    /// Byte-level GPT-2 BPE vocabulary: `"gpt2"`.
    Gpt2Bpe,
}

impl TokenizerModel {
    /// Stable snake_case name for logs and summaries.
    pub fn as_summary_model(self) -> &'static str {
        match self {
            Self::LlamaSpm => "llama_spm",
            Self::Gpt2Bpe => "gpt2_bpe",
        }
    }
}

/// GPT-2/BPE pre-tokenizer dialect (`tokenizer.ggml.pre`). The byte-level BPE
/// merge step is identical across these; the pre-tokenization regex that splits
/// raw text into pieces differs only in digit grouping and mark folding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BpePreTokenizer {
    /// `llama-bpe` (Llama 3): digits group in runs of up to three
    /// (`\p{N}{1,3}`).
    #[default]
    Llama3,
    /// `qwen2` (Qwen2/Qwen3): each digit is its own piece (`\p{N}`).
    /// Identical to `llama-bpe` in every other branch.
    Qwen2,
    /// `qwen35` (Qwen3.5): single-digit grouping like `qwen2`, and the split
    /// regex additionally folds Unicode combining marks `\p{M}` into the
    /// letter class (`\p{L}+` → `[\p{L}\p{M}]+`; the punctuation class
    /// excludes `\p{M}`). Mark-folding is implemented via a generated `\p{M}`
    /// range table — see [`fold_marks`](Self::fold_marks).
    Qwen35,
}

impl BpePreTokenizer {
    /// Maximum number of consecutive digits the pre-tokenizer keeps in one piece.
    fn digit_group_max(self) -> usize {
        match self {
            Self::Llama3 => 3,
            Self::Qwen2 | Self::Qwen35 => 1,
        }
    }

    /// Whether `\p{M}` combining marks fold into the letter class (and are
    /// excluded from the punctuation class) — the qwen35 regex dialect.
    fn fold_marks(self) -> bool {
        matches!(self, Self::Qwen35)
    }
}

/// Per-token type from `tokenizer.ggml.token_type` (values 0-6).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenKind {
    Undefined,
    Normal,
    Unknown,
    Control,
    UserDefined,
    Unused,
    Byte,
}

impl TokenKind {
    fn from_i32(value: i32) -> Result<Self> {
        Ok(match value {
            0 => Self::Undefined,
            1 => Self::Normal,
            2 => Self::Unknown,
            3 => Self::Control,
            4 => Self::UserDefined,
            5 => Self::Unused,
            6 => Self::Byte,
            other => {
                return Err(EngineError::InvalidTokenizerMetadata(format!(
                    "unknown tokenizer token type {other}"
                )))
            }
        })
    }
}

/// One vocabulary entry: `id` equals its index in [`Tokenizer::tokens`].
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub id: TokenId,
    pub text: String,
    pub score: f32,
    pub kind: TokenKind,
}

/// True for `<|...|>` chat-control markers (e.g. Phi-3's `<|end|>`/`<|assistant|>`,
/// ChatML's `<|im_start|>`) carried as `UserDefined` tokens. These are turn
/// scaffolding, never user-visible content, so they are stripped from decoded
/// output under `remove_special` — exactly like `Control` tokens. Content-bearing
/// `UserDefined` tokens that are NOT this shape (e.g. Qwen3's `<think>`/`</think>`,
/// which begin `<` but not `<|`) are preserved.
fn is_chat_control_marker(token: &Token) -> bool {
    token.kind == TokenKind::UserDefined
        && token.text.starts_with("<|")
        && token.text.ends_with("|>")
}

/// BPE merge table: maps an adjacent symbol pair to its merge rank
/// (lower rank = earlier in `tokenizer.ggml.merges` = higher priority).
#[derive(Debug, Clone, Default)]
pub struct BpeRegistry {
    ranks: HashMap<(String, String), usize>,
}

impl BpeRegistry {
    fn from_merges(merges: Vec<String>) -> Self {
        let ranks = merges
            .into_iter()
            .enumerate()
            .filter_map(|(rank, merge)| {
                let (left, right) = merge.split_once(' ')?;
                Some(((left.to_string(), right.to_string()), rank))
            })
            .collect();
        Self { ranks }
    }

    /// Number of merge rules.
    pub fn len(&self) -> usize {
        self.ranks.len()
    }

    /// True when no merge rules are present (SPM models, or a BPE file
    /// without `tokenizer.ggml.merges`).
    pub fn is_empty(&self) -> bool {
        self.ranks.is_empty()
    }

    fn rank(&self, left: &str, right: &str) -> Option<usize> {
        self.ranks
            .get(&(left.to_string(), right.to_string()))
            .copied()
    }

    fn ranks(&self) -> &HashMap<(String, String), usize> {
        &self.ranks
    }

    fn merge_symbols(&self, mut symbols: Vec<String>) -> Vec<String> {
        while symbols.len() > 1 {
            let mut heap = BinaryHeap::new();
            for idx in 0..symbols.len() - 1 {
                if let Some(rank) = self.rank(&symbols[idx], &symbols[idx + 1]) {
                    heap.push(BpeMergeCandidate { rank, index: idx });
                }
            }

            let Some(best) = heap.pop() else { break };
            let left = symbols[best.index].clone();
            let right = symbols[best.index + 1].clone();
            let mut merged = Vec::with_capacity(symbols.len() - 1);
            let mut idx = 0;
            while idx < symbols.len() {
                if idx + 1 < symbols.len() && symbols[idx] == left && symbols[idx + 1] == right {
                    merged.push(format!("{}{}", symbols[idx], symbols[idx + 1]));
                    idx += 2;
                } else {
                    merged.push(symbols[idx].clone());
                    idx += 1;
                }
            }
            symbols = merged;
        }
        symbols
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BpeMergeCandidate {
    rank: usize,
    index: usize,
}

impl Ord for BpeMergeCandidate {
    // Reversed on both fields so the max-heap pops the LOWEST rank, with ties
    // broken by the LOWEST (leftmost) index.
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .rank
            .cmp(&self.rank)
            .then_with(|| other.index.cmp(&self.index))
    }
}

impl PartialOrd for BpeMergeCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Special token ids resolved from metadata, vocab-text fallbacks, and
/// model-family defaults.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SpecialTokens {
    pub bos: Option<TokenId>,
    pub eos: Option<TokenId>,
    pub eot: Option<TokenId>,
    pub eom: Option<TokenId>,
    pub unk: Option<TokenId>,
    pub sep: Option<TokenId>,
    pub pad: Option<TokenId>,
    pub mask: Option<TokenId>,
    /// End-of-generation ids: `{eos, eot, eom}` plus any well-known
    /// end-of-turn marker texts present in the vocabulary.
    pub eog: BTreeSet<TokenId>,
}

/// Behavior flags from the `tokenizer.ggml.add_*` / whitespace metadata keys.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenizerConfig {
    pub add_bos: bool,
    pub add_eos: bool,
    pub add_sep: bool,
    pub add_space_prefix: bool,
    pub remove_extra_whitespaces: bool,
}

/// A tokenizer constructed from GGUF metadata via [`Tokenizer::from_gguf`].
#[derive(Debug, Clone)]
pub struct Tokenizer {
    pub model: TokenizerModel,
    /// GPT-2/BPE pre-tokenizer dialect. Only consulted on the [`TokenizerModel::Gpt2Bpe`]
    /// path; defaults to [`BpePreTokenizer::Llama3`] and is ignored for SPM.
    pub bpe_pre_tokenizer: BpePreTokenizer,
    pub tokens: Vec<Token>,
    pub token_to_id: HashMap<String, TokenId>,
    pub byte_token_to_id: HashMap<u8, TokenId>,
    pub bpe_ranks: HashMap<(String, String), usize>,
    pub bpe_registry: BpeRegistry,
    pub special: SpecialTokens,
    pub config: TokenizerConfig,
    pub chat_template: Option<String>,
}

/// The Llama-3 tokenizer's special-token signature. Llama 3 / 3.1 / 3.2 all
/// place these five stable chat markers at these exact ids in a 128,256-token
/// vocab, and no other tokenizer family does — so a GPT-2/BPE GGUF carrying
/// them IS the llama-bpe tokenizer. Used only to recover a MISSING
/// `tokenizer.ggml.pre` (see [`Tokenizer::from_gguf`]); the checked ids
/// deliberately exclude the reserved slots that were renamed between Llama-3
/// and 3.2 (e.g. 128008 `<|eom_id|>`).
fn is_llama3_bpe_signature(token_texts: &[String]) -> bool {
    token_texts.len() == 128_256
        && token_texts.get(128_000).map(String::as_str) == Some("<|begin_of_text|>")
        && token_texts.get(128_001).map(String::as_str) == Some("<|end_of_text|>")
        && token_texts.get(128_006).map(String::as_str) == Some("<|start_header_id|>")
        && token_texts.get(128_007).map(String::as_str) == Some("<|end_header_id|>")
        && token_texts.get(128_009).map(String::as_str) == Some("<|eot_id|>")
}

/// Resolve the GPT-2/BPE pre-tokenizer dialect from `tokenizer.ggml.pre`. The
/// three known dialects differ only in the split regex (digit grouping / mark
/// folding); the byte-BPE merge step is identical. When the key is ABSENT,
/// recover llama-bpe iff the vocab carries the Llama-3 signature — some
/// Llama-3 GGUF conversions omit the key, and a raw GPT-2 fallback split would
/// silently mis-tokenize them. An explicit-but-unknown `pre`, or a missing key
/// without the signature (e.g. a de-labeled Qwen), is refused. Extracted from
/// `from_gguf` so the decision is unit-testable.
fn resolve_gpt2_pre_tokenizer(
    pre: Option<&str>,
    token_texts: &[String],
) -> Result<BpePreTokenizer> {
    match pre {
        Some("llama-bpe") => Ok(BpePreTokenizer::Llama3),
        Some("qwen2") => Ok(BpePreTokenizer::Qwen2),
        Some("qwen35") => Ok(BpePreTokenizer::Qwen35),
        None if is_llama3_bpe_signature(token_texts) => Ok(BpePreTokenizer::Llama3),
        other => Err(EngineError::UnsupportedTokenizer(format!(
            "unsupported GPT-2/BPE pre-tokenizer {other:?}; currently supported: llama-bpe, qwen2, qwen35"
        ))),
    }
}

impl Tokenizer {
    /// Build a tokenizer from a parsed GGUF header. Fails closed: an unknown
    /// tokenizer model, an unknown pre-tokenizer, malformed metadata arrays,
    /// or an out-of-range special id all refuse construction.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let model_name = file
            .metadata_string("tokenizer.ggml.model")
            .ok_or(EngineError::TokenizerNotAvailable)?;
        let model = match model_name {
            // Gemma uses a SentencePiece (unigram) tokenizer, the same mechanism
            // as Llama SPM — tokens, scores, and the bos/eos/unk ids are all read
            // from the GGUF below.
            "llama" | "gemma4" => TokenizerModel::LlamaSpm,
            "gpt2" => TokenizerModel::Gpt2Bpe,
            other => {
                return Err(EngineError::UnsupportedTokenizer(format!(
                    "unsupported tokenizer model {other:?}; currently supported: llama/SPM, gemma4/SPM, and GPT-2/BPE llama-bpe"
                )))
            }
        };
        // Read the token list up front: it is needed both to recover a missing
        // pre-tokenizer from the vocab signature (below) and to build the vocab.
        let token_texts = file.metadata_array_strings("tokenizer.ggml.tokens")?;
        if token_texts.is_empty() {
            return Err(EngineError::InvalidTokenizerMetadata(
                "tokenizer.ggml.tokens must not be empty".to_string(),
            ));
        }

        let bpe_pre_tokenizer = if model == TokenizerModel::Gpt2Bpe {
            resolve_gpt2_pre_tokenizer(file.metadata_string("tokenizer.ggml.pre"), &token_texts)?
        } else {
            BpePreTokenizer::default()
        };

        let scores = file
            .metadata_array_f32_optional("tokenizer.ggml.scores")?
            .unwrap_or_else(|| vec![0.0; token_texts.len()]);
        if scores.len() < token_texts.len() {
            return Err(EngineError::InvalidTokenizerMetadata(format!(
                "tokenizer.ggml.scores length {} is shorter than token count {}",
                scores.len(),
                token_texts.len()
            )));
        }

        let kinds_raw = file
            .metadata_array_i32_optional("tokenizer.ggml.token_type")?
            .unwrap_or_else(|| vec![1; token_texts.len()]);
        if kinds_raw.len() < token_texts.len() {
            return Err(EngineError::InvalidTokenizerMetadata(format!(
                "tokenizer.ggml.token_type length {} is shorter than token count {}",
                kinds_raw.len(),
                token_texts.len()
            )));
        }

        let bpe_registry = BpeRegistry::from_merges(
            file.metadata_array_strings_optional("tokenizer.ggml.merges")?
                .unwrap_or_default(),
        );
        let bpe_ranks = bpe_registry.ranks().clone();

        let mut tokens = Vec::with_capacity(token_texts.len());
        let mut token_to_id = HashMap::with_capacity(token_texts.len());
        let mut byte_token_to_id = HashMap::new();
        for (idx, text) in token_texts.into_iter().enumerate() {
            let id = idx as TokenId;
            let kind = TokenKind::from_i32(kinds_raw[idx])?;
            if let Some(byte) = parse_byte_token(&text) {
                byte_token_to_id.insert(byte, id);
            }
            token_to_id.insert(text.clone(), id);
            tokens.push(Token {
                id,
                text,
                score: scores[idx],
                kind,
            });
        }

        let default_bos = match model {
            TokenizerModel::LlamaSpm => Some(1),
            TokenizerModel::Gpt2Bpe => token_to_id.get("<|begin_of_text|>").copied(),
        };
        let default_eos = match model {
            TokenizerModel::LlamaSpm => Some(2),
            TokenizerModel::Gpt2Bpe => token_to_id.get("<|end_of_text|>").copied(),
        };
        let default_unk = match model {
            TokenizerModel::LlamaSpm => Some(0),
            TokenizerModel::Gpt2Bpe => None,
        };

        let bos = file
            .metadata_u32("tokenizer.ggml.bos_token_id")
            .or(default_bos);
        let eos = file
            .metadata_u32("tokenizer.ggml.eos_token_id")
            .or(default_eos);
        let unk = file
            .metadata_u32("tokenizer.ggml.unknown_token_id")
            .or(default_unk);
        let eot = file
            .metadata_u32("tokenizer.ggml.eot_token_id")
            .or_else(|| token_to_id.get("<|eot_id|>").copied());
        let eom = file.metadata_u32("tokenizer.ggml.eom_token_id");
        // The misspelled fallback key is intentional: some GGUF exports write
        // "seperator_token_id" and the value must still be honored.
        let sep = file
            .metadata_u32("tokenizer.ggml.separator_token_id")
            .or_else(|| file.metadata_u32("tokenizer.ggml.seperator_token_id"));
        let pad = file.metadata_u32("tokenizer.ggml.padding_token_id");
        let mask = file.metadata_u32("tokenizer.ggml.mask_token_id");
        // Well-known end-of-turn markers used by chat templates. Some GGUFs set
        // `eos` to a generic `<|endoftext|>` but END EACH CHAT TURN with a distinct
        // token and never populate `eot_token_id` — notably Phi-3 (`<|end|>`), so
        // without this its chat turns never stop and the model rambles into new
        // turns. Purely additive: only ids that genuinely exist in this vocab are
        // added, and a vocab whose turn-end is already its `eos`/`eot` has an
        // unchanged stop set.
        const EOG_MARKER_TEXTS: &[&str] = &[
            "<|end|>",       // Phi-3
            "<|eot_id|>",    // Llama 3
            "<|im_end|>",    // ChatML / Qwen
            "<end_of_turn>", // Gemma
            "<|eom_id|>",
        ];
        let mut eog: BTreeSet<TokenId> = [eos, eot, eom].into_iter().flatten().collect();
        for marker in EOG_MARKER_TEXTS {
            if let Some(&id) = token_to_id.get(*marker) {
                eog.insert(id);
            }
        }

        validate_token_id("bos", bos, tokens.len())?;
        validate_token_id("eos", eos, tokens.len())?;
        validate_token_id("unk", unk, tokens.len())?;
        validate_token_id("eot", eot, tokens.len())?;
        validate_token_id("eom", eom, tokens.len())?;
        validate_token_id("sep", sep, tokens.len())?;
        validate_token_id("pad", pad, tokens.len())?;
        validate_token_id("mask", mask, tokens.len())?;

        Ok(Self {
            model,
            bpe_pre_tokenizer,
            tokens,
            token_to_id,
            byte_token_to_id,
            bpe_ranks,
            bpe_registry,
            special: SpecialTokens {
                bos,
                eos,
                eot,
                eom,
                unk,
                sep,
                pad,
                mask,
                eog,
            },
            config: TokenizerConfig {
                // gemma4 workaround: some gemma4 exports ship an incorrect
                // `add_bos_token = false`, but the model is always run with a
                // leading BOS, so the flag is force-overridden to true for the
                // "gemma4" model string (a no-op for exports that already ship
                // true). Without this the BOS is dropped and the whole forward
                // pass diverges from the model's canonical token stream.
                add_bos: if model_name == "gemma4" {
                    true
                } else {
                    file.metadata_bool("tokenizer.ggml.add_bos_token")
                        .unwrap_or(true)
                },
                add_eos: file
                    .metadata_bool("tokenizer.ggml.add_eos_token")
                    .unwrap_or(false),
                add_sep: file
                    .metadata_bool("tokenizer.ggml.add_sep_token")
                    .unwrap_or(false),
                add_space_prefix: file
                    .metadata_bool("tokenizer.ggml.add_space_prefix")
                    .unwrap_or(true),
                remove_extra_whitespaces: file
                    .metadata_bool("tokenizer.ggml.remove_extra_whitespaces")
                    .unwrap_or(false),
            },
            chat_template: file
                .metadata_string("tokenizer.chat_template")
                .map(str::to_owned),
        })
    }

    /// Text of a token id; `None` for `None` input or an out-of-range id.
    pub fn token_text(&self, id: Option<TokenId>) -> Option<&str> {
        id.and_then(|id| self.tokens.get(id as usize))
            .map(|token| token.text.as_str())
    }

    /// Exact-text vocabulary lookup.
    pub fn token_id(&self, text: &str) -> Option<TokenId> {
        self.token_to_id.get(text).copied()
    }

    /// Encode `text` to token ids. `add_special` wraps the result with BOS/EOS
    /// according to the tokenizer config; `parse_special` allows CONTROL-kind
    /// token texts appearing in `text` to match as single tokens (USER_DEFINED
    /// tokens always match regardless).
    pub fn encode(
        &self,
        text: &str,
        add_special: bool,
        parse_special: bool,
    ) -> Result<Vec<TokenId>> {
        let mut out = Vec::new();
        if add_special && self.config.add_bos {
            if let Some(bos) = self.special.bos {
                out.push(bos);
            }
        }

        match self.model {
            TokenizerModel::LlamaSpm => {
                let normalized = self.normalize_spm_text(text, parse_special);
                if !normalized.is_empty() {
                    out.extend(self.encode_piece(&normalized, parse_special)?);
                }
            }
            TokenizerModel::Gpt2Bpe => {
                if !text.is_empty() {
                    out.extend(self.encode_bpe_text(text, parse_special)?);
                }
            }
        }

        if add_special && self.config.add_eos {
            if let Some(eos) = self.special.eos {
                out.push(eos);
            }
        }
        Ok(out)
    }

    /// Decode token ids to text. With `remove_special`, ids in the special set
    /// and Control-kind / `<|...|>` chat-marker tokens are stripped from the
    /// output.
    pub fn decode(&self, token_ids: &[TokenId], remove_special: bool) -> Result<String> {
        if self.model == TokenizerModel::Gpt2Bpe {
            return self.decode_bpe(token_ids, remove_special);
        }

        let mut bytes = Vec::new();
        let mut text = String::new();

        for id in token_ids {
            if remove_special && self.is_special(*id) {
                continue;
            }
            let token = self.tokens.get(*id as usize).ok_or_else(|| {
                EngineError::InvalidTokenizerMetadata(format!("token id {id} out of range"))
            })?;
            if remove_special && (token.kind == TokenKind::Control || is_chat_control_marker(token))
            {
                continue;
            }
            if let Some(byte) = parse_byte_token(&token.text) {
                bytes.push(byte);
                continue;
            }
            flush_bytes(&mut bytes, &mut text)?;
            text.push_str(&token.text.replace(SPM_SPACE, " "));
        }
        flush_bytes(&mut bytes, &mut text)?;
        Ok(text)
    }

    /// Chat prompts are tokenized with special-token parsing for every model:
    /// a rendered chat template's control markers (e.g. SPM `</s>` between
    /// turns) must become control token ids, not literal text. Raw completion
    /// text is unaffected and keeps `parse_special: false` — special parsing
    /// does not spread to raw text.
    pub fn chat_prompt_parse_special(&self) -> bool {
        true
    }

    fn encode_bpe_text(&self, text: &str, parse_special: bool) -> Result<Vec<TokenId>> {
        let mut out = Vec::new();
        let mut byte_start = 0;

        while byte_start < text.len() {
            // The special-token partition runs in BOTH modes: USER_DEFINED
            // ("added") tokens are matched in raw text unconditionally; only
            // CONTROL tokens are gated by `parse_special`. Qwen vocabularies
            // mark <think>/<tool_call> as USER_DEFINED, so those must resolve
            // to single ids even with special parsing off, while a CONTROL
            // marker like <|im_start|> tokenizes as text in that mode.
            if let Some((token_text, token_len)) =
                self.longest_control_token_at(text, byte_start, parse_special)
            {
                if let Some(id) = self.token_to_id.get(token_text) {
                    out.push(*id);
                    byte_start += token_len;
                    continue;
                }
            }

            let byte_end = self
                .next_control_token_start(text, byte_start, parse_special)
                .unwrap_or(text.len());

            for segment in bpe_pretokenize_with(
                &text[byte_start..byte_end],
                self.bpe_pre_tokenizer.digit_group_max(),
                self.bpe_pre_tokenizer.fold_marks(),
            ) {
                self.encode_bpe_segment(segment, &mut out)?;
            }
            byte_start = byte_end;
        }

        Ok(out)
    }

    fn encode_bpe_segment(&self, segment: &str, out: &mut Vec<TokenId>) -> Result<()> {
        if segment.is_empty() {
            return Ok(());
        }

        let mut symbols: Vec<String> = segment
            .as_bytes()
            .iter()
            .map(|byte| bpe_byte_to_char(*byte).to_string())
            .collect();

        symbols = self.bpe_registry.merge_symbols(symbols);

        for symbol in symbols {
            let id = self.token_to_id.get(&symbol).copied().ok_or_else(|| {
                EngineError::InvalidTokenizerMetadata(format!(
                    "GPT-2/BPE token {symbol:?} is missing from tokenizer.ggml.tokens"
                ))
            })?;
            out.push(id);
        }
        Ok(())
    }

    fn decode_bpe(&self, token_ids: &[TokenId], remove_special: bool) -> Result<String> {
        let mut bytes = Vec::new();
        for id in token_ids {
            if remove_special && self.is_special(*id) {
                continue;
            }
            let token = self.tokens.get(*id as usize).ok_or_else(|| {
                EngineError::InvalidTokenizerMetadata(format!("token id {id} out of range"))
            })?;
            if remove_special && (token.kind == TokenKind::Control || is_chat_control_marker(token))
            {
                continue;
            }
            for ch in token.text.chars() {
                if let Some(byte) = bpe_char_to_byte(ch) {
                    bytes.push(byte);
                } else if !remove_special || token.kind != TokenKind::Control {
                    return Err(EngineError::InvalidTokenizerMetadata(format!(
                        "GPT-2/BPE token {:?} contains non-byte character {ch:?}",
                        token.text
                    )));
                }
            }
        }

        // A generated sequence can stop mid-multi-byte-character — e.g. truncated
        // by max_tokens partway through an emoji — leaving valid byte-tokens that
        // don't yet form complete UTF-8. That is normal model output, not corrupt
        // tokenizer metadata, so return the valid UTF-8 prefix and hold back the
        // incomplete trailing bytes instead of failing the whole request. Holding
        // the bytes back — rather than emitting a U+FFFD — lets a streaming
        // re-decode append the character cleanly once the next token completes it
        // (a transient U+FFFD would break prefix-based delta diffing). For
        // complete sequences the valid prefix is the whole string, byte-for-byte
        // identical to a strict decode.
        match std::str::from_utf8(&bytes) {
            Ok(text) => Ok(text.to_string()),
            Err(err) => Ok(std::str::from_utf8(&bytes[..err.valid_up_to()])
                .unwrap_or("")
                .to_string()),
        }
    }

    fn normalize_spm_text(&self, text: &str, parse_special: bool) -> String {
        let mut normalized = String::new();
        if text.is_empty() {
            return normalized;
        }
        // Always prepend the dummy `▁` when add_space_prefix is set, including
        // when the text already begins with whitespace — SentencePiece's
        // Metaspace normalization prepends unconditionally. The only exception:
        // under parse_special, a text that BEGINS with a special token gets no
        // prefix.
        if self.config.add_space_prefix
            && !(parse_special && self.longest_control_token_at(text, 0, true).is_some())
        {
            normalized.push(SPM_SPACE);
        }
        for ch in text.chars() {
            if ch == ' ' {
                normalized.push(SPM_SPACE);
            } else {
                normalized.push(ch);
            }
        }
        if parse_special {
            normalized
        } else {
            self.add_dummy_prefix_after_control_tokens(&normalized)
        }
    }

    fn add_dummy_prefix_after_control_tokens(&self, text: &str) -> String {
        if !self.config.add_space_prefix || text.is_empty() {
            return text.to_string();
        }

        let mut normalized = String::with_capacity(text.len());
        let mut byte_start = 0;
        while byte_start < text.len() {
            if let Some((token_text, token_len)) =
                self.longest_control_token_at(text, byte_start, true)
            {
                normalized.push_str(token_text);
                byte_start += token_len;

                let rest = &text[byte_start..];
                let next_is_control = self
                    .longest_control_token_at(text, byte_start, true)
                    .is_some();
                let should_insert_dummy_prefix =
                    self.should_insert_dummy_after_control(token_text, rest, next_is_control);
                if should_insert_dummy_prefix {
                    normalized.push(SPM_SPACE);
                }
                continue;
            }

            let ch = text[byte_start..]
                .chars()
                .next()
                .expect("byte_start is in-bounds");
            normalized.push(ch);
            byte_start += ch.len_utf8();
        }
        normalized
    }

    fn should_insert_dummy_after_control(
        &self,
        token_text: &str,
        rest: &str,
        next_is_control: bool,
    ) -> bool {
        if rest.is_empty() || next_is_control {
            return false;
        }

        // Instruct-template special cases (vocabs carrying [INST]/[/INST]):
        // no dummy between BOS and "[INST]", but always one after "[INST]".
        if self
            .token_text(self.special.bos)
            .is_some_and(|bos| token_text == bos)
            && rest.starts_with("[INST]")
            && self.token_to_id.contains_key("[INST]")
            && self.token_to_id.contains_key("[/INST]")
        {
            return false;
        }

        if token_text == "[INST]"
            && self.token_to_id.contains_key("[INST]")
            && self.token_to_id.contains_key("[/INST]")
        {
            return true;
        }

        !rest.starts_with(SPM_SPACE)
    }

    /// Longest special token whose text starts at `byte_start`. USER_DEFINED
    /// ("added") tokens always participate; CONTROL tokens only when
    /// `include_control` (the `parse_special` partition rule).
    fn longest_control_token_at<'a>(
        &'a self,
        text: &str,
        byte_start: usize,
        include_control: bool,
    ) -> Option<(&'a str, usize)> {
        if !text.is_char_boundary(byte_start) {
            return None;
        }

        // USER_DEFINED ("added") tokens always match. Qwen-family vocabs mark
        // <think>/</think> (and many <|...|> markers) as USER_DEFINED (type 4)
        // rather than CONTROL (type 3); without matching USER_DEFINED, a
        // rendered chat template's literal "</think>" tokenizes as text instead
        // of the single special token. CONTROL tokens participate only under
        // `include_control` (= `parse_special`).
        self.tokens
            .iter()
            .filter(|token| {
                matches!(token.kind, TokenKind::UserDefined)
                    || (include_control && matches!(token.kind, TokenKind::Control))
            })
            .filter(|token| !token.text.is_empty())
            .filter(|token| text[byte_start..].starts_with(&token.text))
            .max_by_key(|token| token.text.len())
            .map(|token| (token.text.as_str(), token.text.len()))
    }

    fn encode_piece(&self, piece: &str, parse_special: bool) -> Result<Vec<TokenId>> {
        if self.bpe_ranks.is_empty() && !parse_special {
            return self.encode_piece_greedy(piece);
        }

        let mut out = Vec::new();
        let mut byte_start = 0;
        while byte_start < piece.len() {
            if parse_special {
                if let Some((token_text, token_len)) =
                    self.longest_control_token_at(piece, byte_start, true)
                {
                    if let Some(id) = self.token_to_id.get(token_text) {
                        out.push(*id);
                        byte_start += token_len;
                        let rest = &piece[byte_start..];
                        let next_is_control = self
                            .longest_control_token_at(piece, byte_start, true)
                            .is_some();
                        if self.config.add_space_prefix
                            && self.should_insert_dummy_after_control(
                                token_text,
                                rest,
                                next_is_control,
                            )
                        {
                            if let Some(dummy_prefix) = self.token_to_id.get(&SPM_SPACE.to_string())
                            {
                                out.push(*dummy_prefix);
                            }
                        }
                        continue;
                    }
                }
            }

            let byte_end = if parse_special {
                self.next_control_token_start(piece, byte_start, true)
                    .unwrap_or(piece.len())
            } else {
                piece.len()
            };
            if self.bpe_ranks.is_empty() {
                if parse_special {
                    self.encode_spm_segment(&piece[byte_start..byte_end], &mut out)?;
                } else {
                    out.extend(self.encode_piece_greedy(&piece[byte_start..byte_end])?);
                }
            } else {
                self.encode_spm_segment(&piece[byte_start..byte_end], &mut out)?;
            }
            byte_start = byte_end;
        }
        Ok(out)
    }

    fn next_control_token_start(
        &self,
        text: &str,
        byte_start: usize,
        include_control: bool,
    ) -> Option<usize> {
        text[byte_start..]
            .char_indices()
            .map(|(offset, _)| byte_start + offset)
            .find(|idx| {
                self.longest_control_token_at(text, *idx, include_control)
                    .is_some()
            })
    }

    fn encode_spm_segment(&self, segment: &str, out: &mut Vec<TokenId>) -> Result<()> {
        if segment.is_empty() {
            return Ok(());
        }

        let symbols = if self.bpe_ranks.is_empty() {
            self.merge_spm_symbols_by_score(segment)
        } else {
            self.bpe_registry
                .merge_symbols(segment.chars().map(|ch| ch.to_string()).collect())
        };

        let mut unresolved = String::new();
        for symbol in symbols {
            // The multi-space (▁▁) deferral belongs to the score-merge path
            // only. Rank-based BPE (merges present, e.g. the gemma4 family)
            // merges multi-space runs into single vocab tokens — those vocabs
            // carry real ▁▁ / ▁▁▁ entries — and deferring them would diverge
            // from the model's canonical tokenization.
            if self.bpe_ranks.is_empty() && symbol.contains("▁▁") {
                unresolved.push_str(&symbol);
                continue;
            }

            if let Some(id) = self.token_to_id.get(&symbol).copied() {
                if !unresolved.is_empty() {
                    out.extend(self.encode_piece_greedy(&unresolved)?);
                    unresolved.clear();
                }
                out.push(id);
            } else {
                unresolved.push_str(&symbol);
            }
        }
        if !unresolved.is_empty() {
            out.extend(self.encode_piece_greedy(&unresolved)?);
        }
        Ok(())
    }

    fn merge_spm_symbols_by_score(&self, segment: &str) -> Vec<String> {
        let mut symbols: Vec<String> = segment.chars().map(|ch| ch.to_string()).collect();

        loop {
            let mut best: Option<(f32, usize)> = None;
            for idx in 0..symbols.len().saturating_sub(1) {
                let candidate = format!("{}{}", symbols[idx], symbols[idx + 1]);
                if candidate.contains("▁▁") {
                    continue;
                }
                let Some(id) = self.token_to_id.get(&candidate).copied() else {
                    continue;
                };
                let score = self.tokens[id as usize].score;
                match best {
                    Some((best_score, best_idx))
                        if score < best_score || (score == best_score && idx >= best_idx) => {}
                    _ => best = Some((score, idx)),
                }
            }

            let Some((_, idx)) = best else { break };
            symbols[idx] = format!("{}{}", symbols[idx], symbols[idx + 1]);
            symbols.remove(idx + 1);
        }

        symbols
    }

    fn encode_unknown_symbol_bytes(&self, symbol: &str, out: &mut Vec<TokenId>) -> Result<()> {
        for byte in symbol.as_bytes() {
            let id = self
                .byte_token_to_id
                .get(byte)
                .copied()
                .or(self.special.unk);
            match id {
                Some(id) => out.push(id),
                None => {
                    return Err(EngineError::InvalidTokenizerMetadata(format!(
                        "SPM byte fallback token <0x{byte:02X}> is missing"
                    )))
                }
            }
        }
        Ok(())
    }

    fn encode_piece_greedy(&self, piece: &str) -> Result<Vec<TokenId>> {
        let chars: Vec<(usize, char)> = piece.char_indices().collect();
        let mut out = Vec::new();
        let mut byte_start = 0;

        while byte_start < piece.len() {
            let mut best: Option<(usize, TokenId, f32)> = None;
            for byte_end in piece[byte_start..]
                .char_indices()
                .skip(1)
                .map(|(offset, _)| byte_start + offset)
                .chain(std::iter::once(piece.len()))
            {
                let candidate = &piece[byte_start..byte_end];
                if candidate.contains("▁▁") {
                    continue;
                }
                if let Some(id) = self.token_to_id.get(candidate) {
                    let score = self.tokens[*id as usize].score;
                    let len = byte_end - byte_start;
                    match best {
                        Some((best_len, _, best_score))
                            if len < best_len || (len == best_len && score <= best_score) => {}
                        _ => best = Some((len, *id, score)),
                    }
                }
            }

            if let Some((len, id, _)) = best {
                out.push(id);
                byte_start += len;
                continue;
            }

            let ch = chars
                .iter()
                .find(|(idx, _)| *idx == byte_start)
                .map(|(_, ch)| *ch)
                .ok_or_else(|| {
                    EngineError::InvalidTokenizerMetadata(
                        "internal UTF-8 tokenizer cursor error".to_string(),
                    )
                })?;
            let mut buf = [0u8; 4];
            self.encode_unknown_symbol_bytes(ch.encode_utf8(&mut buf), &mut out)?;
            byte_start += ch.len_utf8();
        }
        Ok(out)
    }

    fn is_special(&self, id: TokenId) -> bool {
        self.special.bos == Some(id)
            || self.special.eos == Some(id)
            || self.special.eot == Some(id)
            || self.special.eom == Some(id)
            || self.special.sep == Some(id)
            || self.special.pad == Some(id)
            || self.special.mask == Some(id)
    }
}

#[cfg(test)]
fn bpe_pretokenize(text: &str) -> Vec<&str> {
    // Test-only convenience wrapper: the default llama-bpe digit grouping.
    bpe_pretokenize_with(text, 3, false)
}

fn bpe_pretokenize_with(text: &str, digit_group_max: usize, fold_marks: bool) -> Vec<&str> {
    // GPT-2/BPE pre-tokenizer: a hand-rolled equivalent of the split regex
    //   (?i:'s|'t|'re|'ve|'m|'ll|'d)
    //   | [^\r\n\p{L}\p{N}]?\p{L}+
    //   | \p{N}{1,N}              (N = digit_group_max: 3 for llama-bpe, 1 for qwen2)
    //   |  ?[^\s\p{L}\p{N}]+[\r\n]*
    //   | \s*[\r\n]+
    //   | \s+(?!\S)
    //   | \s+
    // Keep the branch order identical: the whitespace branches intentionally
    // leave one prefix char behind when that enables the next token to be an
    // optional-prefix letters or punctuation segment. The ONLY dialect
    // difference (llama-bpe vs qwen2) is the digit-run cap.
    let mut segments = Vec::new();
    let mut byte_start = 0;

    while byte_start < text.len() {
        let byte_end = next_llama_bpe_segment_end(text, byte_start, digit_group_max, fold_marks);
        segments.push(&text[byte_start..byte_end]);
        byte_start = byte_end;
    }

    segments
}

fn next_llama_bpe_segment_end(
    text: &str,
    byte_start: usize,
    digit_group_max: usize,
    fold_marks: bool,
) -> usize {
    if let Some(end) = consume_contraction(text, byte_start) {
        return end;
    }
    if let Some(end) = consume_optional_prefix_letters(text, byte_start, fold_marks) {
        return end;
    }

    let ch = next_char(text, byte_start).expect("byte_start is in-bounds");
    if is_number(ch) {
        return consume_digits(text, byte_start, digit_group_max);
    }
    if let Some(end) = consume_optional_space_punctuation(text, byte_start, fold_marks) {
        return end;
    }
    if let Some(end) = consume_whitespace_with_newline(text, byte_start) {
        return end;
    }
    if is_whitespace(ch) {
        return consume_whitespace_before_nonspace(text, byte_start);
    }

    byte_start + ch.len_utf8()
}

fn consume_contraction(text: &str, byte_start: usize) -> Option<usize> {
    ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"]
        .into_iter()
        .find_map(|suffix| {
            text[byte_start..]
                .get(..suffix.len())
                .filter(|candidate| candidate.eq_ignore_ascii_case(suffix))
                .map(|_| byte_start + suffix.len())
        })
}

fn consume_optional_prefix_letters(
    text: &str,
    byte_start: usize,
    fold_marks: bool,
) -> Option<usize> {
    let ch = next_char(text, byte_start).expect("byte_start is in-bounds");
    if is_letter_class(ch, fold_marks) {
        return Some(consume_letters(text, byte_start, fold_marks));
    }
    if ch == '\r' || ch == '\n' || is_number(ch) {
        return None;
    }

    let next_idx = byte_start + ch.len_utf8();
    let next = (next_idx < text.len()).then(|| next_char(text, next_idx))??;
    is_letter_class(next, fold_marks).then(|| consume_letters(text, next_idx, fold_marks))
}

fn consume_optional_space_punctuation(
    text: &str,
    byte_start: usize,
    fold_marks: bool,
) -> Option<usize> {
    let ch = next_char(text, byte_start).expect("byte_start is in-bounds");
    let punctuation_start = if ch == ' ' {
        let next_idx = byte_start + ch.len_utf8();
        let next = (next_idx < text.len()).then(|| next_char(text, next_idx))??;
        if is_punctuation_for_bpe(next, fold_marks) {
            next_idx
        } else {
            return None;
        }
    } else if is_punctuation_for_bpe(ch, fold_marks) {
        byte_start
    } else {
        return None;
    };

    let mut idx = punctuation_start;
    while idx < text.len() {
        let ch = next_char(text, idx).expect("idx is in-bounds");
        if !is_punctuation_for_bpe(ch, fold_marks) {
            break;
        }
        idx += ch.len_utf8();
    }
    while idx < text.len() {
        let ch = next_char(text, idx).expect("idx is in-bounds");
        if ch != '\n' && ch != '\r' {
            break;
        }
        idx += ch.len_utf8();
    }
    Some(idx)
}

fn consume_whitespace_with_newline(text: &str, byte_start: usize) -> Option<usize> {
    let ch = next_char(text, byte_start).expect("byte_start is in-bounds");
    if !is_whitespace(ch) {
        return None;
    }

    let mut idx = byte_start;
    let mut last_newline_end = None;
    while idx < text.len() {
        let ch = next_char(text, idx).expect("idx is in-bounds");
        if !is_whitespace(ch) {
            break;
        }
        idx += ch.len_utf8();
        if ch == '\n' || ch == '\r' {
            last_newline_end = Some(idx);
        }
    }
    last_newline_end
}

fn consume_whitespace_before_nonspace(text: &str, byte_start: usize) -> usize {
    let whitespace_end = consume_whitespace(text, byte_start);
    if whitespace_end == text.len() {
        return whitespace_end;
    }

    // Implements \s+(?!\S): if a whitespace run is followed by a non-space,
    // leave one horizontal space for the optional-prefix branch that follows.
    let chars: Vec<(usize, char)> = text[byte_start..whitespace_end]
        .char_indices()
        .map(|(offset, ch)| (byte_start + offset, ch))
        .collect();
    if chars.len() > 1 {
        chars[chars.len() - 1].0
    } else {
        whitespace_end
    }
}

fn next_char(text: &str, byte_start: usize) -> Option<char> {
    text[byte_start..].chars().next()
}

fn is_letter(ch: char) -> bool {
    ch.is_alphabetic()
}

fn is_number(ch: char) -> bool {
    ch.is_numeric()
}

fn is_whitespace(ch: char) -> bool {
    ch.is_whitespace()
}

/// Unicode `\p{M}` (Mn | Mc | Me) via a generated inclusive-range table — the
/// qwen35 pre-tokenizer folds these into the letter class. NOTE `is_letter`
/// (Rust `is_alphabetic` = derived Alphabetic) already covers the
/// Other_Alphabetic subset of marks (e.g. Devanagari matras, Arabic harakat);
/// this table adds the rest (viramas, NFD accents, enclosing marks), so the
/// union of the two matches strict general-category `[\p{L}\p{M}]`.
fn is_mark(ch: char) -> bool {
    const MARK_RANGES: &[(u32, u32)] = include!("tokenizer_mark_ranges.rs.inc");
    let cp = ch as u32;
    MARK_RANGES
        .binary_search_by(|&(lo, hi)| {
            if hi < cp {
                std::cmp::Ordering::Less
            } else if lo > cp {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        })
        .is_ok()
}

/// The pre-tokenizer's letter class: `\p{L}` for llama-bpe/qwen2,
/// `[\p{L}\p{M}]` for qwen35 (`fold_marks`).
fn is_letter_class(ch: char, fold_marks: bool) -> bool {
    is_letter(ch) || (fold_marks && is_mark(ch))
}

fn is_punctuation_for_bpe(ch: char, fold_marks: bool) -> bool {
    !is_whitespace(ch) && !is_letter_class(ch, fold_marks) && !is_number(ch)
}

fn consume_letters(text: &str, byte_start: usize, fold_marks: bool) -> usize {
    let mut idx = byte_start;
    while idx < text.len() {
        let ch = next_char(text, idx).expect("idx is in-bounds");
        if !is_letter_class(ch, fold_marks) {
            break;
        }
        idx += ch.len_utf8();
    }
    idx
}

fn consume_digits(text: &str, byte_start: usize, max_digits: usize) -> usize {
    let mut idx = byte_start;
    let mut count = 0;
    while idx < text.len() && count < max_digits {
        let ch = next_char(text, idx).expect("idx is in-bounds");
        if !is_number(ch) {
            break;
        }
        idx += ch.len_utf8();
        count += 1;
    }
    idx
}

fn consume_whitespace(text: &str, byte_start: usize) -> usize {
    let mut idx = byte_start;
    while idx < text.len() {
        let ch = next_char(text, idx).expect("idx is in-bounds");
        if !is_whitespace(ch) {
            break;
        }
        idx += ch.len_utf8();
    }
    idx
}

fn bpe_byte_to_char(byte: u8) -> char {
    // The GPT-2 byte-to-visible-character bijection: printable bytes map to
    // themselves; every other byte maps to 256 + its position among the
    // non-printable bytes.
    let byte = u32::from(byte);
    if (33..=126).contains(&byte) || (161..=172).contains(&byte) || (174..=255).contains(&byte) {
        return char::from_u32(byte).expect("visible byte maps to Unicode scalar");
    }

    let offset = (0..byte)
        .filter(|candidate| {
            !((33..=126).contains(candidate)
                || (161..=172).contains(candidate)
                || (174..=255).contains(candidate))
        })
        .count() as u32;
    char::from_u32(256 + offset).expect("GPT-2 byte fallback maps to Unicode scalar")
}

fn bpe_char_to_byte(ch: char) -> Option<u8> {
    (0..=u8::MAX).find(|byte| bpe_byte_to_char(*byte) == ch)
}

fn validate_token_id(name: &str, id: Option<TokenId>, len: usize) -> Result<()> {
    if let Some(id) = id {
        if id as usize >= len {
            return Err(EngineError::InvalidTokenizerMetadata(format!(
                "{name} token id {id} out of range for vocab size {len}"
            )));
        }
    }
    Ok(())
}

fn parse_byte_token(text: &str) -> Option<u8> {
    let hex = text.strip_prefix("<0x")?.strip_suffix('>')?;
    if hex.len() != 2 {
        return None;
    }
    u8::from_str_radix(hex, 16).ok()
}

fn flush_bytes(bytes: &mut Vec<u8>, text: &mut String) -> Result<()> {
    if bytes.is_empty() {
        return Ok(());
    }
    // SPM byte-fallback can likewise end mid-character when generation is truncated;
    // push the valid UTF-8 prefix and hold back any incomplete trailing bytes rather
    // than erroring. Identical to a strict decode for complete sequences.
    let taken = std::mem::take(bytes);
    match std::str::from_utf8(&taken) {
        Ok(decoded) => text.push_str(decoded),
        Err(err) => text.push_str(std::str::from_utf8(&taken[..err.valid_up_to()]).unwrap_or("")),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        bpe_pretokenize, bpe_pretokenize_with, is_chat_control_marker, is_mark, BpeRegistry, Token,
        TokenKind,
    };

    fn tok(text: &str, kind: TokenKind) -> Token {
        Token {
            id: 0,
            text: text.to_string(),
            score: 0.0,
            kind,
        }
    }

    #[test]
    fn llama3_bpe_signature_matches_only_the_llama3_vocab() {
        use super::is_llama3_bpe_signature;
        // Minimal stand-in for the Llama-3 vocab: 128,256 entries with the five
        // stable chat markers at their canonical ids.
        let mut v = vec![String::new(); 128_256];
        v[128_000] = "<|begin_of_text|>".to_string();
        v[128_001] = "<|end_of_text|>".to_string();
        v[128_006] = "<|start_header_id|>".to_string();
        v[128_007] = "<|end_header_id|>".to_string();
        v[128_009] = "<|eot_id|>".to_string();
        assert!(is_llama3_bpe_signature(&v));

        // Wrong vocab size never matches (Qwen's 151,936, or a truncated vocab).
        assert!(!is_llama3_bpe_signature(&v[..128_255]));
        let qwen_sized = vec!["x".to_string(); 151_936];
        assert!(!is_llama3_bpe_signature(&qwen_sized));

        // Missing any core marker fails — this is what refuses a de-pre'd non-Llama-3
        // vocab that happens to be 128,256 tokens.
        let mut missing = v.clone();
        missing[128_009] = "<|not_eot|>".to_string();
        assert!(!is_llama3_bpe_signature(&missing));
    }

    #[test]
    fn resolve_gpt2_pre_tokenizer_gates_the_missing_pre_recovery() {
        use super::{resolve_gpt2_pre_tokenizer, BpePreTokenizer};
        let mut sig = vec![String::new(); 128_256];
        sig[128_000] = "<|begin_of_text|>".to_string();
        sig[128_001] = "<|end_of_text|>".to_string();
        sig[128_006] = "<|start_header_id|>".to_string();
        sig[128_007] = "<|end_header_id|>".to_string();
        sig[128_009] = "<|eot_id|>".to_string();
        let no_sig = vec!["x".to_string(); 128_256];

        // Explicit dialects resolve regardless of the vocab.
        assert!(matches!(
            resolve_gpt2_pre_tokenizer(Some("llama-bpe"), &no_sig),
            Ok(BpePreTokenizer::Llama3)
        ));
        assert!(matches!(
            resolve_gpt2_pre_tokenizer(Some("qwen2"), &no_sig),
            Ok(BpePreTokenizer::Qwen2)
        ));
        assert!(matches!(
            resolve_gpt2_pre_tokenizer(Some("qwen35"), &no_sig),
            Ok(BpePreTokenizer::Qwen35)
        ));

        // Missing pre + Llama-3 signature => recovered as Llama3 (the fix).
        assert!(matches!(
            resolve_gpt2_pre_tokenizer(None, &sig),
            Ok(BpePreTokenizer::Llama3)
        ));

        // Missing pre WITHOUT the signature => still refused (guards a de-labeled Qwen).
        assert!(resolve_gpt2_pre_tokenizer(None, &no_sig).is_err());

        // An explicit-but-unknown pre is refused EVEN WITH the signature — we only
        // rescue an absent key, never override a stated (if unrecognized) dialect.
        assert!(resolve_gpt2_pre_tokenizer(Some("smaug-bpe"), &sig).is_err());
    }

    // Parity gate for the missing-`pre` Llama-3 rescue: a Llama-3 GGUF that omits
    // tokenizer.ggml.pre must tokenize BYTE-IDENTICALLY to a known-good
    // pre=llama-bpe Llama-3 GGUF. Ignored by default (needs the multi-GB models);
    // run locally with both env vars set:
    //   CAMELID_LLAMA3_MISSING_PRE_GGUF, CAMELID_LLAMA3_REFERENCE_GGUF
    #[test]
    #[ignore = "needs real Llama-3 GGUFs; set CAMELID_LLAMA3_MISSING_PRE_GGUF and CAMELID_LLAMA3_REFERENCE_GGUF"]
    fn missing_pre_llama3_tokenizes_identically_to_canonical() {
        use super::Tokenizer;
        use std::path::Path;
        let a = std::env::var("CAMELID_LLAMA3_MISSING_PRE_GGUF")
            .expect("set CAMELID_LLAMA3_MISSING_PRE_GGUF");
        let b = std::env::var("CAMELID_LLAMA3_REFERENCE_GGUF")
            .expect("set CAMELID_LLAMA3_REFERENCE_GGUF");
        let ga = crate::gguf::read_metadata(Path::new(&a)).expect("read missing-pre gguf");
        let gb = crate::gguf::read_metadata(Path::new(&b)).expect("read reference gguf");
        // Guard against a tautological pass: the reference MUST carry an explicit
        // pre=llama-bpe (a genuine oracle), and the file under test MUST actually omit
        // the key (so it drives the recovery branch, not the normal path). Without
        // these, pointing both env vars at missing-pre files would pass trivially.
        assert_eq!(
            gb.metadata_string("tokenizer.ggml.pre"),
            Some("llama-bpe"),
            "CAMELID_LLAMA3_REFERENCE_GGUF must be an explicit pre=llama-bpe oracle"
        );
        assert_eq!(
            ga.metadata_string("tokenizer.ggml.pre"),
            None,
            "CAMELID_LLAMA3_MISSING_PRE_GGUF must actually omit tokenizer.ggml.pre"
        );
        let ta = Tokenizer::from_gguf(&ga)
            .expect("missing-pre Llama-3 gguf must now load (llama-bpe recovered)");
        let tb = Tokenizer::from_gguf(&gb).expect("reference gguf loads");
        // This proves the RECOVERED tokenizer equals an explicit pre=llama-bpe oracle
        // over strings that exercise the split regex (contractions, multi-digit runs,
        // whitespace, unicode, chat markers). The llama-bpe-vs-qwen digit-grouping
        // discrimination itself is covered by qwen2_pretokenizer_splits_digits_singly.
        let battery = [
            "It's a test, don't you think? We'll see.",
            "1234567890 and 42 plus 007 and 100000",
            "The quick brown fox jumps over 13 lazy dogs.",
            "café — naïve — Zürich — 你好 — 🚀",
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nhi<|eot_id|>",
            "    leading and    inner   spaces\tand\ttabs",
        ];
        for s in battery {
            let ea = ta.encode(s, false, true).expect("encode missing-pre");
            let eb = tb.encode(s, false, true).expect("encode reference");
            assert_eq!(
                ea, eb,
                "token mismatch for {s:?}:\n  missing-pre: {ea:?}\n  reference:   {eb:?}"
            );
        }
    }

    #[test]
    fn chat_control_markers_are_stripped_but_think_tags_are_kept() {
        // Phi-3 / ChatML <|...|> markers are turn scaffolding → strippable.
        assert!(is_chat_control_marker(&tok(
            "<|end|>",
            TokenKind::UserDefined
        )));
        assert!(is_chat_control_marker(&tok(
            "<|assistant|>",
            TokenKind::UserDefined
        )));
        assert!(is_chat_control_marker(&tok(
            "<|im_end|>",
            TokenKind::UserDefined
        )));
        // Qwen3 reasoning tags are content (and <...>, not <|...|>) → preserved.
        assert!(!is_chat_control_marker(&tok(
            "<think>",
            TokenKind::UserDefined
        )));
        assert!(!is_chat_control_marker(&tok(
            "</think>",
            TokenKind::UserDefined
        )));
        // Normal/content tokens are never markers regardless of shape.
        assert!(!is_chat_control_marker(&tok("<|end|>", TokenKind::Normal)));
        assert!(!is_chat_control_marker(&tok(
            "hello",
            TokenKind::UserDefined
        )));
    }

    #[test]
    fn bpe_registry_uses_ranked_heap_priority_for_merges() {
        let registry = BpeRegistry::from_merges(vec![
            "a b".to_string(),
            "ab c".to_string(),
            "c d".to_string(),
            "abc d".to_string(),
        ]);

        assert_eq!(registry.len(), 4);
        assert_eq!(
            registry.merge_symbols(vec![
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
                "d".to_string(),
            ]),
            vec!["abcd".to_string()]
        );
    }

    #[test]
    fn bpe_registry_prefers_lowest_rank_over_leftmost_pair() {
        let registry = BpeRegistry::from_merges(vec![
            "b c".to_string(),
            "bc d".to_string(),
            "a b".to_string(),
        ]);

        assert_eq!(
            registry.merge_symbols(vec![
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
                "d".to_string(),
            ]),
            vec!["a".to_string(), "bcd".to_string()]
        );
    }

    #[test]
    fn llama_bpe_pretokenizer_matches_core_llama3_shapes() {
        assert_eq!(bpe_pretokenize("hello world"), vec!["hello", " world"]);
        assert_eq!(bpe_pretokenize("it's"), vec!["it", "'s"]);
        assert_eq!(bpe_pretokenize("WE'LL"), vec!["WE", "'LL"]);
        assert_eq!(bpe_pretokenize("1234"), vec!["123", "4"]);
        assert_eq!(bpe_pretokenize("  hello"), vec![" ", " hello"]);
        assert_eq!(bpe_pretokenize(" !\n\n"), vec![" !\n\n"]);
        assert_eq!(bpe_pretokenize("foo...bar"), vec!["foo", "...", "bar"]);
        assert_eq!(bpe_pretokenize("hi\n  "), vec!["hi", "\n", "  "]);
    }

    #[test]
    fn llama_bpe_pretokenizer_matches_llama3_regex_edge_cases() {
        let cases = [
            ("!hello", vec!["!hello"]),
            ("\thello", vec!["\thello"]),
            ("  \thello", vec!["  ", "\thello"]),
            ("don't", vec!["don", "'t"]),
            ("can'T", vec!["can", "'T"]),
            ("abc12345def", vec!["abc", "123", "45", "def"]),
            ("café déjà", vec!["café", " déjà"]),
            ("!!!\r\nnext", vec!["!!!\r\n", "next"]),
            ("line\r\n  next", vec!["line", "\r\n", " ", " next"]),
            ("tabs\t\tword", vec!["tabs", "\t", "\tword"]),
            ("\t!!!", vec!["\t", "!!!"]),
            ("  !!!", vec![" ", " !!!"]),
            ("hello🙂world", vec!["hello", "🙂world"]),
            ("1२٣4", vec!["1२٣", "4"]),
            ("   ", vec!["   "]),
            ("\r\nhello", vec!["\r\n", "hello"]),
        ];

        for (input, expected) in cases {
            assert_eq!(bpe_pretokenize(input), expected, "input {input:?}");
        }
    }

    #[test]
    fn qwen2_pretokenizer_splits_digits_singly_but_matches_llama3_otherwise() {
        // The ONLY difference between the qwen2 (digit cap 1) and llama-bpe
        // (digit cap 3) pre-tokenizers is digit grouping: qwen2 emits each digit
        // as its own piece (`\p{N}`), llama-bpe groups runs of up to three
        // (`\p{N}{1,3}`).
        const QWEN2: usize = 1;
        const LLAMA3: usize = 3;

        // Digits split one-at-a-time under qwen2 …
        assert_eq!(
            bpe_pretokenize_with("1234", QWEN2, false),
            vec!["1", "2", "3", "4"]
        );
        assert_eq!(
            bpe_pretokenize_with("abc12345def", QWEN2, false),
            vec!["abc", "1", "2", "3", "4", "5", "def"]
        );
        // … while the rest of the grammar is byte-for-byte identical to llama-bpe.
        for input in [
            "hello world",
            "it's",
            "WE'LL",
            "  hello",
            "foo...bar",
            "café déjà",
            "line\r\n  next",
            "hello🙂world",
        ] {
            assert_eq!(
                bpe_pretokenize_with(input, QWEN2, false),
                bpe_pretokenize_with(input, LLAMA3, false),
                "non-digit input {input:?} must tokenize identically under both dialects"
            );
        }
    }

    #[test]
    fn qwen35_pretokenizer_folds_combining_marks_into_letter_runs() {
        // qwen35's regex letter branch is `[\p{L}\p{M}]+` and its punctuation
        // class excludes `\p{M}`. These lock the split behavior at the unit
        // level.
        const QWEN35: usize = 1;

        // NFD: base letter + U+0301 combining acute stays one letter run.
        assert_eq!(
            bpe_pretokenize_with("cafe\u{301} bar", QWEN35, true),
            vec!["cafe\u{301}", " bar"]
        );
        // Without folding (qwen2), the bare accent splits off as punctuation.
        assert_eq!(
            bpe_pretokenize_with("cafe\u{301} bar", QWEN35, false),
            vec!["cafe", "\u{301}", " bar"]
        );
        // Devanagari virama (U+094D, Mn but NOT Other_Alphabetic — invisible to
        // `char::is_alphabetic`) must not break a cluster: नमस्ते is one run.
        assert_eq!(
            bpe_pretokenize_with("\u{928}\u{92E}\u{938}\u{94D}\u{924}\u{947}", QWEN35, true),
            vec!["\u{928}\u{92E}\u{938}\u{94D}\u{924}\u{947}"]
        );
        // A mark with no preceding letter still starts a letter-class run.
        assert_eq!(
            bpe_pretokenize_with("\u{301}x", QWEN35, true),
            vec!["\u{301}x"]
        );
        // Punctuation runs stop at a mark (punctuation class excludes \p{M}).
        assert_eq!(
            bpe_pretokenize_with("!!\u{301}!!", QWEN35, true),
            vec!["!!", "\u{301}", "!!"]
        );
    }

    #[test]
    fn is_mark_matches_unicode_m_category_samples() {
        assert!(is_mark('\u{301}')); // combining acute (Mn)
        assert!(is_mark('\u{94D}')); // Devanagari virama (Mn)
        assert!(is_mark('\u{93E}')); // Devanagari matra (Mc)
        assert!(is_mark('\u{20DD}')); // enclosing circle (Me)
        assert!(!is_mark('a'));
        assert!(!is_mark('!'));
        assert!(!is_mark(' '));
        assert!(!is_mark('\u{4E2D}')); // CJK letter
    }
}

#[cfg(test)]
mod real_model_fixtures {
    use super::*;
    use crate::gguf;
    use std::path::Path;

    /// Known-good tokenizations of the test model, asserted exactly.
    /// Set CAMELID_ENTERPRISE_TEST_MODEL to a local GGUF to enable; the ids
    /// are verifiable against the model's own vocabulary and merge table.
    /// Rows: (name, text, add_special, parse_special, expected ids).
    const FIXTURES: &[(&str, &str, bool, bool, &[u32])] = &[
        ("empty", "", true, false, &[128000]),
        ("hello", "Hello, world!", true, false, &[128000, 9906, 11, 1917, 0]),
        ("math", "2+2=", true, false, &[128000, 17, 10, 17, 28]),
        ("leading-space", " leading space", true, false, &[128000, 6522, 3634]),
        ("repeat", "camel camel camel", true, false, &[128000, 94421, 50252, 50252]),
        ("unicode", "héllo wörld 東京 🦙", true, false, &[128000, 71, 19010, 385, 289, 9603, 509, 119109, 11410, 99, 247]),
        ("digits", "1234567890", true, false, &[128000, 4513, 10961, 16474, 15]),
        ("contractions", "don't we'll it's", true, false, &[128000, 15357, 956, 584, 3358, 433, 596]),
        ("code", "fn main() { println!(\"hi\"); }", true, false, &[128000, 8998, 1925, 368, 314, 14069, 17667, 6151, 5146, 335]),
        ("whitespace-mix", "a\n\tb  c", true, false, &[128000, 64, 198, 2282, 220, 272]),
        ("no-bos", "Hello, world!", false, false, &[9906, 11, 1917, 0]),
        ("special-parsed", "<|begin_of_text|>test", false, true, &[128000, 1985]),
        ("special-literal", "<|begin_of_text|>test", false, false, &[27, 91, 7413, 3659, 4424, 91, 29, 1985]),
    ];

    #[test]
    fn real_model_tokenizations_match_known_good_ids() {
        let Ok(path) = std::env::var("CAMELID_ENTERPRISE_TEST_MODEL") else { return };
        let file = gguf::read_metadata(Path::new(&path)).unwrap();
        let tok = Tokenizer::from_gguf(&file).unwrap();
        for (name, text, add_special, parse_special, expected) in FIXTURES {
            let got = tok.encode(text, *add_special, *parse_special).unwrap();
            assert_eq!(&got, expected, "fixture {name} diverged");
            if !add_special && !parse_special {
                let round = tok.decode(&got, false).unwrap();
                assert_eq!(&round, text, "fixture {name} decode round-trip diverged");
            }
        }
    }
}
