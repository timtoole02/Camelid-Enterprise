//! GGUF container parsing.
//!
//! GGUF is a single-file model container: a little-endian header (magic,
//! version, counts), a metadata key/value block, a tensor-descriptor block,
//! and then the tensor data starting at an aligned offset. This module parses
//! and validates everything *before* the tensor data — descriptors carry the
//! absolute byte range of each tensor so callers can map or read data lazily.
//!
//! Validation is fail-closed: unknown tensor types, non-contiguous tensor
//! offsets, out-of-range byte spans, duplicate names or keys, and oversized
//! strings/arrays are all hard errors rather than best-effort skips. A model
//! file this module accepts is one whose every declared byte range is in
//! bounds.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::{ErrorKind, Read},
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};

use crate::{EngineError, Result};

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const DEFAULT_ALIGNMENT: u64 = 32;
const MAX_TENSOR_DIMS: u32 = 4;
const MAX_TENSOR_NAME: usize = 64;
const MAX_STRING_BYTES: u64 = 16 * 1024 * 1024;
const MAX_ARRAY_LEN: u64 = 1_000_000;

/// A metadata value. GGUF metadata is a flat map of string keys to typed
/// scalars, strings, or single-level arrays.
#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(untagged)]
pub enum MetadataValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<MetadataValue>),
    U64(u64),
    I64(i64),
    F64(f64),
}

/// On-wire tensor element types. The numeric ids and block layouts are fixed
/// by the GGUF/GGML wire format; ids this engine has never seen parse to
/// `Unknown` and are rejected as soon as a byte size is needed.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TensorType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    IQ4NL,
    IQ4XS,
    Tq1_0,
    Tq2_0,
    I8,
    I16,
    I32,
    I64,
    F64,
    BF16,
    /// 64-element / 36-byte superblock: 4 UE4M3 sub-scales + 32 bytes of
    /// packed E2M1 nibbles (4.5 bits per weight). Wire type id 40.
    NVFP4,
    Unknown(i32),
}

impl TensorType {
    pub fn from_id(value: i32) -> Self {
        match value {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            15 => Self::Q8K,
            20 => Self::IQ4NL,
            23 => Self::IQ4XS,
            34 => Self::Tq1_0,
            35 => Self::Tq2_0,
            24 => Self::I8,
            25 => Self::I16,
            26 => Self::I32,
            27 => Self::I64,
            28 => Self::F64,
            30 => Self::BF16,
            40 => Self::NVFP4,
            other => Self::Unknown(other),
        }
    }

    /// `(elements_per_block, bytes_per_block)` for each storable type.
    /// `None` for types whose layout this engine does not know — a tensor of
    /// such a type cannot be sized and is rejected.
    pub fn layout(self) -> Option<(u64, u64)> {
        match self {
            Self::F32 => Some((1, 4)),
            Self::F16 => Some((1, 2)),
            // f16 scale + 16 nibble bytes
            Self::Q4_0 => Some((32, 18)),
            // f16 scale + f16 min + 16 nibble bytes (20, not 18)
            Self::Q4_1 => Some((32, 20)),
            Self::Q5_0 | Self::Q5_1 => Some((32, 22)),
            // f16 scale + 32 signed bytes
            Self::Q8_0 => Some((32, 34)),
            Self::Q8_1 => Some((32, 36)),
            Self::Q2K => Some((256, 84)),
            Self::Q3K => Some((256, 110)),
            Self::Q4K => Some((256, 144)),
            Self::Q5K => Some((256, 176)),
            Self::Q6K => Some((256, 210)),
            Self::Q8K => Some((256, 292)),
            Self::IQ4NL => Some((32, 18)),
            // f16 scale(2) + scale-high u16(2) + 4 low-scale bytes + 128 nibble bytes
            Self::IQ4XS => Some((256, 136)),
            // f16 scale(2) + 4 high bytes + 48 packed ternary bytes
            Self::Tq1_0 => Some((256, 54)),
            // 64 packed ternary bytes + f16 scale(2)
            Self::Tq2_0 => Some((256, 66)),
            // 4 UE4M3 sub-scales + 32 packed E2M1 nibble bytes
            Self::NVFP4 => Some((64, 36)),
            Self::I8 => Some((1, 1)),
            Self::I16 | Self::BF16 => Some((1, 2)),
            Self::I32 => Some((1, 4)),
            Self::I64 | Self::F64 => Some((1, 8)),
            Self::Unknown(_) => None,
        }
    }
}

/// One tensor's entry in the descriptor block. Byte ranges are validated
/// against the file length at parse time.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct TensorDescriptor {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub tensor_type: TensorType,
    /// Offset within the tensor-data region.
    pub relative_offset: u64,
    /// Offset from the start of the file.
    pub absolute_offset: u64,
    pub n_bytes: u64,
}

/// A parsed GGUF header: everything before the tensor data.
#[derive(Debug, Clone, Serialize)]
pub struct GgufFile {
    pub path: PathBuf,
    pub version: u32,
    pub tensor_count: i64,
    pub metadata_count: i64,
    pub alignment: u64,
    /// Aligned file offset where tensor data begins.
    pub data_start_offset: u64,
    pub metadata: BTreeMap<String, MetadataValue>,
    pub tensors: Vec<TensorDescriptor>,
}

impl GgufFile {
    pub fn architecture(&self) -> Option<&str> {
        self.metadata_string("general.architecture")
    }

    pub fn model_name(&self) -> Option<&str> {
        self.metadata_string("general.name")
    }

    pub fn metadata_string(&self, key: &str) -> Option<&str> {
        match self.metadata.get(key) {
            Some(MetadataValue::String(value)) => Some(value.as_str()),
            _ => None,
        }
    }

    pub fn metadata_bool(&self, key: &str) -> Option<bool> {
        match self.metadata.get(key) {
            Some(MetadataValue::Bool(value)) => Some(*value),
            _ => None,
        }
    }

    pub fn metadata_u32(&self, key: &str) -> Option<u32> {
        match self.metadata.get(key) {
            Some(MetadataValue::U32(value)) => Some(*value),
            Some(MetadataValue::U64(value)) => (*value).try_into().ok(),
            _ => None,
        }
    }

    pub fn metadata_f32(&self, key: &str) -> Option<f32> {
        match self.metadata.get(key) {
            Some(MetadataValue::F32(value)) => Some(*value),
            Some(MetadataValue::F64(value)) => Some(*value as f32),
            _ => None,
        }
    }

    /// Required `array<string>` metadata (e.g. the tokenizer vocabulary).
    pub fn metadata_array_strings(&self, key: &str) -> Result<Vec<String>> {
        self.metadata_array_strings_optional(key)?
            .ok_or_else(|| EngineError::InvalidGguf(format!("required metadata {key} is missing")))
    }

    pub fn metadata_array_strings_optional(&self, key: &str) -> Result<Option<Vec<String>>> {
        self.typed_array_optional(key, "array<string>", |value| match value {
            MetadataValue::String(value) => Some(value.clone()),
            _ => None,
        })
    }

    pub fn metadata_array_f32_optional(&self, key: &str) -> Result<Option<Vec<f32>>> {
        self.typed_array_optional(key, "array<float>", |value| match value {
            MetadataValue::F32(value) => Some(*value),
            _ => None,
        })
    }

    pub fn metadata_array_u32_optional(&self, key: &str) -> Result<Option<Vec<u32>>> {
        self.typed_array_optional(key, "array<uint>", |value| match value {
            MetadataValue::U32(value) => Some(*value),
            MetadataValue::I32(value) => u32::try_from(*value).ok(),
            MetadataValue::U64(value) => u32::try_from(*value).ok(),
            _ => None,
        })
    }

    pub fn metadata_array_i32_optional(&self, key: &str) -> Result<Option<Vec<i32>>> {
        self.typed_array_optional(key, "array<int>", |value| match value {
            MetadataValue::I32(value) => Some(*value),
            MetadataValue::U32(value) => i32::try_from(*value).ok(),
            _ => None,
        })
    }

    pub fn metadata_array_bools_optional(&self, key: &str) -> Result<Option<Vec<bool>>> {
        self.typed_array_optional(key, "array<bool>", |value| match value {
            MetadataValue::Bool(value) => Some(*value),
            _ => None,
        })
    }

    /// Shared shape for the typed-array accessors: absent key is `Ok(None)`;
    /// a present key whose value is not an array, or any element the
    /// `extract` closure cannot represent (wrong type, out of range), is an
    /// error naming the expected type.
    fn typed_array_optional<T>(
        &self,
        key: &str,
        expected: &str,
        extract: impl Fn(&MetadataValue) -> Option<T>,
    ) -> Result<Option<Vec<T>>> {
        match self.metadata.get(key) {
            None => Ok(None),
            Some(MetadataValue::Array(values)) => values
                .iter()
                .map(|value| {
                    extract(value).ok_or_else(|| {
                        EngineError::InvalidGguf(format!("metadata {key} must be {expected}"))
                    })
                })
                .collect::<Result<Vec<_>>>()
                .map(Some),
            Some(_) => Err(EngineError::InvalidGguf(format!(
                "metadata {key} must be {expected}"
            ))),
        }
    }
}

/// Parse a GGUF file's metadata and tensor descriptors.
pub fn read_metadata(path: &Path) -> Result<GgufFile> {
    let file_len = fs::metadata(path)
        .map_err(|source| EngineError::Io { path: path.to_path_buf(), source })?
        .len();
    read_metadata_with_len(path, file_len)
}

/// Parse GGUF metadata, validating tensor byte ranges against `declared_len`
/// instead of the on-disk length.
///
/// All metadata and tensor descriptors sit at the front of a GGUF, so a caller
/// holding only a downloaded header *prefix* can still parse fully by passing
/// the model's true length: every bounds check runs against the real size and
/// no tensor data is touched.
pub fn read_metadata_with_len(path: &Path, declared_len: u64) -> Result<GgufFile> {
    let file = File::open(path)
        .map_err(|source| EngineError::Io { path: path.to_path_buf(), source })?;
    let mut cursor = Cursor::new(file, path.to_path_buf());

    let magic = cursor.read_bytes(4)?;
    if magic != GGUF_MAGIC {
        return Err(EngineError::InvalidGguf("bad magic; expected GGUF".to_string()));
    }

    let version = cursor.read_u32()?;
    if !(2..=3).contains(&version) {
        return Err(EngineError::UnsupportedGguf(format!(
            "version {version}; expected v2 or v3"
        )));
    }

    let tensor_count = cursor.read_i64()?;
    let metadata_count = cursor.read_i64()?;
    if tensor_count < 0 || metadata_count < 0 {
        return Err(EngineError::InvalidGguf(
            "negative tensor or metadata count".to_string(),
        ));
    }

    let mut metadata = BTreeMap::new();
    for _ in 0..metadata_count {
        let key = cursor.read_string()?;
        let value = read_value(&mut cursor)?;
        if metadata.insert(key.clone(), value).is_some() {
            return Err(EngineError::InvalidGguf(format!("duplicate metadata key {key}")));
        }
    }

    let alignment = match metadata.get("general.alignment") {
        Some(MetadataValue::U32(value)) => u64::from(*value),
        Some(MetadataValue::U64(value)) => *value,
        Some(_) => {
            return Err(EngineError::InvalidGguf(
                "general.alignment has non-integer type".to_string(),
            ))
        }
        None => DEFAULT_ALIGNMENT,
    };
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(EngineError::InvalidGguf(format!("invalid alignment {alignment}")));
    }

    let mut raw_tensors = Vec::new();
    for _ in 0..tensor_count {
        let name = cursor.read_string()?;
        if name.len() >= MAX_TENSOR_NAME {
            return Err(EngineError::InvalidGguf(format!(
                "tensor name {name} exceeds the {MAX_TENSOR_NAME}-byte limit"
            )));
        }
        let n_dimensions = cursor.read_u32()?;
        if n_dimensions == 0 || n_dimensions > MAX_TENSOR_DIMS {
            return Err(EngineError::InvalidGguf(format!(
                "tensor {name} has invalid dimension count {n_dimensions}"
            )));
        }
        let mut dimensions = Vec::with_capacity(n_dimensions as usize);
        for _ in 0..n_dimensions {
            let dim = cursor.read_i64()?;
            if dim < 0 {
                return Err(EngineError::InvalidGguf(format!(
                    "tensor {name} has negative dimension {dim}"
                )));
            }
            dimensions.push(dim as u64);
        }
        let tensor_type = TensorType::from_id(cursor.read_i32()?);
        let relative_offset = cursor.read_u64()?;
        raw_tensors.push((name, dimensions, tensor_type, relative_offset));
    }

    let data_start_offset = align_to(cursor.position(), alignment)?;
    if data_start_offset > declared_len {
        return Err(EngineError::InvalidGguf(
            "aligned tensor data start is beyond end of file".to_string(),
        ));
    }

    // Descriptors must tile the data region contiguously in declaration order:
    // each tensor starts exactly where the previous one's aligned span ended.
    // This closes the door on overlapping or gapped ranges up front.
    let mut tensors = Vec::with_capacity(raw_tensors.len());
    let mut seen_names = BTreeSet::new();
    let mut expected_offset = 0u64;
    for (name, dimensions, tensor_type, relative_offset) in raw_tensors {
        if !seen_names.insert(name.clone()) {
            return Err(EngineError::InvalidGguf(format!("duplicate tensor name {name}")));
        }
        if relative_offset != expected_offset {
            return Err(EngineError::InvalidGguf(format!(
                "tensor {name} offset {relative_offset} is not contiguous; expected {expected_offset}"
            )));
        }
        let n_bytes = tensor_nbytes(&name, &dimensions, tensor_type)?;
        let absolute_offset = data_start_offset
            .checked_add(relative_offset)
            .ok_or_else(|| EngineError::InvalidGguf(format!("tensor {name} absolute offset overflow")))?;
        let end = absolute_offset
            .checked_add(n_bytes)
            .ok_or_else(|| EngineError::InvalidGguf(format!("tensor {name} byte range overflow")))?;
        if end > declared_len {
            return Err(EngineError::InvalidGguf(format!(
                "tensor {name} data extends beyond end of file"
            )));
        }
        tensors.push(TensorDescriptor {
            name,
            dimensions,
            tensor_type,
            relative_offset,
            absolute_offset,
            n_bytes,
        });
        expected_offset = align_to(
            relative_offset
                .checked_add(n_bytes)
                .ok_or_else(|| EngineError::InvalidGguf("tensor offset overflow".to_string()))?,
            alignment,
        )?;
    }

    Ok(GgufFile {
        path: path.to_path_buf(),
        version,
        tensor_count,
        metadata_count,
        alignment,
        data_start_offset,
        metadata,
        tensors,
    })
}

fn read_value(cursor: &mut Cursor) -> Result<MetadataValue> {
    let ty = cursor.read_i32()?;
    read_value_of_type(cursor, ty)
}

fn read_value_of_type(cursor: &mut Cursor, ty: i32) -> Result<MetadataValue> {
    Ok(match ty {
        0 => MetadataValue::U8(cursor.read_u8()?),
        1 => MetadataValue::I8(cursor.read_i8()?),
        2 => MetadataValue::U16(cursor.read_u16()?),
        3 => MetadataValue::I16(cursor.read_i16()?),
        4 => MetadataValue::U32(cursor.read_u32()?),
        5 => MetadataValue::I32(cursor.read_i32()?),
        6 => MetadataValue::F32(cursor.read_f32()?),
        7 => MetadataValue::Bool(cursor.read_bool()?),
        8 => MetadataValue::String(cursor.read_string()?),
        9 => {
            let element_ty = cursor.read_i32()?;
            if element_ty == 9 {
                return Err(EngineError::UnsupportedGguf("nested metadata arrays".to_string()));
            }
            let len = cursor.read_u64()?;
            if len > MAX_ARRAY_LEN {
                return Err(EngineError::InvalidGguf(format!("metadata array too large: {len}")));
            }
            let mut values = Vec::with_capacity(len as usize);
            for _ in 0..len {
                values.push(read_value_of_type(cursor, element_ty)?);
            }
            MetadataValue::Array(values)
        }
        10 => MetadataValue::U64(cursor.read_u64()?),
        11 => MetadataValue::I64(cursor.read_i64()?),
        12 => MetadataValue::F64(cursor.read_f64()?),
        other => return Err(EngineError::UnsupportedGguf(format!("metadata value type {other}"))),
    })
}

fn tensor_nbytes(name: &str, dimensions: &[u64], tensor_type: TensorType) -> Result<u64> {
    let (block_size, type_size) = tensor_type.layout().ok_or_else(|| {
        EngineError::UnsupportedGguf(format!(
            "tensor {name} has unknown wire type {tensor_type:?}"
        ))
    })?;
    let first_dim = *dimensions.first().unwrap_or(&1);
    if first_dim % block_size != 0 {
        return Err(EngineError::InvalidGguf(format!(
            "tensor {name} first dimension {first_dim} is not divisible by block size {block_size}"
        )));
    }
    let mut elements = 1u64;
    for dim in dimensions {
        elements = elements
            .checked_mul(*dim)
            .ok_or_else(|| EngineError::InvalidGguf(format!("tensor {name} element count overflow")))?;
    }
    elements
        .checked_div(block_size)
        .and_then(|blocks| blocks.checked_mul(type_size))
        .ok_or_else(|| EngineError::InvalidGguf(format!("tensor {name} byte size overflow")))
}

fn align_to(value: u64, alignment: u64) -> Result<u64> {
    let add = alignment - 1;
    value
        .checked_add(add)
        .map(|v| v & !add)
        .ok_or_else(|| EngineError::InvalidGguf("alignment overflow".to_string()))
}

/// Buffered little-endian reader that tracks its file position for the
/// alignment computation and turns unexpected EOF into a format error.
struct Cursor {
    reader: File,
    path: PathBuf,
    pos: u64,
}

impl Cursor {
    fn new(reader: File, path: PathBuf) -> Self {
        Self { reader, path, pos: 0 }
    }

    fn position(&self) -> u64 {
        self.pos
    }

    fn read_exact_into(&mut self, out: &mut [u8]) -> Result<()> {
        self.reader.read_exact(out).map_err(|source| {
            if source.kind() == ErrorKind::UnexpectedEof {
                EngineError::InvalidGguf("unexpected end of file".to_string())
            } else {
                EngineError::Io { path: self.path.clone(), source }
            }
        })?;
        self.pos = self
            .pos
            .checked_add(out.len() as u64)
            .ok_or_else(|| EngineError::InvalidGguf("cursor overflow".to_string()))?;
        Ok(())
    }

    fn read_bytes(&mut self, n: usize) -> Result<Vec<u8>> {
        let mut out = vec![0; n];
        self.read_exact_into(&mut out)?;
        Ok(out)
    }

    fn read_u8(&mut self) -> Result<u8> {
        let mut b = [0; 1];
        self.read_exact_into(&mut b)?;
        Ok(b[0])
    }

    fn read_i8(&mut self) -> Result<i8> {
        Ok(self.read_u8()? as i8)
    }

    fn read_bool(&mut self) -> Result<bool> {
        Ok(self.read_u8()? != 0)
    }

    fn read_u16(&mut self) -> Result<u16> {
        let mut b = [0; 2];
        self.read_exact_into(&mut b)?;
        Ok(u16::from_le_bytes(b))
    }

    fn read_i16(&mut self) -> Result<i16> {
        let mut b = [0; 2];
        self.read_exact_into(&mut b)?;
        Ok(i16::from_le_bytes(b))
    }

    fn read_u32(&mut self) -> Result<u32> {
        let mut b = [0; 4];
        self.read_exact_into(&mut b)?;
        Ok(u32::from_le_bytes(b))
    }

    fn read_i32(&mut self) -> Result<i32> {
        let mut b = [0; 4];
        self.read_exact_into(&mut b)?;
        Ok(i32::from_le_bytes(b))
    }

    fn read_f32(&mut self) -> Result<f32> {
        Ok(f32::from_bits(self.read_u32()?))
    }

    fn read_u64(&mut self) -> Result<u64> {
        let mut b = [0; 8];
        self.read_exact_into(&mut b)?;
        Ok(u64::from_le_bytes(b))
    }

    fn read_i64(&mut self) -> Result<i64> {
        let mut b = [0; 8];
        self.read_exact_into(&mut b)?;
        Ok(i64::from_le_bytes(b))
    }

    fn read_f64(&mut self) -> Result<f64> {
        Ok(f64::from_bits(self.read_u64()?))
    }

    fn read_string(&mut self) -> Result<String> {
        let len = self.read_u64()?;
        if len > MAX_STRING_BYTES {
            return Err(EngineError::InvalidGguf(format!("string too large: {len}")));
        }
        let bytes = self.read_bytes(len as usize)?;
        String::from_utf8(bytes)
            .map_err(|_| EngineError::InvalidGguf("invalid UTF-8 string".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal in-memory GGUF writer for tests: v3, little-endian, one
    /// metadata block and one tensor block, data aligned per `alignment`.
    struct TestGguf {
        bytes: Vec<u8>,
        metadata: Vec<(String, Vec<u8>)>,
        tensors: Vec<(String, Vec<u64>, i32, u64, u64)>, // name, dims, type id, rel offset, data bytes
        alignment: u64,
    }

    impl TestGguf {
        fn new() -> Self {
            Self { bytes: Vec::new(), metadata: Vec::new(), tensors: Vec::new(), alignment: 32 }
        }

        fn meta_string(mut self, key: &str, value: &str) -> Self {
            let mut v = 8i32.to_le_bytes().to_vec();
            v.extend((value.len() as u64).to_le_bytes());
            v.extend(value.as_bytes());
            self.metadata.push((key.to_string(), v));
            self
        }

        fn meta_u32(mut self, key: &str, value: u32) -> Self {
            let mut v = 4i32.to_le_bytes().to_vec();
            v.extend(value.to_le_bytes());
            self.metadata.push((key.to_string(), v));
            self
        }

        fn meta_string_array(mut self, key: &str, values: &[&str]) -> Self {
            let mut v = 9i32.to_le_bytes().to_vec();
            v.extend(8i32.to_le_bytes()); // element type: string
            v.extend((values.len() as u64).to_le_bytes());
            for s in values {
                v.extend((s.len() as u64).to_le_bytes());
                v.extend(s.as_bytes());
            }
            self.metadata.push((key.to_string(), v));
            self
        }

        /// F32 tensor whose data is zeros; `rel_offset` must be the aligned
        /// running offset for the file to be valid.
        fn tensor_f32(mut self, name: &str, dims: &[u64], rel_offset: u64) -> Self {
            let n_bytes = dims.iter().product::<u64>() * 4;
            self.tensors.push((name.to_string(), dims.to_vec(), 0, rel_offset, n_bytes));
            self
        }

        fn write_string(out: &mut Vec<u8>, s: &str) {
            out.extend((s.len() as u64).to_le_bytes());
            out.extend(s.as_bytes());
        }

        fn build(mut self) -> Vec<u8> {
            self.bytes.extend(b"GGUF");
            self.bytes.extend(3u32.to_le_bytes());
            self.bytes.extend((self.tensors.len() as i64).to_le_bytes());
            self.bytes.extend((self.metadata.len() as i64).to_le_bytes());
            for (key, value) in &self.metadata {
                Self::write_string(&mut self.bytes, key);
                self.bytes.extend(value);
            }
            for (name, dims, type_id, rel_offset, _) in &self.tensors {
                Self::write_string(&mut self.bytes, name);
                self.bytes.extend((dims.len() as u32).to_le_bytes());
                for d in dims {
                    self.bytes.extend((*d as i64).to_le_bytes());
                }
                self.bytes.extend(type_id.to_le_bytes());
                self.bytes.extend(rel_offset.to_le_bytes());
            }
            // Pad to the aligned data start, then emit zeroed tensor data with
            // inter-tensor alignment padding.
            let align = |v: u64| (v + self.alignment - 1) & !(self.alignment - 1);
            while (self.bytes.len() as u64) < align(self.bytes.len() as u64) {
                self.bytes.push(0);
            }
            let mut data_len = 0u64;
            for (_, _, _, rel_offset, n_bytes) in &self.tensors {
                data_len = data_len.max(rel_offset + n_bytes);
            }
            self.bytes.extend(std::iter::repeat_n(0u8, data_len as usize));
            self.bytes
        }
    }

    fn write_temp(bytes: &[u8], name: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!("engine-core-gguf-test-{name}-{}.gguf", std::process::id()));
        fs::write(&path, bytes).unwrap();
        path
    }

    #[test]
    fn parses_a_minimal_valid_file() {
        let bytes = TestGguf::new()
            .meta_string("general.architecture", "llama")
            .meta_string("general.name", "Test Model")
            .meta_u32("llama.block_count", 2)
            .meta_string_array("tokenizer.ggml.tokens", &["<s>", "a", "b"])
            .tensor_f32("token_embd.weight", &[32, 2], 0)
            .tensor_f32("output_norm.weight", &[32], 256) // 32*2*4 = 256, already 32-aligned
            .build();
        let path = write_temp(&bytes, "minimal");
        let parsed = read_metadata(&path).unwrap();
        fs::remove_file(&path).ok();

        assert_eq!(parsed.version, 3);
        assert_eq!(parsed.architecture(), Some("llama"));
        assert_eq!(parsed.model_name(), Some("Test Model"));
        assert_eq!(parsed.metadata_u32("llama.block_count"), Some(2));
        assert_eq!(
            parsed.metadata_array_strings("tokenizer.ggml.tokens").unwrap(),
            vec!["<s>", "a", "b"]
        );
        assert_eq!(parsed.tensors.len(), 2);
        assert_eq!(parsed.tensors[0].name, "token_embd.weight");
        assert_eq!(parsed.tensors[0].n_bytes, 256);
        assert_eq!(parsed.tensors[1].relative_offset, 256);
        assert_eq!(
            parsed.tensors[1].absolute_offset,
            parsed.data_start_offset + 256
        );
    }

    #[test]
    fn rejects_bad_magic() {
        let path = write_temp(b"NOPE01234567", "magic");
        let err = read_metadata(&path).unwrap_err();
        fs::remove_file(&path).ok();
        assert!(matches!(err, EngineError::InvalidGguf(_)), "{err}");
    }

    #[test]
    fn rejects_unsupported_version() {
        let mut bytes = TestGguf::new().build();
        bytes[4..8].copy_from_slice(&9u32.to_le_bytes());
        let path = write_temp(&bytes, "version");
        let err = read_metadata(&path).unwrap_err();
        fs::remove_file(&path).ok();
        assert!(matches!(err, EngineError::UnsupportedGguf(_)), "{err}");
    }

    #[test]
    fn rejects_non_contiguous_tensor_offsets() {
        let bytes = TestGguf::new()
            .tensor_f32("a.weight", &[32], 0)
            .tensor_f32("b.weight", &[32], 512) // aligned end of a.weight is 128
            .build();
        let path = write_temp(&bytes, "gap");
        let err = read_metadata(&path).unwrap_err();
        fs::remove_file(&path).ok();
        let msg = err.to_string();
        assert!(msg.contains("not contiguous"), "{msg}");
    }

    #[test]
    fn rejects_tensor_data_past_end_of_file() {
        let mut bytes = TestGguf::new().tensor_f32("a.weight", &[32, 4], 0).build();
        bytes.truncate(bytes.len() - 64);
        let path = write_temp(&bytes, "trunc");
        let err = read_metadata(&path).unwrap_err();
        fs::remove_file(&path).ok();
        assert!(err.to_string().contains("beyond end of file"), "{err}");
    }

    #[test]
    fn prefix_parse_with_declared_len_succeeds() {
        // A header-only prefix parses when the caller declares the true length.
        let full = TestGguf::new().tensor_f32("a.weight", &[32, 4], 0).build();
        let parsed_full = read_metadata(&write_temp(&full, "full")).unwrap();
        let header_only = &full[..parsed_full.data_start_offset as usize];
        let path = write_temp(header_only, "prefix");
        let parsed = read_metadata_with_len(&path, full.len() as u64).unwrap();
        fs::remove_file(&path).ok();
        assert_eq!(parsed.tensors[0].n_bytes, 32 * 4 * 4);
    }

    #[test]
    fn wire_type_ids_and_layouts_are_pinned() {
        assert_eq!(TensorType::from_id(8), TensorType::Q8_0);
        assert_eq!(TensorType::Q8_0.layout(), Some((32, 34)));
        assert_eq!(TensorType::from_id(40), TensorType::NVFP4);
        assert_eq!(TensorType::NVFP4.layout(), Some((64, 36)));
        assert_eq!(TensorType::from_id(41), TensorType::Unknown(41));
        assert_eq!(TensorType::Unknown(41).layout(), None);
    }

    /// Full-file integration parse, gated on a real model being present.
    /// Set CAMELID_ENTERPRISE_TEST_MODEL to a local GGUF path to enable.
    #[test]
    fn parses_a_real_model_when_available() {
        let Ok(path) = std::env::var("CAMELID_ENTERPRISE_TEST_MODEL") else { return };
        let parsed = read_metadata(Path::new(&path)).unwrap();
        assert!(parsed.tensor_count > 0);
        assert!(parsed.architecture().is_some());
        let last = parsed.tensors.last().unwrap();
        assert!(last.absolute_offset + last.n_bytes <= fs::metadata(&path).unwrap().len());
    }
}
