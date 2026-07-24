//! GGUF-backed tensor loading.
//!
//! `TensorStore` maps tensor names to their GGUF descriptors and reads each
//! tensor's payload bytes with a fresh seek + exact read per call — no
//! memory mapping, no caching. The loaders materialize `CpuTensor`s in the
//! storage the compute kernels expect: eager f32 for float and low-bit
//! types, decoded blocks for 2-D Q8_0 linears, and raw wire bytes (with
//! empty f32 `data`) for the quantized 2-D linears whose full f32 expansion
//! would not fit in memory.

use std::{
    collections::HashMap,
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    sync::Arc,
};

use crate::gguf::{GgufFile, TensorDescriptor, TensorType};
use crate::tensor::blocks::{
    decode_bf16_tensor, decode_f16_tensor, decode_f32_tensor, decode_iq4_nl_tensor,
    decode_iq4_xs_tensor, decode_q2_k_tensor, decode_q3_k_tensor, decode_q4_0_tensor,
    decode_q4_1_tensor, decode_q4_k_tensor, decode_q5_0_tensor, decode_q5_1_tensor,
    decode_q5_k_tensor, decode_q6_k_tensor, decode_q8_0_blocks, decode_q8_0_tensor,
    decode_q8_k_tensor, decode_tq1_0_tensor, decode_tq2_0_tensor, TensorShape,
};
use crate::tensor::{CpuTensor, Q8_0TensorBlocks};
use crate::{EngineError, Result};

pub struct TensorStore {
    path: PathBuf,
    descriptors: HashMap<String, TensorDescriptor>,
}

impl TensorStore {
    pub fn open(path: impl AsRef<Path>, gguf: &GgufFile) -> Self {
        let descriptors = gguf
            .tensors
            .iter()
            .cloned()
            .map(|desc| (desc.name.clone(), desc))
            .collect();
        Self {
            path: path.as_ref().to_path_buf(),
            descriptors,
        }
    }

    pub fn descriptor(&self, name: &str) -> Result<&TensorDescriptor> {
        self.descriptors
            .get(name)
            .ok_or_else(|| EngineError::TensorNotFound(name.to_string()))
    }

    pub fn tensor_bytes(&self, name: &str) -> Result<Vec<u8>> {
        let desc = self.descriptor(name)?;
        let len = usize::try_from(desc.n_bytes).map_err(|_| {
            EngineError::InvalidTensorData(format!("tensor {name} byte length does not fit usize"))
        })?;
        let mut file = File::open(&self.path).map_err(|source| EngineError::Io {
            path: self.path.clone(),
            source,
        })?;
        file.seek(SeekFrom::Start(desc.absolute_offset))
            .map_err(|source| EngineError::Io {
                path: self.path.clone(),
                source,
            })?;
        let mut bytes = vec![0u8; len];
        file.read_exact(&mut bytes)
            .map_err(|source| EngineError::Io {
                path: self.path.clone(),
                source,
            })?;
        Ok(bytes)
    }

    pub fn load_q8_0_blocks(&self, name: &str) -> Result<Q8_0TensorBlocks> {
        let desc = self.descriptor(name)?.clone();
        if desc.tensor_type != TensorType::Q8_0 {
            return Err(EngineError::UnsupportedTensorType(format!(
                "tensor {name} has storage type {:?}; q8_0 block-only load requires Q8_0",
                desc.tensor_type
            )));
        }
        let bytes = self.tensor_bytes(name)?;
        let shape = TensorShape::from_gguf_dims(&desc.dimensions)?;
        let expected_elements = shape.element_count()?;
        let blocks = decode_q8_0_blocks(name, &bytes, expected_elements)?;
        Ok(Q8_0TensorBlocks {
            name: name.to_string(),
            shape,
            blocks,
        })
    }

    pub fn load_q8_0_block_backed_linear(&self, name: &str) -> Result<CpuTensor> {
        self.load_q8_0_block_backed_linear_as(name, name)
    }

    /// Load a 2-D Q8_0 linear as decoded blocks with EMPTY f32 `data`; only
    /// the block-streaming kernels can consume the result. Non-Q8_0 and
    /// non-2-D tensors fall back to the f32 loader, renamed to `tensor_name`.
    pub fn load_q8_0_block_backed_linear_as(
        &self,
        source_name: &str,
        tensor_name: &str,
    ) -> Result<CpuTensor> {
        let desc = self.descriptor(source_name)?.clone();
        if desc.tensor_type != TensorType::Q8_0 {
            let mut tensor = self.load_cpu_f32(source_name)?;
            tensor.name = tensor_name.to_string();
            return Ok(tensor);
        }
        let shape = TensorShape::from_gguf_dims(&desc.dimensions)?;
        if shape.dims.len() != 2 {
            let mut tensor = self.load_cpu_f32(source_name)?;
            tensor.name = tensor_name.to_string();
            return Ok(tensor);
        }
        let expected_elements = shape.element_count()?;
        let bytes = self.tensor_bytes(source_name)?;
        let blocks = decode_q8_0_blocks(source_name, &bytes, expected_elements)?;
        CpuTensor::from_q8_0_blocks(tensor_name, shape, blocks)
    }

    /// Load a 2-D K-quant linear retaining ONLY the raw super-block wire
    /// bytes — NO f32 `data` materialization. This mirrors
    /// [`CpuTensor::from_q8_0_blocks`] (which leaves `data` empty): a large
    /// model fully decoded to f32 would not fit in memory, so consumers must
    /// stream the wire bytes. Supported types are Q4_K, Q5_K, and Q6_K —
    /// the only K-quants with block-streaming CPU kernels; a 2-D Q2_K or
    /// Q3_K linear is refused loudly rather than silently materialized.
    /// Non-K-quant and non-2-D tensors fall back to the f32 loader.
    pub fn load_kquant_wire_linear(&self, name: &str) -> Result<CpuTensor> {
        let desc = self.descriptor(name)?.clone();
        let shape = TensorShape::from_gguf_dims(&desc.dimensions)?;
        let is_kquant = matches!(
            desc.tensor_type,
            TensorType::Q4K
                | TensorType::Q5K
                | TensorType::Q6K
                | TensorType::Q2K
                | TensorType::Q3K
        );
        if !is_kquant || shape.dims.len() != 2 {
            return self.load_cpu_f32(name);
        }
        if matches!(desc.tensor_type, TensorType::Q2K | TensorType::Q3K) {
            return Err(EngineError::UnsupportedTensorType(format!(
                "tensor {name} has storage type {:?}; wire-only linear load supports Q4_K, Q5_K, Q6_K (a 2-D {:?} linear has no block-streaming CPU kernel, and full f32 materialization of a wire-only linear is refused)",
                desc.tensor_type, desc.tensor_type
            )));
        }
        let bytes = self.tensor_bytes(name)?;
        let mut q4_k_wire_bytes = None;
        let mut q5_k_wire_bytes = None;
        let mut q6_k_wire_bytes = None;
        match desc.tensor_type {
            TensorType::Q4K => q4_k_wire_bytes = Some(Arc::new(bytes)),
            TensorType::Q5K => q5_k_wire_bytes = Some(Arc::new(bytes)),
            TensorType::Q6K => q6_k_wire_bytes = Some(Arc::new(bytes)),
            _ => unreachable!(),
        }
        Ok(CpuTensor {
            name: name.to_string(),
            shape,
            dtype: crate::tensor::RuntimeDType::F32,
            source_type: Some(desc.tensor_type),
            q8_0_blocks: None,
            q4_k_wire_bytes,
            q5_k_wire_bytes,
            q6_k_wire_bytes,
            tq2_0_wire_bytes: None,
            iq4_xs_wire_bytes: None,
            data: Vec::new(),
        })
    }

    /// Load a TQ2_0 (ternary) 2-D linear by retaining its raw wire bytes only
    /// — no f32 materialization. The ternary block-dot streams these
    /// directly. Mirrors [`Self::load_kquant_wire_linear`]. Falls back to
    /// f32 for non-TQ2_0 / non-2-D tensors.
    pub fn load_tq2_0_wire_linear(&self, name: &str) -> Result<CpuTensor> {
        let desc = self.descriptor(name)?.clone();
        let shape = TensorShape::from_gguf_dims(&desc.dimensions)?;
        if !matches!(desc.tensor_type, TensorType::Tq2_0) || shape.dims.len() != 2 {
            return self.load_cpu_f32(name);
        }
        let bytes = self.tensor_bytes(name)?;
        Ok(CpuTensor {
            name: name.to_string(),
            shape,
            dtype: crate::tensor::RuntimeDType::F32,
            source_type: Some(desc.tensor_type),
            q8_0_blocks: None,
            q4_k_wire_bytes: None,
            q5_k_wire_bytes: None,
            q6_k_wire_bytes: None,
            tq2_0_wire_bytes: Some(Arc::new(bytes)),
            iq4_xs_wire_bytes: None,
            data: Vec::new(),
        })
    }

    /// Load an IQ4_XS (i-quant) 2-D linear by retaining its raw wire bytes
    /// only — no f32 materialization. The i-quant block-dot streams these
    /// directly. Mirrors [`Self::load_tq2_0_wire_linear`]. Falls back to f32
    /// for non-IQ4_XS / non-2-D tensors.
    pub fn load_iq4_xs_wire_linear(&self, name: &str) -> Result<CpuTensor> {
        let desc = self.descriptor(name)?.clone();
        let shape = TensorShape::from_gguf_dims(&desc.dimensions)?;
        if !matches!(desc.tensor_type, TensorType::IQ4XS) || shape.dims.len() != 2 {
            return self.load_cpu_f32(name);
        }
        let bytes = self.tensor_bytes(name)?;
        Ok(CpuTensor {
            name: name.to_string(),
            shape,
            dtype: crate::tensor::RuntimeDType::F32,
            source_type: Some(desc.tensor_type),
            q8_0_blocks: None,
            q4_k_wire_bytes: None,
            q5_k_wire_bytes: None,
            q6_k_wire_bytes: None,
            tq2_0_wire_bytes: None,
            iq4_xs_wire_bytes: Some(Arc::new(bytes)),
            data: Vec::new(),
        })
    }

    /// Eagerly decode a tensor to row-major f32 `data`. Q8_0 tensors also
    /// retain their decoded blocks; Q4_K and Q6_K also retain their raw wire
    /// bytes alongside the decoded f32 (Q5_K deliberately does not — the
    /// wire-only loader covers it with empty `data` instead). The retention
    /// choices never change `data` itself.
    pub fn load_cpu_f32(&self, name: &str) -> Result<CpuTensor> {
        let desc = self.descriptor(name)?.clone();
        let bytes = self.tensor_bytes(name)?;
        let shape = TensorShape::from_gguf_dims(&desc.dimensions)?;
        let expected_elements = shape.element_count()?;
        let mut q8_0_blocks = None;
        let mut q4_k_wire_bytes = None;
        let mut q6_k_wire_bytes = None;
        let data = match desc.tensor_type {
            TensorType::F32 => decode_f32_tensor(name, &bytes, expected_elements)?,
            TensorType::F16 => decode_f16_tensor(name, &bytes, expected_elements)?,
            TensorType::BF16 => decode_bf16_tensor(name, &bytes, expected_elements)?,
            TensorType::Q8_0 => {
                let decoded = decode_q8_0_tensor(name, &bytes, expected_elements)?;
                q8_0_blocks = Some(decode_q8_0_blocks(name, &bytes, expected_elements)?);
                decoded
            }
            TensorType::Q4_0 => decode_q4_0_tensor(name, &bytes, expected_elements)?,
            TensorType::Q4_1 => decode_q4_1_tensor(name, &bytes, expected_elements)?,
            TensorType::Q5_0 => decode_q5_0_tensor(name, &bytes, expected_elements)?,
            TensorType::Q5_1 => decode_q5_1_tensor(name, &bytes, expected_elements)?,
            TensorType::Q2K => decode_q2_k_tensor(name, &bytes, expected_elements)?,
            TensorType::Q3K => decode_q3_k_tensor(name, &bytes, expected_elements)?,
            TensorType::Q4K => {
                // The GGUF bytes ARE the 144-byte super-block wire layout the
                // block-streaming kernels read; keep them alongside the decoded f32.
                q4_k_wire_bytes = Some(Arc::new(bytes.clone()));
                decode_q4_k_tensor(name, &bytes, expected_elements)?
            }
            TensorType::Q5K => decode_q5_k_tensor(name, &bytes, expected_elements)?,
            TensorType::Q6K => {
                // 210-byte super-block wire layout, read directly by the q6_k kernels.
                q6_k_wire_bytes = Some(Arc::new(bytes.clone()));
                decode_q6_k_tensor(name, &bytes, expected_elements)?
            }
            TensorType::Q8K => decode_q8_k_tensor(name, &bytes, expected_elements)?,
            TensorType::IQ4NL => decode_iq4_nl_tensor(name, &bytes, expected_elements)?,
            TensorType::IQ4XS => decode_iq4_xs_tensor(name, &bytes, expected_elements)?,
            TensorType::Tq1_0 => decode_tq1_0_tensor(name, &bytes, expected_elements)?,
            TensorType::Tq2_0 => decode_tq2_0_tensor(name, &bytes, expected_elements)?,
            other => {
                return Err(EngineError::UnsupportedTensorType(format!(
                    "tensor {name} has unsupported storage type {other:?}; supported for CPU f32 load: F32, F16, BF16, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K, IQ4_NL, IQ4_XS, TQ1_0, TQ2_0"
                )))
            }
        };
        Ok(CpuTensor {
            name: name.to_string(),
            shape,
            dtype: crate::tensor::RuntimeDType::F32,
            source_type: Some(desc.tensor_type),
            q8_0_blocks,
            q4_k_wire_bytes,
            q5_k_wire_bytes: None,
            q6_k_wire_bytes,
            tq2_0_wire_bytes: None,
            iq4_xs_wire_bytes: None,
            data,
        })
    }
}

/// Layer-streaming support: materialize a `CpuTensor` from raw GGUF tensor
/// bytes that were already read out of a layer group, without a
/// `TensorStore`. 2-D Q8_0 linears come back as RAM-resident blocks (the
/// same storage the block-backed loader produces, so the CPU forward path
/// runs unchanged); float tensors decode to f32. Note the deliberate
/// metadata asymmetry with [`TensorStore::load_cpu_f32`]: tensors built here
/// via `from_f32` carry `source_type: None`. Anything else is a loud error,
/// never a silent fallback.
pub fn cpu_tensor_from_gguf_bytes(
    name: &str,
    tensor_type: TensorType,
    dims: &[u64],
    bytes: &[u8],
) -> Result<CpuTensor> {
    let shape = TensorShape::from_gguf_dims(dims)?;
    let expected_elements = shape.element_count()?;
    match tensor_type {
        TensorType::F32 => CpuTensor::from_f32(
            name,
            shape.dims.clone(),
            decode_f32_tensor(name, bytes, expected_elements)?,
        ),
        TensorType::F16 => CpuTensor::from_f32(
            name,
            shape.dims.clone(),
            decode_f16_tensor(name, bytes, expected_elements)?,
        ),
        TensorType::BF16 => CpuTensor::from_f32(
            name,
            shape.dims.clone(),
            decode_bf16_tensor(name, bytes, expected_elements)?,
        ),
        TensorType::Q8_0 if shape.dims.len() == 2 => {
            let blocks = decode_q8_0_blocks(name, bytes, expected_elements)?;
            CpuTensor::from_q8_0_blocks(name, shape, blocks)
        }
        TensorType::Q8_0 => CpuTensor::from_f32(
            name,
            shape.dims.clone(),
            decode_q8_0_tensor(name, bytes, expected_elements)?,
        ),
        other => Err(EngineError::UnsupportedTensorType(format!(
            "tensor {name} has storage type {other:?}; ghost v1 supports F32, F16, BF16, Q8_0"
        ))),
    }
}
