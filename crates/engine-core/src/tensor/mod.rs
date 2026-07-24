//! Tensor data layer: quantized block formats, the CPU tensor type with its
//! f32 compute kernels, and the GGUF-backed tensor store.
//!
//! Every kernel in this module is serial and order-preserving: each reduction
//! runs over a single f32 accumulator in ascending index order, with no
//! fused multiply-add and no reassociation, so results are bit-deterministic
//! across runs and hosts.

pub mod blocks;
pub mod store;

use crate::gguf::TensorType;
use crate::{EngineError, Result};
use blocks::{
    Q4KBlock, Q5KBlock, Q6KBlock, Q4_K_BLOCK_BYTES, Q5_K_BLOCK_BYTES, Q6_K_BLOCK_BYTES,
    QK_K_BLOCK_SIZE,
};

pub use blocks::{Q8_0Block, RuntimeDType, TensorShape};
pub use store::{cpu_tensor_from_gguf_bytes, TensorStore};

/// A CPU-resident tensor. Activations and float-sourced weights carry
/// row-major f32 `data`; quantized 2-D weights may instead carry their
/// decoded blocks (`q8_0_blocks`) or raw wire bytes (the `*_wire_bytes`
/// fields) with `data` left empty, in which case only the matching
/// block-streaming kernels can consume them — there is no f32 fallback.
#[derive(Debug, Clone, PartialEq)]
pub struct CpuTensor {
    pub name: String,
    pub shape: TensorShape,
    pub dtype: RuntimeDType,
    pub source_type: Option<TensorType>,
    /// Decoded Q8_0 blocks (f16 scale widened to f32, 32 signed byte quants
    /// per block), retained when the tensor came from Q8_0 storage. 2-D Q8_0
    /// linears store ONLY these (`data` stays empty).
    pub q8_0_blocks: Option<Vec<Q8_0Block>>,
    /// Q4_K super-block wire bytes (144 bytes per 256-value super-block,
    /// row-major), retained when `source_type` is `Q4K` so row gathers and
    /// block-dot kernels can stream them without f32 materialization.
    /// `None` for non-Q4_K tensors.
    pub q4_k_wire_bytes: Option<std::sync::Arc<Vec<u8>>>,
    /// Q5_K super-block wire bytes (176 bytes per 256-value super-block,
    /// row-major), retained when `source_type` is `Q5K`. `None` otherwise.
    pub q5_k_wire_bytes: Option<std::sync::Arc<Vec<u8>>>,
    /// Q6_K super-block wire bytes (210 bytes per 256-value super-block,
    /// row-major), retained when `source_type` is `Q6K`. `None` otherwise.
    pub q6_k_wire_bytes: Option<std::sync::Arc<Vec<u8>>>,
    /// Ternary TQ2_0 wire bytes (66 bytes per 256-value block, row-major),
    /// retained when `source_type` is `Tq2_0` so the ternary block-dot
    /// streams the quantized weights instead of materializing f32 (a fully
    /// decoded multi-billion-parameter tensor set would not fit in memory).
    pub tq2_0_wire_bytes: Option<std::sync::Arc<Vec<u8>>>,
    /// IQ4_XS wire bytes (136 bytes per 256-value super-block, row-major),
    /// retained when `source_type` is `IQ4XS` so the i-quant block-dot
    /// streams the quantized weights instead of materializing f32.
    pub iq4_xs_wire_bytes: Option<std::sync::Arc<Vec<u8>>>,
    pub data: Vec<f32>,
}

/// A Q8_0 tensor held purely as decoded blocks, with per-element and per-row
/// dequantization plus row-dot kernels that never materialize the full f32
/// payload.
#[derive(Debug, Clone, PartialEq)]
pub struct Q8_0TensorBlocks {
    pub name: String,
    pub shape: TensorShape,
    pub blocks: Vec<Q8_0Block>,
}

impl Q8_0TensorBlocks {
    pub fn element_count(&self) -> Result<usize> {
        self.shape.element_count()
    }

    pub fn byte_size_if_f32_materialized(&self) -> Result<usize> {
        self.element_count()?.checked_mul(4).ok_or_else(|| {
            EngineError::InvalidTensorData(format!(
                "tensor {} f32 materialization byte size overflow",
                self.name
            ))
        })
    }

    pub fn dequantize_elements(&self, start: usize, len: usize) -> Result<Vec<f32>> {
        const BLOCK_VALUES: usize = 32;
        let end = start.checked_add(len).ok_or_else(|| {
            EngineError::InvalidTensorData(format!(
                "tensor {} q8_0 dequant range overflows usize",
                self.name
            ))
        })?;
        let element_count = self.element_count()?;
        if end > element_count {
            return Err(EngineError::ShapeMismatch(format!(
                "tensor {} q8_0 dequant range {start}..{end} exceeds element count {element_count}",
                self.name
            )));
        }

        let mut out = Vec::with_capacity(len);
        for element_idx in start..end {
            let block_idx = element_idx / BLOCK_VALUES;
            let quant_idx = element_idx % BLOCK_VALUES;
            let block = self.blocks.get(block_idx).ok_or_else(|| {
                EngineError::InvalidTensorData(format!(
                    "tensor {} q8_0 block index {block_idx} missing for element {element_idx}",
                    self.name
                ))
            })?;
            out.push(block.scale * f32::from(block.quants[quant_idx]));
        }
        Ok(out)
    }

    pub fn dequantize_row(&self, row: usize) -> Result<Vec<f32>> {
        let (_rows, cols) = self.rank2_row_shape(row, "row dequant")?;
        self.dequantize_elements(row * cols, cols)
    }

    /// Dot one weight row against an f32 input. One sequential accumulator,
    /// columns in ascending order, per column exactly
    /// `sum += (scale * quant) * input` — dequantize first, then multiply by
    /// the input, then add. No FMA, no partial sums.
    pub fn dot_row_f32(&self, row: usize, input: &[f32]) -> Result<f32> {
        const BLOCK_VALUES: usize = 32;
        let (_rows, cols) = self.rank2_row_shape(row, "row dot")?;
        if input.len() != cols {
            return Err(EngineError::ShapeMismatch(format!(
                "tensor {} q8_0 row dot expected input width {cols}, got {}",
                self.name,
                input.len()
            )));
        }

        let row_start = row.checked_mul(cols).ok_or_else(|| {
            EngineError::InvalidTensorData(format!(
                "tensor {} q8_0 row dot offset overflows usize",
                self.name
            ))
        })?;
        let mut sum = 0.0f32;
        for (col, input_value) in input.iter().enumerate() {
            let element_idx = row_start + col;
            let block_idx = element_idx / BLOCK_VALUES;
            let quant_idx = element_idx % BLOCK_VALUES;
            let block = self.blocks.get(block_idx).ok_or_else(|| {
                EngineError::InvalidTensorData(format!(
                    "tensor {} q8_0 block index {block_idx} missing for row {row} col {col}",
                    self.name
                ))
            })?;
            sum += (block.scale * f32::from(block.quants[quant_idx])) * input_value;
        }
        Ok(sum)
    }

    /// Dot every weight row against one f32 input, producing a rank-1 tensor
    /// of row sums. The block-aligned fast path walks whole blocks; its
    /// operation sequence and accumulation order are identical to
    /// [`Self::dot_row_f32`], so both paths are bit-identical.
    pub fn dot_all_rows_f32(&self, input: &[f32], name: impl Into<String>) -> Result<CpuTensor> {
        const BLOCK_VALUES: usize = 32;
        let (rows, cols) = self.rank2_shape("all-row dot")?;
        if input.len() != cols {
            return Err(EngineError::ShapeMismatch(format!(
                "tensor {} q8_0 all-row dot expected input width {cols}, got {}",
                self.name,
                input.len()
            )));
        }

        let mut data = Vec::with_capacity(rows);
        if cols % BLOCK_VALUES == 0 {
            let blocks_per_row = cols / BLOCK_VALUES;
            let expected_blocks = rows.checked_mul(blocks_per_row).ok_or_else(|| {
                EngineError::InvalidTensorData(format!(
                    "tensor {} q8_0 all-row dot block count overflows usize",
                    self.name
                ))
            })?;
            if self.blocks.len() != expected_blocks {
                return Err(EngineError::ShapeMismatch(format!(
                    "tensor {} q8_0 all-row dot expected {expected_blocks} blocks for shape {:?}, got {}",
                    self.name,
                    self.shape.dims,
                    self.blocks.len()
                )));
            }

            for row_blocks in self.blocks.chunks_exact(blocks_per_row) {
                let mut row_sum = 0.0_f32;
                for (block, input_block) in row_blocks.iter().zip(input.chunks_exact(BLOCK_VALUES))
                {
                    for (quant, input_value) in block.quants.iter().zip(input_block) {
                        row_sum += (block.scale * f32::from(*quant)) * input_value;
                    }
                }
                data.push(row_sum);
            }
        } else {
            for row in 0..rows {
                data.push(self.dot_row_f32(row, input)?);
            }
        }

        Ok(CpuTensor {
            name: name.into(),
            shape: TensorShape { dims: vec![rows] },
            dtype: RuntimeDType::F32,
            source_type: None,
            q8_0_blocks: None,
            q4_k_wire_bytes: None,
            q5_k_wire_bytes: None,
            q6_k_wire_bytes: None,
            tq2_0_wire_bytes: None,
            iq4_xs_wire_bytes: None,
            data,
        })
    }

    pub fn dot_single_input_row_f32(
        &self,
        input: &CpuTensor,
        name: impl Into<String>,
    ) -> Result<CpuTensor> {
        if input.shape.dims.len() != 2 || input.shape.dims[0] != 1 {
            return Err(EngineError::ShapeMismatch(format!(
                "tensor {} q8_0 lazy linear expected single input row, got {:?}",
                self.name, input.shape.dims
            )));
        }
        let mut output = self.dot_all_rows_f32(&input.data, name)?;
        output.shape.dims.insert(0, 1);
        Ok(output)
    }

    fn rank2_shape(&self, op: &str) -> Result<(usize, usize)> {
        if self.shape.dims.len() != 2 {
            return Err(EngineError::ShapeMismatch(format!(
                "tensor {} q8_0 {op} requires rank-2 shape, got {:?}",
                self.name, self.shape.dims
            )));
        }
        let rows = self.shape.dims[0];
        let cols = self.shape.dims[1];
        Ok((rows, cols))
    }

    fn rank2_row_shape(&self, row: usize, op: &str) -> Result<(usize, usize)> {
        let (rows, cols) = self.rank2_shape(op)?;
        if row >= rows {
            return Err(EngineError::ShapeMismatch(format!(
                "tensor {} q8_0 row {row} out of range for {rows} rows",
                self.name
            )));
        }
        Ok((rows, cols))
    }
}

impl CpuTensor {
    /// Decompose into the owned name, dims, and f32 data buffer so a decode
    /// scratch pool can recycle all three. Only meaningful for plain-F32
    /// tensors; quantized side-storage (never present on decode
    /// intermediates) is dropped.
    pub fn into_parts(self) -> (String, Vec<usize>, Vec<f32>) {
        (self.name, self.shape.dims, self.data)
    }

    pub fn from_f32(name: impl Into<String>, dims: Vec<usize>, data: Vec<f32>) -> Result<Self> {
        let shape = TensorShape { dims };
        let expected = shape.element_count()?;
        if expected != data.len() {
            return Err(EngineError::ShapeMismatch(format!(
                "tensor data length {} does not match shape element count {expected}",
                data.len()
            )));
        }
        Ok(Self {
            name: name.into(),
            shape,
            dtype: RuntimeDType::F32,
            source_type: None,
            q8_0_blocks: None,
            q4_k_wire_bytes: None,
            q5_k_wire_bytes: None,
            q6_k_wire_bytes: None,
            tq2_0_wire_bytes: None,
            iq4_xs_wire_bytes: None,
            data,
        })
    }

    pub fn from_f32_with_source_type(
        name: impl Into<String>,
        dims: Vec<usize>,
        data: Vec<f32>,
        source_type: Option<TensorType>,
    ) -> Result<Self> {
        let mut tensor = Self::from_f32(name, dims, data)?;
        tensor.source_type = source_type;
        Ok(tensor)
    }

    pub fn from_f32_with_q8_0_blocks(
        name: impl Into<String>,
        dims: Vec<usize>,
        data: Vec<f32>,
        q8_0_blocks: Vec<Q8_0Block>,
    ) -> Result<Self> {
        let mut tensor = Self::from_f32(name, dims, data)?;
        tensor.source_type = Some(TensorType::Q8_0);
        tensor.q8_0_blocks = Some(q8_0_blocks);
        Ok(tensor)
    }

    /// A block-resident Q8_0 tensor: `data` stays EMPTY, so only the block
    /// streaming kernels (row gather, block dot) can consume it.
    pub fn from_q8_0_blocks(
        name: impl Into<String>,
        shape: TensorShape,
        q8_0_blocks: Vec<Q8_0Block>,
    ) -> Result<Self> {
        let expected_elements = shape.element_count()?;
        if !expected_elements.is_multiple_of(32) {
            return Err(EngineError::InvalidTensorData(format!(
                "q8_0 block-backed tensor element count {expected_elements} is not block aligned"
            )));
        }
        let expected_blocks = expected_elements / 32;
        if q8_0_blocks.len() != expected_blocks {
            return Err(EngineError::InvalidTensorData(format!(
                "q8_0 block-backed tensor expected {expected_blocks} blocks, got {}",
                q8_0_blocks.len()
            )));
        }
        Ok(Self {
            name: name.into(),
            shape,
            dtype: RuntimeDType::F32,
            source_type: Some(TensorType::Q8_0),
            q8_0_blocks: Some(q8_0_blocks),
            q4_k_wire_bytes: None,
            q5_k_wire_bytes: None,
            q6_k_wire_bytes: None,
            tq2_0_wire_bytes: None,
            iq4_xs_wire_bytes: None,
            data: Vec::new(),
        })
    }

    pub fn rank(&self) -> usize {
        self.shape.dims.len()
    }

    pub fn dim(&self, idx: usize) -> Result<usize> {
        self.shape.dims.get(idx).copied().ok_or_else(|| {
            EngineError::ShapeMismatch(format!(
                "tensor {} rank {} has no dimension {idx}",
                self.name,
                self.rank()
            ))
        })
    }

    /// `[m, k] x [k, n] -> [m, n]`. Serial: for each output row, ascending
    /// inner index, `out_row[col] += lhs_value * rhs_row[col]`. Inner values
    /// that compare equal to 0.0 are skipped entirely, so `0.0 * inf` /
    /// `0.0 * NaN` contributions are never added (and `-0.0` is skipped too).
    pub fn matmul(&self, rhs: &Self, name: impl Into<String>) -> Result<Self> {
        require_rank(self, 2, "matmul lhs")?;
        require_rank(rhs, 2, "matmul rhs")?;
        let m = self.dim(0)?;
        let k = self.dim(1)?;
        let rhs_k = rhs.dim(0)?;
        let n = rhs.dim(1)?;
        if k != rhs_k {
            return Err(EngineError::ShapeMismatch(format!(
                "matmul shape mismatch: lhs {:?}, rhs {:?}",
                self.shape.dims, rhs.shape.dims
            )));
        }
        let mut out = vec![0.0; m * n];

        for row in 0..m {
            let lhs_start = row * k;
            let out_start = row * n;
            let out_row = &mut out[out_start..out_start + n];
            for inner in 0..k {
                let lhs_value = self.data[lhs_start + inner];
                if lhs_value == 0.0 {
                    continue;
                }
                let rhs_start = inner * n;
                let rhs_row = &rhs.data[rhs_start..rhs_start + n];
                for col in 0..n {
                    out_row[col] += lhs_value * rhs_row[col];
                }
            }
        }

        Self::from_f32(name, vec![m, n], out)
    }

    /// `[m, k] x [n, k] -> [m, n]` with the rhs stored row-major so its rows
    /// are the output columns. Each output element is one [`dot_product`]
    /// over the shared `k` dimension; no zero-skip here (unlike
    /// [`Self::matmul`]).
    pub fn matmul_rhs_transposed(&self, rhs: &Self, name: impl Into<String>) -> Result<Self> {
        require_rank(self, 2, "matmul rhs-transposed lhs")?;
        require_rank(rhs, 2, "matmul rhs-transposed rhs")?;
        rhs.require_row_major_f32_data("matmul rhs-transposed rhs")?;
        let m = self.dim(0)?;
        let k = self.dim(1)?;
        let n = rhs.dim(0)?;
        let rhs_k = rhs.dim(1)?;
        if k != rhs_k {
            return Err(EngineError::ShapeMismatch(format!(
                "matmul rhs-transposed shape mismatch: lhs {:?}, rhs {:?}",
                self.shape.dims, rhs.shape.dims
            )));
        }
        let mut out = vec![0.0; m * n];

        for row in 0..m {
            let lhs_start = row * k;
            let lhs_row = &self.data[lhs_start..lhs_start + k];
            let out_start = row * n;
            let out_row = &mut out[out_start..out_start + n];
            for (col, out_value) in out_row.iter_mut().enumerate() {
                let rhs_start = col * k;
                let rhs_row = &rhs.data[rhs_start..rhs_start + k];
                *out_value = dot_product(lhs_row, rhs_row);
            }
        }

        Self::from_f32(name, vec![m, n], out)
    }

    fn require_row_major_f32_data(&self, context: &str) -> Result<()> {
        let expected_len = self.shape.element_count()?;
        if self.data.len() == expected_len {
            return Ok(());
        }
        let storage = if self.q8_0_blocks.is_some() {
            "retained-q8-blocks"
        } else if self.data.is_empty() {
            "no-row-major-data"
        } else {
            "invalid-row-major-f32"
        };
        Err(EngineError::InvalidTensorData(format!(
            "{context} cannot read tensor {} as row-major f32: storage={storage}, shape={:?}, data_len={}, expected_len={expected_len}",
            self.name, self.shape.dims, self.data.len()
        )))
    }

    pub fn add(&self, rhs: &Self, name: impl Into<String>) -> Result<Self> {
        let mut out = vec![0.0; self.data.len()];
        self.add_into(rhs, &mut out)?;
        Self::from_f32(name, self.shape.dims.clone(), out)
    }

    /// The exact kernel of [`Self::add`], writing into a caller-provided
    /// buffer (same length as `self.data`). Shared by the allocating path
    /// above and any scratch-pool path so both are one numeric path.
    pub fn add_into(&self, rhs: &Self, out: &mut [f32]) -> Result<()> {
        if self.shape != rhs.shape {
            return Err(EngineError::ShapeMismatch(format!(
                "shape mismatch: lhs {:?}, rhs {:?}",
                self.shape.dims, rhs.shape.dims
            )));
        }
        let len = self.data.len();
        debug_assert_eq!(out.len(), len);
        for (i, output) in out.iter_mut().enumerate().take(len) {
            *output = self.data[i] + rhs.data[i];
        }
        Ok(())
    }

    pub fn mul(&self, rhs: &Self, name: impl Into<String>) -> Result<Self> {
        if self.shape != rhs.shape {
            return Err(EngineError::ShapeMismatch(format!(
                "shape mismatch: lhs {:?}, rhs {:?}",
                self.shape.dims, rhs.shape.dims
            )));
        }
        let mut out = vec![0.0; self.data.len()];
        let len = self.data.len();
        for (i, output) in out.iter_mut().enumerate().take(len) {
            *output = self.data[i] * rhs.data[i];
        }
        Self::from_f32(name, self.shape.dims.clone(), out)
    }

    /// Per element exactly `(a / (1.0 + (-a).exp())) * b` — sigmoid by
    /// division, not reciprocal-multiply. The order of operations is
    /// load-bearing for bit parity.
    pub fn silu_mul(&self, rhs: &Self, name: impl Into<String>) -> Result<Self> {
        if self.shape != rhs.shape {
            return Err(EngineError::ShapeMismatch(format!(
                "shape mismatch: lhs {:?}, rhs {:?}",
                self.shape.dims, rhs.shape.dims
            )));
        }
        let len = self.data.len();
        let mut out = vec![0.0; len];
        for (i, o) in out.iter_mut().enumerate().take(len) {
            let a = self.data[i];
            let b = rhs.data[i];
            *o = (a / (1.0 + (-a).exp())) * b;
        }
        Self::from_f32(name, self.shape.dims.clone(), out)
    }

    /// Per element exactly `x / (1.0 + (-x).exp())`.
    pub fn silu(&self, name: impl Into<String>) -> Result<Self> {
        let len = self.data.len();
        let mut out = vec![0.0; len];
        for (i, o) in out.iter_mut().enumerate().take(len) {
            let x = self.data[i];
            *o = x / (1.0 + (-x).exp());
        }
        Self::from_f32(name, self.shape.dims.clone(), out)
    }

    pub fn rms_norm(&self, weight: &Self, eps: f32, name: impl Into<String>) -> Result<Self> {
        let mut out = vec![0.0; self.data.len()];
        self.rms_norm_into(weight, eps, &mut out)?;
        Self::from_f32(name, self.shape.dims.clone(), out)
    }

    /// The exact kernel of [`Self::rms_norm`], writing into a caller-provided
    /// buffer (same length as `self.data`). Per row: a sequential ascending
    /// sum of squares divided by `cols` once, then
    /// `scale = 1.0 / (mean_square + eps).sqrt()` (sqrt then divide, no
    /// reciprocal-sqrt), then `(input * scale) * weight` left-associated.
    pub fn rms_norm_into(&self, weight: &Self, eps: f32, out: &mut [f32]) -> Result<()> {
        require_rank(self, 2, "rms_norm input")?;
        require_rank(weight, 1, "rms_norm weight")?;
        let rows = self.dim(0)?;
        let cols = self.dim(1)?;
        if weight.dim(0)? != cols {
            return Err(EngineError::ShapeMismatch(format!(
                "rms_norm weight shape {:?} does not match input shape {:?}",
                weight.shape.dims, self.shape.dims
            )));
        }
        debug_assert_eq!(out.len(), self.data.len());

        for row in 0..rows {
            let start = row * cols;
            let end = start + cols;
            let mean_square =
                self.data[start..end].iter().map(|v| v * v).sum::<f32>() / cols as f32;
            let scale = 1.0 / (mean_square + eps).sqrt();
            for col in 0..cols {
                out[start + col] = self.data[start + col] * scale * weight.data[col];
            }
        }

        Ok(())
    }

    /// Per-head RMSNorm. Treats each row of this `[rows, cols]` tensor as
    /// `head_count` contiguous heads of `head_dim = cols / head_count` and
    /// RMS-normalizes each head independently with the shared `[head_dim]`
    /// weight.
    ///
    /// Because the data is row-major, the head slices of a `[rows, cols]`
    /// tensor are exactly the rows of a `[rows*head_count, head_dim]` tensor,
    /// so this reuses [`Self::rms_norm`] verbatim — one numeric path for
    /// every RMS norm in the engine.
    pub fn per_head_rms_norm(
        &self,
        weight: &Self,
        head_count: usize,
        eps: f32,
        name: impl Into<String>,
    ) -> Result<Self> {
        require_rank(self, 2, "per_head_rms_norm input")?;
        let rows = self.dim(0)?;
        let cols = self.dim(1)?;
        if head_count == 0 || !cols.is_multiple_of(head_count) {
            return Err(EngineError::ShapeMismatch(format!(
                "per_head_rms_norm width {cols} is not divisible by head count {head_count}"
            )));
        }
        let head_dim = cols / head_count;
        let name = name.into();
        let per_head = Self::from_f32(
            name.clone(),
            vec![rows * head_count, head_dim],
            self.data.clone(),
        )?;
        let normed = per_head.rms_norm(weight, eps, name.clone())?;
        Self::from_f32(name, vec![rows, cols], normed.data)
    }

    /// Row-wise softmax over the last dimension. Per row: ascending max fold
    /// from `NEG_INFINITY` (`f32::max` drops NaN operands), one ascending
    /// pass storing `exp(v - max)` in place while summing the stored values,
    /// a guard on a zero or non-finite sum, then per-element DIVISION by the
    /// sum (not multiply-by-reciprocal).
    pub fn softmax_last_dim(&self, name: impl Into<String>) -> Result<Self> {
        if self.shape.dims.is_empty() {
            return Err(EngineError::ShapeMismatch(
                "softmax requires at least one dimension".to_string(),
            ));
        }
        let cols = *self.shape.dims.last().expect("non-empty dims");
        if cols == 0 || !self.data.len().is_multiple_of(cols) {
            return Err(EngineError::ShapeMismatch(format!(
                "softmax invalid shape {:?} for data length {}",
                self.shape.dims,
                self.data.len()
            )));
        }
        let mut out = self.data.clone();
        for row in out.chunks_exact_mut(cols) {
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0;
            for v in row.iter_mut() {
                *v = (*v - max).exp();
                sum += *v;
            }
            if sum == 0.0 || !sum.is_finite() {
                return Err(EngineError::ShapeMismatch(
                    "softmax produced invalid normalization sum".to_string(),
                ));
            }
            for v in row.iter_mut() {
                *v /= sum;
            }
        }
        Self::from_f32(name, self.shape.dims.clone(), out)
    }

    /// Gather embedding rows for `token_ids`. Dispatch order is load-bearing:
    /// decoded Q8_0 blocks first, then the Q4_K/Q5_K/Q6_K wire-byte arms,
    /// then the plain row-major f32 gather. A wire-only tensor of any other
    /// type (TQ2_0 / IQ4_XS) reaches the plain path with empty `data` and
    /// panics on the row slice — those types are consumed by their block-dot
    /// kernels, never by embedding lookup.
    pub fn embedding_lookup(&self, token_ids: &[u32], name: impl Into<String>) -> Result<Self> {
        require_rank(self, 2, "embedding weight")?;
        let vocab = self.dim(0)?;
        let width = self.dim(1)?;
        if let Some(blocks) = self.q8_0_blocks.as_deref() {
            return self.embedding_lookup_q8_0_block_backed(token_ids, name, vocab, width, blocks);
        }
        // K-quant token-embedding: the wire-only loader leaves `data` empty, so gather
        // each requested row by dequantizing its super-blocks straight from wire bytes.
        if let Some(wire) = self.q4_k_wire_bytes.as_deref() {
            return self.embedding_lookup_kquant_wire(
                token_ids,
                name,
                vocab,
                width,
                wire,
                Q4_K_BLOCK_BYTES,
                |b, out| {
                    let blk: &[u8; Q4_K_BLOCK_BYTES] = b.try_into().unwrap();
                    Q4KBlock::from_bytes(blk).dequantize(out);
                },
            );
        }
        if let Some(wire) = self.q5_k_wire_bytes.as_deref() {
            return self.embedding_lookup_kquant_wire(
                token_ids,
                name,
                vocab,
                width,
                wire,
                Q5_K_BLOCK_BYTES,
                |b, out| {
                    let blk: &[u8; Q5_K_BLOCK_BYTES] = b.try_into().unwrap();
                    Q5KBlock::from_bytes(blk).dequantize(out);
                },
            );
        }
        if let Some(wire) = self.q6_k_wire_bytes.as_deref() {
            return self.embedding_lookup_kquant_wire(
                token_ids,
                name,
                vocab,
                width,
                wire,
                Q6_K_BLOCK_BYTES,
                |b, out| {
                    let blk: &[u8; Q6_K_BLOCK_BYTES] = b.try_into().unwrap();
                    Q6KBlock::from_bytes(blk).dequantize(out);
                },
            );
        }
        let output_len = token_ids.len().checked_mul(width).ok_or_else(|| {
            EngineError::ShapeMismatch("embedding lookup output element count overflow".to_string())
        })?;
        let mut out = Vec::with_capacity(output_len);
        for token_id in token_ids {
            let token_idx = usize::try_from(*token_id).map_err(|_| {
                EngineError::ShapeMismatch(format!("token id {token_id} does not fit usize"))
            })?;
            if token_idx >= vocab {
                return Err(EngineError::ShapeMismatch(format!(
                    "token id {token_id} out of range for vocab size {vocab}"
                )));
            }
            let start = token_idx.checked_mul(width).ok_or_else(|| {
                EngineError::ShapeMismatch("embedding lookup row start overflow".to_string())
            })?;
            let end = start.checked_add(width).ok_or_else(|| {
                EngineError::ShapeMismatch("embedding lookup row end overflow".to_string())
            })?;
            out.extend_from_slice(&self.data[start..end]);
        }
        Self::from_f32(name, vec![token_ids.len(), width], out)
    }

    fn embedding_lookup_q8_0_block_backed(
        &self,
        token_ids: &[u32],
        name: impl Into<String>,
        vocab: usize,
        width: usize,
        blocks: &[Q8_0Block],
    ) -> Result<Self> {
        const Q8_0_BLOCK_VALUES: usize = 32;
        if self.source_type != Some(TensorType::Q8_0) {
            return Err(EngineError::ShapeMismatch(format!(
                "block-backed embedding {} must come from Q8_0 storage",
                self.name
            )));
        }
        if !width.is_multiple_of(Q8_0_BLOCK_VALUES) {
            return Err(EngineError::ShapeMismatch(format!(
                "block-backed q8_0 embedding width {width} is not divisible by {Q8_0_BLOCK_VALUES}"
            )));
        }
        let blocks_per_row = width / Q8_0_BLOCK_VALUES;
        let expected_blocks = vocab.checked_mul(blocks_per_row).ok_or_else(|| {
            EngineError::ShapeMismatch(
                "block-backed q8_0 embedding block count overflow".to_string(),
            )
        })?;
        if blocks.len() != expected_blocks {
            return Err(EngineError::ShapeMismatch(format!(
                "block-backed q8_0 embedding block count {} does not match expected {expected_blocks}",
                blocks.len()
            )));
        }
        let output_len = token_ids.len().checked_mul(width).ok_or_else(|| {
            EngineError::ShapeMismatch(
                "block-backed q8_0 embedding output element count overflow".to_string(),
            )
        })?;
        let mut out = Vec::with_capacity(output_len);
        for token_id in token_ids {
            let token_idx = usize::try_from(*token_id).map_err(|_| {
                EngineError::ShapeMismatch(format!("token id {token_id} does not fit usize"))
            })?;
            if token_idx >= vocab {
                return Err(EngineError::ShapeMismatch(format!(
                    "token id {token_id} out of range for vocab size {vocab}"
                )));
            }
            let block_start = token_idx.checked_mul(blocks_per_row).ok_or_else(|| {
                EngineError::ShapeMismatch(
                    "block-backed q8_0 embedding row start overflow".to_string(),
                )
            })?;
            for block in &blocks[block_start..block_start + blocks_per_row] {
                out.extend(
                    block
                        .quants
                        .iter()
                        .map(|quant| block.scale * f32::from(*quant)),
                );
            }
        }
        Self::from_f32(name, vec![token_ids.len(), width], out)
    }

    /// Gather K-quant (Q4_K / Q5_K / Q6_K) embedding rows straight from the
    /// super-block wire bytes (the wire-only loader leaves `data` empty).
    /// Only the requested rows are dequantized via `dequant_block`.
    /// `block_bytes` is the wire super-block size (144 for Q4_K, 176 for
    /// Q5_K, 210 for Q6_K); each super-block holds 256 values.
    #[allow(clippy::too_many_arguments)]
    fn embedding_lookup_kquant_wire(
        &self,
        token_ids: &[u32],
        name: impl Into<String>,
        vocab: usize,
        width: usize,
        wire: &[u8],
        block_bytes: usize,
        dequant_block: impl Fn(&[u8], &mut [f32; QK_K_BLOCK_SIZE]),
    ) -> Result<Self> {
        if !width.is_multiple_of(QK_K_BLOCK_SIZE) {
            return Err(EngineError::ShapeMismatch(format!(
                "K-quant embedding width {width} is not divisible by {QK_K_BLOCK_SIZE}"
            )));
        }
        let blocks_per_row = width / QK_K_BLOCK_SIZE;
        let row_bytes = blocks_per_row * block_bytes;
        let expected = vocab.checked_mul(row_bytes).ok_or_else(|| {
            EngineError::ShapeMismatch("K-quant embedding byte count overflow".to_string())
        })?;
        if wire.len() != expected {
            return Err(EngineError::ShapeMismatch(format!(
                "K-quant embedding wire bytes {} do not match expected {expected}",
                wire.len()
            )));
        }
        let mut out = Vec::with_capacity(token_ids.len() * width);
        let mut values = [0.0_f32; QK_K_BLOCK_SIZE];
        for token_id in token_ids {
            let token_idx = usize::try_from(*token_id).map_err(|_| {
                EngineError::ShapeMismatch(format!("token id {token_id} does not fit usize"))
            })?;
            if token_idx >= vocab {
                return Err(EngineError::ShapeMismatch(format!(
                    "token id {token_id} out of range for vocab size {vocab}"
                )));
            }
            let row = &wire[token_idx * row_bytes..(token_idx + 1) * row_bytes];
            for b in 0..blocks_per_row {
                dequant_block(&row[b * block_bytes..(b + 1) * block_bytes], &mut values);
                out.extend_from_slice(&values);
            }
        }
        Self::from_f32(name, vec![token_ids.len(), width], out)
    }

    pub fn transpose_2d(&self, name: impl Into<String>) -> Result<Self> {
        require_rank(self, 2, "transpose")?;
        let rows = self.dim(0)?;
        let cols = self.dim(1)?;
        let mut out = vec![0.0; self.data.len()];
        for row in 0..rows {
            for col in 0..cols {
                out[col * rows + row] = self.data[row * cols + col];
            }
        }
        Self::from_f32(name, vec![cols, rows], out)
    }
}

fn require_rank(tensor: &CpuTensor, rank: usize, op: &str) -> Result<()> {
    if tensor.rank() != rank {
        return Err(EngineError::ShapeMismatch(format!(
            "{op} expected rank {rank}, got shape {:?}",
            tensor.shape.dims
        )));
    }
    Ok(())
}

/// f32 dot product with ONE accumulator advancing in strict ascending index
/// order. The manual 4-wide unroll issues the adds sequentially, so the
/// result is bit-identical to a plain one-element loop; any multi-accumulator
/// or SIMD rewrite would change the rounding.
pub fn dot_product(lhs: &[f32], rhs: &[f32]) -> f32 {
    debug_assert_eq!(lhs.len(), rhs.len());
    let mut sum = 0.0;
    let mut idx = 0;
    while idx + 4 <= lhs.len() {
        sum += lhs[idx] * rhs[idx];
        sum += lhs[idx + 1] * rhs[idx + 1];
        sum += lhs[idx + 2] * rhs[idx + 2];
        sum += lhs[idx + 3] * rhs[idx + 3];
        idx += 4;
    }
    while idx < lhs.len() {
        sum += lhs[idx] * rhs[idx];
        idx += 1;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q8_block_backed_embedding_dequantizes_selected_rows() {
        let row0 = Q8_0Block {
            scale: 0.5,
            quants: [2; 32],
        };
        let row1 = Q8_0Block {
            scale: 0.25,
            quants: [-4; 32],
        };
        let tensor = CpuTensor::from_q8_0_blocks(
            "token_embd.weight",
            TensorShape { dims: vec![2, 32] },
            vec![row0, row1],
        )
        .unwrap();

        let embedding = tensor.embedding_lookup(&[1, 0], "embedding").unwrap();

        assert_eq!(embedding.shape.dims, vec![2, 32]);
        assert_eq!(&embedding.data[..32], &[-1.0; 32]);
        assert_eq!(&embedding.data[32..], &[1.0; 32]);
    }

    #[test]
    fn matmul_rhs_transposed_handles_single_row_vectors() {
        let lhs = CpuTensor::from_f32("lhs", vec![1, 5], vec![1.0, -2.0, 3.0, 0.5, 4.0]).unwrap();
        let rhs = CpuTensor::from_f32(
            "rhs_t",
            vec![3, 5],
            vec![
                2.0, 0.0, -1.0, 4.0, 0.5, // first output row
                -3.0, 1.0, 0.0, 2.0, -0.5, // second output row
                1.0, 1.0, 1.0, 1.0, 1.0, // third output row
            ],
        )
        .unwrap();

        let actual = lhs.matmul_rhs_transposed(&rhs, "out").unwrap();

        assert_eq!(actual.shape.dims, vec![1, 3]);
        assert_eq!(actual.data, vec![3.0, -6.0, 6.5]);
    }

    #[test]
    fn matmul_rhs_transposed_handles_rectangular_batches() {
        let lhs = CpuTensor::from_f32(
            "lhs",
            vec![2, 3],
            vec![
                1.0, 2.0, 3.0, // row 0
                4.0, 5.0, 6.0, // row 1
            ],
        )
        .unwrap();
        let rhs = CpuTensor::from_f32(
            "rhs_t",
            vec![2, 3],
            vec![
                7.0, 8.0, 9.0, // output 0
                1.0, 0.0, -1.0, // output 1
            ],
        )
        .unwrap();

        let actual = lhs.matmul_rhs_transposed(&rhs, "out").unwrap();

        assert_eq!(actual.shape.dims, vec![2, 2]);
        assert_eq!(actual.data, vec![50.0, -2.0, 122.0, -2.0]);
    }

    #[test]
    fn matmul_wide_output_matches_reference() {
        let lhs_values = vec![1.0, -2.0, 0.5, 3.0, -0.25];
        let output_width = 1031;
        let rhs_values = (0..lhs_values.len() * output_width)
            .map(|idx| ((idx % 37) as f32 - 18.0) * 0.01)
            .collect::<Vec<_>>();
        let lhs =
            CpuTensor::from_f32("lhs", vec![1, lhs_values.len()], lhs_values.clone()).unwrap();
        let rhs = CpuTensor::from_f32(
            "rhs",
            vec![lhs_values.len(), output_width],
            rhs_values.clone(),
        )
        .unwrap();

        let actual = lhs.matmul(&rhs, "out").unwrap();

        let expected = (0..output_width)
            .map(|col| {
                lhs_values
                    .iter()
                    .enumerate()
                    .map(|(inner, lhs_value)| lhs_value * rhs_values[inner * output_width + col])
                    .sum::<f32>()
            })
            .collect::<Vec<_>>();
        assert_eq!(actual.shape.dims, vec![1, output_width]);
        for (idx, &actual_val) in actual.data.iter().enumerate() {
            let expected_val = expected[idx];
            assert!(
                (actual_val - expected_val).abs() < 1e-4,
                "mismatch at index {idx}: actual {actual_val}, expected {expected_val}"
            );
        }
    }

    #[test]
    fn matmul_rhs_transposed_wide_output_matches_reference() {
        let lhs_values = vec![1.0, -2.0, 0.5, 3.0, -0.25];
        let output_width = 1031;
        let rhs_values = (0..output_width * lhs_values.len())
            .map(|idx| ((idx % 41) as f32 - 20.0) * 0.01)
            .collect::<Vec<_>>();
        let lhs =
            CpuTensor::from_f32("lhs", vec![1, lhs_values.len()], lhs_values.clone()).unwrap();
        let rhs = CpuTensor::from_f32(
            "rhs_t",
            vec![output_width, lhs_values.len()],
            rhs_values.clone(),
        )
        .unwrap();

        let actual = lhs.matmul_rhs_transposed(&rhs, "out").unwrap();

        let expected = (0..output_width)
            .map(|row| {
                let row_start = row * lhs_values.len();
                lhs_values
                    .iter()
                    .zip(&rhs_values[row_start..row_start + lhs_values.len()])
                    .map(|(left, right)| left * right)
                    .sum::<f32>()
            })
            .collect::<Vec<_>>();
        assert_eq!(actual.shape.dims, vec![1, output_width]);
        for (idx, &actual_val) in actual.data.iter().enumerate() {
            let expected_val = expected[idx];
            assert!(
                (actual_val - expected_val).abs() < 1e-4,
                "mismatch at index {idx}: actual {actual_val}, expected {expected_val}"
            );
        }
    }
}
