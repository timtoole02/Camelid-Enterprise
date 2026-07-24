use engine_core::tensor::blocks::{
    decode_nvfp4_tensor, nvfp4_find_nan_scale, nvfp4_wire_block_dequant, KVALUES_MXFP4_I8,
    NVFP4_VALUES_PER_BLOCK, NVFP4_WIRE_BYTES_PER_BLOCK, UE4M3_TO_F32,
};
use serde::Deserialize;
use std::path::{Path, PathBuf};
use serde_json::Value;


fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("dequant")
}

fn load_fixture(name: &str) -> Value {
    let path = fixtures_dir().join(name);
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("missing fixture {}: {e}", path.display()));
    let v: Value = serde_json::from_str(&raw).unwrap_or_else(|e| panic!("{name} parses: {e}"));
    v
}

fn hex_u32(h: &str) -> u32 {
    u32::from_str_radix(h, 16).unwrap_or_else(|e| panic!("bad hex u32 {h:?}: {e}"))
}

/// Concatenated `%08x` hex row (element order 0..63) -> u32 bit patterns.
fn hex_row_bits(h: &str) -> Vec<u32> {
    assert!(h.len().is_multiple_of(8), "hex row length {}", h.len());
    (0..h.len())
        .step_by(8)
        .map(|i| hex_u32(&h[i..i + 8]))
        .collect()
}

/// Minimal RFC 4648 base64 decoder (fixtures only; the crate deliberately takes no
/// base64 dependency).
fn b64_decode(s: &str) -> Vec<u8> {
    let mut table = [255u8; 256];
    for (i, c) in (b'A'..=b'Z').enumerate() {
        table[c as usize] = i as u8;
    }
    for (i, c) in (b'a'..=b'z').enumerate() {
        table[c as usize] = 26 + i as u8;
    }
    for (i, c) in (b'0'..=b'9').enumerate() {
        table[c as usize] = 52 + i as u8;
    }
    table[b'+' as usize] = 62;
    table[b'/' as usize] = 63;
    let bytes: Vec<u8> = s.bytes().filter(|&b| b != b'=').collect();
    let mut out = Vec::with_capacity(bytes.len() * 3 / 4);
    for chunk in bytes.chunks(4) {
        let mut acc = 0u32;
        for (k, &b) in chunk.iter().enumerate() {
            let v = table[b as usize];
            assert_ne!(v, 255, "bad base64 byte {b}");
            acc |= u32::from(v) << (18 - 6 * k);
        }
        out.push((acc >> 16) as u8);
        if chunk.len() > 2 {
            out.push((acc >> 8) as u8);
        }
        if chunk.len() > 3 {
            out.push(acc as u8);
        }
    }
    out
}

fn assert_block_bits(got: &[f32; 64], want_bits: &[u32], ctx: &str) {
    assert_eq!(want_bits.len(), 64, "{ctx}: expected 64 reference values");
    for (j, (g, w)) in got.iter().zip(want_bits.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            *w,
            "{ctx}: element {j} got {:#010x} want {:#010x}",
            g.to_bits(),
            w
        );
    }
}

// ---- fixture 1: 256-entry UE4M3 table ---------------------------------------------------

#[test]
fn ue4m3_table_bit_exact_vs_pin() {
    let fx = load_fixture("nvfp4_ue4m3_table.json");
    let table = fx["table"].as_array().expect("table array");
    assert_eq!(table.len(), 256);
    for (b, entry) in table.iter().enumerate() {
        let want = hex_u32(entry.as_str().expect("hex entry"));
        assert_eq!(
            UE4M3_TO_F32[b].to_bits(),
            want,
            "UE4M3 byte {b:#04x}: got {:#010x} want {want:#010x}",
            UE4M3_TO_F32[b].to_bits()
        );
    }
    // Spot-lock the sentinel semantics the fixture encodes: raw 0x00/0x7F flush to
    // 0.0, raw 0xFF does NOT (the reference CPU check is on the raw byte) and
    // decodes to 240.0; bit 7 is otherwise stripped by the exp/man extraction.
    assert_eq!(UE4M3_TO_F32[0x00].to_bits(), 0);
    assert_eq!(UE4M3_TO_F32[0x7F].to_bits(), 0);
    assert_eq!(UE4M3_TO_F32[0x80].to_bits(), 0); // masked 0x00: subnormal man=0
    assert_eq!(UE4M3_TO_F32[0xFF].to_bits(), 240.0_f32.to_bits());
    assert_eq!(UE4M3_TO_F32[0x38].to_bits(), 0.5_f32.to_bits());
    assert_eq!(UE4M3_TO_F32[0x7E].to_bits(), 224.0_f32.to_bits());
}

// ---- fixture 2: 256 x 16 decode table + nibble-position probes --------------------------

#[test]
fn decode_table_all_4096_scale_code_pairs_bit_exact() {
    let fx = load_fixture("nvfp4_decode_table.json");
    let kvalues: Vec<i64> = fx["kvalues"]
        .as_array()
        .expect("kvalues")
        .iter()
        .map(|v| v.as_i64().expect("kvalue int"))
        .collect();
    assert_eq!(
        kvalues,
        KVALUES_MXFP4_I8
            .iter()
            .map(|&v| i64::from(v))
            .collect::<Vec<_>>(),
        "fixture kvalues disagree with KVALUES_MXFP4_I8"
    );
    let entries = fx["entries"].as_array().expect("entries");
    assert_eq!(entries.len(), 256);
    for (scale, row) in entries.iter().enumerate() {
        let row = row.as_array().expect("row");
        assert_eq!(row.len(), 16);
        for (code, entry) in row.iter().enumerate() {
            let want = hex_u32(entry.as_str().expect("hex entry"));
            // Same crafted block the generator drove through the reference:
            // d[0..3] = scale, every qs nibble = code.
            let mut block = [0u8; NVFP4_WIRE_BYTES_PER_BLOCK];
            block[..4].fill(scale as u8);
            block[4..].fill((code | (code << 4)) as u8);
            let got = nvfp4_wire_block_dequant(&block);
            for (j, g) in got.iter().enumerate() {
                assert_eq!(
                    g.to_bits(),
                    want,
                    "scale {scale:#04x} code {code}: element {j} got {:#010x} want {want:#010x}",
                    g.to_bits()
                );
            }
        }
    }
}

#[test]
fn decode_table_nibble_probes_lock_packing_order() {
    let fx = load_fixture("nvfp4_decode_table.json");
    let probes = fx["nibble_probes"].as_array().expect("nibble_probes");
    assert_eq!(probes.len(), 4);
    for probe in probes {
        let name = probe["name"].as_str().expect("probe name");
        let wire = b64_decode(probe["wire"].as_str().expect("wire"));
        assert_eq!(
            wire.len(),
            NVFP4_WIRE_BYTES_PER_BLOCK,
            "{name}: wire length"
        );
        let want = hex_row_bits(probe["expected"].as_str().expect("expected"));
        let got = nvfp4_wire_block_dequant(&wire);
        assert_block_bits(&got, &want, name);
    }
}

// ---- fixtures 3+4: reference-quantized random blocks and real GGUF blocks ---------------

#[derive(Deserialize)]
struct WireBlock {
    #[serde(default)]
    tag: String,
    #[serde(default)]
    t: usize,
    w: String,
    e: String,
}

fn check_blocks_both_paths(
    blocks: &[WireBlock],
    suite: &str,
    group_label: impl Fn(&WireBlock) -> String,
) {
    // Path 1: the bit-exact per-block seam.
    let mut all_wire = Vec::with_capacity(blocks.len() * NVFP4_WIRE_BYTES_PER_BLOCK);
    let mut all_bits = Vec::with_capacity(blocks.len() * NVFP4_VALUES_PER_BLOCK);
    for (i, blk) in blocks.iter().enumerate() {
        let wire = b64_decode(&blk.w);
        assert_eq!(
            wire.len(),
            NVFP4_WIRE_BYTES_PER_BLOCK,
            "{suite} block {i} ({}): wire length",
            group_label(blk)
        );
        let want = hex_row_bits(&blk.e);
        let got = nvfp4_wire_block_dequant(&wire);
        assert_block_bits(
            &got,
            &want,
            &format!("{suite} block {i} ({})", group_label(blk)),
        );
        all_wire.extend_from_slice(&wire);
        all_bits.extend_from_slice(&want);
    }
    // Path 2: the fail-closed load seam over the concatenated blocks (these
    // fixtures are quantizer-produced, so they carry no sentinel scale bytes and
    // must decode identically).
    let n_elements = blocks.len() * NVFP4_VALUES_PER_BLOCK;
    let out = decode_nvfp4_tensor(&format!("fixture:{suite}"), &all_wire, n_elements)
        .unwrap_or_else(|e| panic!("{suite}: decode_nvfp4_tensor failed: {e}"));
    assert_eq!(out.len(), n_elements);
    for (j, (g, w)) in out.iter().zip(all_bits.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            *w,
            "{suite}: concatenated element {j} got {:#010x} want {w:#010x}",
            g.to_bits()
        );
    }
}

#[test]
fn random_blocks_bit_exact_through_both_paths() {
    let fx = load_fixture("nvfp4_random_blocks.json");
    let blocks: Vec<WireBlock> =
        serde_json::from_value(fx["blocks"].clone()).expect("blocks parse");
    assert_eq!(blocks.len(), 10031, "10000 PRNG + 31 edge blocks");
    check_blocks_both_paths(&blocks, "random_blocks", |b| b.tag.clone());
}

#[test]
fn real_gguf_blocks_bit_exact_through_both_paths() {
    let fx = load_fixture("nvfp4_real_blocks.json");
    let blocks: Vec<WireBlock> =
        serde_json::from_value(fx["blocks"].clone()).expect("blocks parse");
    assert_eq!(blocks.len(), 2048);
    let tensor_names: Vec<String> = fx["tensors"]
        .as_array()
        .expect("tensors")
        .iter()
        .map(|t| t["name"].as_str().expect("tensor name").to_string())
        .collect();
    assert_eq!(tensor_names.len(), 6);
    // Group per source tensor so decode_nvfp4_tensor sees one multi-block tensor
    // per real GGUF tensor (sampled blocks concatenated in fixture order).
    for (t, name) in tensor_names.iter().enumerate() {
        let group: Vec<WireBlock> = blocks
            .iter()
            .filter(|b| b.t == t)
            .map(|b| WireBlock {
                tag: String::new(),
                t: b.t,
                w: b.w.clone(),
                e: b.e.clone(),
            })
            .collect();
        assert!(!group.is_empty(), "tensor {t} ({name}) has sampled blocks");
        check_blocks_both_paths(&group, &format!("real_blocks:{name}"), |b| {
            format!("t={}", b.t)
        });
    }
}

// ---- fail-closed seam --------------------------------------------------------------------

/// A benign block: d = 0x38 (0.5) everywhere, every element code 3 (decodes 1.5).
fn clean_block() -> [u8; NVFP4_WIRE_BYTES_PER_BLOCK] {
    let mut block = [0x33u8; NVFP4_WIRE_BYTES_PER_BLOCK];
    block[..4].fill(0x38);
    block
}

#[test]
fn decode_nvfp4_tensor_refuses_nan_sentinel_scales_but_block_dequant_stays_pin_bitwise() {
    for sentinel in [0x7Fu8, 0xFFu8] {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&clean_block());
        bytes.extend_from_slice(&clean_block());
        let mut poisoned = clean_block();
        poisoned[2] = sentinel; // d[2] of block 2
        bytes.extend_from_slice(&poisoned);
        bytes.extend_from_slice(&clean_block());

        let err = decode_nvfp4_tensor("blk.7.ffn_up.weight", &bytes, 4 * NVFP4_VALUES_PER_BLOCK)
            .expect_err("sentinel scale must refuse");
        let msg = err.to_string();
        assert!(
            msg.contains("blk.7.ffn_up.weight"),
            "error must carry the tensor name: {msg}"
        );
        assert!(
            msg.contains("block 2"),
            "error must carry the first offending block index: {msg}"
        );

        // The scanner seam agrees and reports the FIRST offending block.
        assert_eq!(
            nvfp4_find_nan_scale(&bytes),
            Some(2),
            "sentinel {sentinel:#04x}"
        );
    }

    // Deliberately different seam: the bit-exact per-block dequant does NOT
    // error on the very same bytes. Raw 0x7F flushes to d = 0.0 (positive codes
    // decode to +0.0, negative codes to -0.0); raw 0xFF is NOT flushed by the
    // reference CPU path and decodes to d = 240.0 (bits below are literals from
    // the golden decode-table fixture, row 0xFF).
    let mut flushed = [0x91u8; NVFP4_WIRE_BYTES_PER_BLOCK]; // low nibble code 1, high code 9
    flushed[..4].fill(0x7F);
    let got = nvfp4_wire_block_dequant(&flushed);
    for s in 0..4 {
        for j in 0..8 {
            assert_eq!(got[s * 16 + j].to_bits(), 0x0000_0000, "0x7F: +0.0 flush");
            assert_eq!(
                got[s * 16 + 8 + j].to_bits(),
                0x8000_0000,
                "0x7F: -0.0 flush"
            );
        }
    }
    let mut not_flushed = [0x91u8; NVFP4_WIRE_BYTES_PER_BLOCK];
    not_flushed[..4].fill(0xFF);
    let got = nvfp4_wire_block_dequant(&not_flushed);
    for s in 0..4 {
        for j in 0..8 {
            assert_eq!(
                got[s * 16 + j].to_bits(),
                0x4370_0000,
                "0xFF code 1 -> 240.0"
            );
            assert_eq!(
                got[s * 16 + 8 + j].to_bits(),
                0xC370_0000,
                "0xFF code 9 -> -240.0"
            );
        }
    }
}

#[test]
fn nan_scale_scanner_ignores_qs_bytes_and_reports_first_block() {
    // Sentinel byte values inside qs (element codes) must NOT trip the scanner.
    let mut qs_lookalike = [0xFFu8; NVFP4_WIRE_BYTES_PER_BLOCK];
    qs_lookalike[..4].fill(0x38);
    assert_eq!(nvfp4_find_nan_scale(&qs_lookalike), None);

    let mut bytes = Vec::new();
    let mut first = clean_block();
    first[0] = 0x7F;
    bytes.extend_from_slice(&first);
    let mut second = clean_block();
    second[3] = 0xFF;
    bytes.extend_from_slice(&second);
    assert_eq!(nvfp4_find_nan_scale(&bytes), Some(0), "first offender wins");

    assert_eq!(nvfp4_find_nan_scale(&clean_block()), None);
}

#[test]
fn decode_nvfp4_tensor_refuses_wrong_byte_length() {
    let block = clean_block();
    let err = decode_nvfp4_tensor("t.short", &block[..35], NVFP4_VALUES_PER_BLOCK)
        .expect_err("short wire must refuse");
    let msg = err.to_string();
    assert!(
        msg.contains("t.short") && msg.contains("NVFP4 wire length"),
        "{msg}"
    );

    let mut long = block.to_vec();
    long.push(0);
    let err = decode_nvfp4_tensor("t.long", &long, NVFP4_VALUES_PER_BLOCK)
        .expect_err("long wire must refuse");
    assert!(err.to_string().contains("NVFP4 wire length"), "{err}");
}

#[test]
fn decode_nvfp4_tensor_refuses_non_multiple_of_64_elements() {
    let block = clean_block();
    let err = decode_nvfp4_tensor("t.ragged", &block, 63).expect_err("63 elements must refuse");
    let msg = err.to_string();
    assert!(
        msg.contains("t.ragged") && msg.contains("not a multiple"),
        "{msg}"
    );
}
