//! IQ4_NL row dequantization — the one IQ quant this port has to read.
//!
//! The PLE n-gram table ships as IQ4_NL in every Unsloth mix down to Q3 (the
//! table is random-access, so it is held at "4-bit minimum" while the trunk
//! goes lower), and the table is read one 160-float row at a time on the CPU —
//! never uploaded, never matmul'd. That is D8 class 2: a row dequantizer, no
//! Metal kernel. candle's `GgmlDType` does not know the type at all, so nothing
//! upstream can do this for us.
//!
//! Ground truth is ggml, replicated exactly:
//! `reference/llama.cpp/ggml/src/ggml-common.h` for `block_iq4_nl` (an f16
//! scale followed by 16 packed bytes, 32 elements) and `kvalues_iq4nl`, and
//! `dequantize_row_iq4_nl` in `ggml-quants.c` for the unpack order.

use half::f16;

/// Elements per IQ4_NL block (ggml `QK4_NL`).
pub const QK4_NL: usize = 32;

/// Bytes per IQ4_NL block: an f16 scale plus `QK4_NL / 2` packed nibble pairs.
/// `sizeof(block_iq4_nl)` in ggml, which static-asserts the same arithmetic.
pub const BLOCK_BYTES: usize = 2 + QK4_NL / 2;

/// The non-linear codebook, `kvalues_iq4nl` verbatim
/// (`reference/llama.cpp/ggml/src/ggml-common.h:1120`). This is the whole point
/// of the quant — the 16 levels are NOT evenly spaced, so a linear
/// reconstruction (`(q - 8) * d`, the Q4_0 shape) produces plausible garbage
/// rather than an error.
pub const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

/// Dequantizes `out.len()` elements from `src`, which must hold whole IQ4_NL
/// blocks in ggml's on-disk layout.
///
/// Nibble order is the trap: within a block the LOW nibble of byte `j` is
/// element `j` and the HIGH nibble is element `j + 16` — the two halves of a
/// block are interleaved in the file, not consecutive. Reading the pair as
/// `(2j, 2j + 1)` gives a permutation of the right 32 values, which for a
/// gathered embedding row means a shuffled vector that keeps its norm and every
/// shape: nothing downstream would fail, the model would just be worse.
pub fn dequant_row(src: &[u8], out: &mut [f32]) {
    assert!(
        out.len().is_multiple_of(QK4_NL),
        "IQ4_NL output length {} is not a whole number of {QK4_NL}-element blocks",
        out.len()
    );
    let nb = out.len() / QK4_NL;
    assert!(
        src.len() >= nb * BLOCK_BYTES,
        "IQ4_NL source has {} bytes, short of the {} needed for {nb} blocks",
        src.len(),
        nb * BLOCK_BYTES
    );

    for (block, y) in src
        .chunks_exact(BLOCK_BYTES)
        .take(nb)
        .zip(out.chunks_exact_mut(QK4_NL))
    {
        // ggml_half is little-endian f16 on every platform ggml supports; the
        // scale is not clamped or biased, so subnormals and negatives both
        // reconstruct as-is.
        let d = f32::from(f16::from_le_bytes([block[0], block[1]]));
        let qs = &block[2..];
        for j in 0..QK4_NL / 2 {
            y[j] = d * f32::from(KVALUES_IQ4NL[(qs[j] & 0xf) as usize]);
            y[j + QK4_NL / 2] = d * f32::from(KVALUES_IQ4NL[(qs[j] >> 4) as usize]);
        }
    }
}

/// Q8_0 row dequantization, the PLE table's other shipped dtype (the Q5/Q6/Q8
/// Unsloth mixes hold the table at Q8_0). Ten lines rather than a `QTensor`
/// round-trip, for the same reason IQ4_NL is here: the table is read row by row
/// on the CPU and never becomes a device tensor.
///
/// Layout is ggml's `block_q8_0`: an f16 scale then 32 signed bytes, in order —
/// no nibble interleave to get wrong.
pub const QK8_0: usize = 32;
pub const BLOCK_BYTES_Q8_0: usize = 2 + QK8_0;

pub fn dequant_row_q8_0(src: &[u8], out: &mut [f32]) {
    assert!(
        out.len().is_multiple_of(QK8_0),
        "Q8_0 output length {} is not a whole number of {QK8_0}-element blocks",
        out.len()
    );
    let nb = out.len() / QK8_0;
    assert!(
        src.len() >= nb * BLOCK_BYTES_Q8_0,
        "Q8_0 source has {} bytes, short of the {} needed for {nb} blocks",
        src.len(),
        nb * BLOCK_BYTES_Q8_0
    );

    for (block, y) in src
        .chunks_exact(BLOCK_BYTES_Q8_0)
        .take(nb)
        .zip(out.chunks_exact_mut(QK8_0))
    {
        let d = f32::from(f16::from_le_bytes([block[0], block[1]]));
        for (q, o) in block[2..].iter().zip(y.iter_mut()) {
            *o = d * f32::from(*q as i8);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Packs one IQ4_NL block from a scale and 32 codebook indices, in the
    /// file's own layout: index `j` goes in the low nibble of byte `j` and
    /// index `j + 16` in the high nibble.
    fn pack_block(d: f16, idx: &[u8; 32]) -> Vec<u8> {
        let mut b = d.to_le_bytes().to_vec();
        for j in 0..16 {
            b.push((idx[j] & 0xf) | (idx[j + 16] << 4));
        }
        b
    }

    #[test]
    fn block_layout_is_eighteen_bytes() {
        assert_eq!(BLOCK_BYTES, 18);
        assert_eq!(QK4_NL, 32);
    }

    /// A unit scale makes every output exactly a codebook entry, so this pins
    /// the table AND the nibble order at once: any other unpack order would
    /// permute the 32 values.
    #[test]
    fn unit_scale_reproduces_the_codebook() {
        let idx: [u8; 32] = std::array::from_fn(|i| if i < 16 { i as u8 } else { 31 - i as u8 });
        let bytes = pack_block(f16::ONE, &idx);
        let mut out = [0.0f32; 32];
        dequant_row(&bytes, &mut out);
        for j in 0..16 {
            assert_eq!(
                out[j],
                f32::from(KVALUES_IQ4NL[j]),
                "low nibble of byte {j}"
            );
            assert_eq!(
                out[j + 16],
                f32::from(KVALUES_IQ4NL[15 - j]),
                "high nibble of byte {j}"
            );
        }
    }

    /// The two nibbles of one byte land 16 elements apart, not next to each
    /// other. Written as an explicit inequality so a "fix" to consecutive
    /// unpacking breaks here rather than degrading the model silently.
    #[test]
    fn nibble_halves_are_interleaved_not_consecutive() {
        let mut idx = [0u8; 32];
        idx[0] = 15; // low nibble of byte 0 -> element 0
        idx[16] = 8; // high nibble of byte 0 -> element 16
        let bytes = pack_block(f16::ONE, &idx);
        let mut out = [0.0f32; 32];
        dequant_row(&bytes, &mut out);
        assert_eq!(out[0], f32::from(KVALUES_IQ4NL[15]));
        assert_eq!(out[16], f32::from(KVALUES_IQ4NL[8]));
        // element 1 is byte 1's low nibble (index 0), not byte 0's high nibble
        assert_eq!(out[1], f32::from(KVALUES_IQ4NL[0]));
    }

    /// A negative scale flips every reconstructed sign, and an f16 subnormal
    /// scale survives the f16->f32 widening. Neither is hypothetical: ggml's
    /// IQ4_NL quantizer picks the scale from the extremum of the block, so a
    /// block whose largest magnitude is negative gets `d < 0`, and a
    /// near-constant-zero row of the table gets a subnormal one.
    #[test]
    fn negative_and_subnormal_scales() {
        let idx: [u8; 32] = std::array::from_fn(|i| (i % 16) as u8);

        let neg = f16::from_f32(-2.0);
        let mut out = [0.0f32; 32];
        dequant_row(&pack_block(neg, &idx), &mut out);
        // `pack_block` places index `i` at element `i`, so the expectation is
        // positional regardless of which nibble carried it.
        for (i, v) in out.iter().enumerate() {
            assert_eq!(*v, -2.0 * f32::from(KVALUES_IQ4NL[idx[i] as usize]));
        }

        // The smallest positive f16 subnormal, 2^-24.
        let sub = f16::from_bits(0x0001);
        assert!(f32::from(sub) > 0.0 && f32::from(sub) < 1e-7);
        let mut out = [0.0f32; 32];
        dequant_row(&pack_block(sub, &idx), &mut out);
        assert_eq!(out[0], f32::from(sub) * f32::from(KVALUES_IQ4NL[0]));
        assert_ne!(out[0], 0.0, "a subnormal scale must not flush to zero");
    }

    /// Blocks stride by exactly 18 bytes and each carries its own scale.
    #[test]
    fn multiple_blocks_stride_independently() {
        let idx_a: [u8; 32] = [3; 32];
        let idx_b: [u8; 32] = [12; 32];
        let mut bytes = pack_block(f16::from_f32(0.5), &idx_a);
        bytes.extend(pack_block(f16::from_f32(4.0), &idx_b));
        assert_eq!(bytes.len(), 2 * BLOCK_BYTES);

        let mut out = [0.0f32; 64];
        dequant_row(&bytes, &mut out);
        for v in &out[..32] {
            assert_eq!(*v, 0.5 * f32::from(KVALUES_IQ4NL[3]));
        }
        for v in &out[32..] {
            assert_eq!(*v, 4.0 * f32::from(KVALUES_IQ4NL[12]));
        }
    }

    #[test]
    fn q8_0_round_trips_a_hand_built_block() {
        let mut bytes = f16::from_f32(0.25).to_le_bytes().to_vec();
        let qs: Vec<i8> = (0..32).map(|i| (i as i8) - 16).collect();
        bytes.extend(qs.iter().map(|q| *q as u8));
        let mut out = [0.0f32; 32];
        dequant_row_q8_0(&bytes, &mut out);
        for (i, v) in out.iter().enumerate() {
            assert_eq!(*v, 0.25 * f32::from(qs[i]));
        }
    }

    #[test]
    #[should_panic(expected = "not a whole number")]
    fn a_partial_block_is_rejected() {
        let mut out = [0.0f32; 8];
        dequant_row(&[0u8; BLOCK_BYTES], &mut out);
    }
}
