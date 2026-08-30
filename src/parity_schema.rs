//! Single source of truth for the parity-dump `provenance` schema shared by the
//! writer (`src/bin/logits-dump.rs`) and the readers (`tests/parity.rs`,
//! `scripts/parity-gate.ts` mirrors it in TypeScript).
//!
//! Why versioning exists: the parity gate treats a missing provenance field as
//! a hard "stale binary" failure — the only sound default, since a dump from a
//! binary predating a field is otherwise indistinguishable from a current one.
//! But without versioning, ADDING a field retroactively invalidates every
//! cached/committed reference dump (each regeneration is ~40 min of GPU time).
//! The `schema_version` stamped into each dump resolves the ambiguity: a field
//! missing from a dump whose version PREDATES the field's introduction takes
//! the field's grandfather value (the only value any binary of that era could
//! have produced), while a field missing at/after its introduction version is
//! still the stale-binary hard failure.
//!
//! A dump with no `schema_version` at all is version 1: the baseline field set
//! written before versioning existed. Every version-1 field is REQUIRED with no
//! grandfather — the references regenerated at that era carry all of them, and
//! grandfathering any would let a genuinely stale dump pass.

/// The schema version the current `logits-dump` writes.
pub const PROVENANCE_SCHEMA_VERSION: u32 = 9;

/// One provenance field's introduction record.
pub struct ProvenanceField {
    pub name: &'static str,
    /// First schema version whose dumps carry this field.
    pub introduced: u32,
    /// Value a dump from a pre-`introduced` schema version is resolved to when
    /// the field is missing (the only value binaries of that era could produce).
    /// `None` for the version-1 baseline fields: they are required outright.
    pub grandfather: Option<&'static str>,
}

/// Field-introduction table, one entry per provenance field.
/// Version 1 (baseline, all required): moe_impl, seq_len, mm_variant,
/// mm_min_seq, no_mm_id, attn_dtype, combine, attn_mm, attn_glue.
/// Version 2: sdpa (the sdpa compute dtype; every earlier binary ran the f16
/// sdpa kernel unconditionally, hence grandfather "f16").
/// Version 3: flash (the prefill attention kernel; every earlier binary ran
/// the candle sdpa prefill — today's classic path — hence grandfather
/// "classic").
/// Version 4: act (the routed-expert SwiGLU activation kernel; every earlier
/// binary ran candle's `silu(gate) * up` chain — today's classic path — hence
/// grandfather "classic").
/// Version 5: attn_decode (the attention DECODE-projection path: "q8" for a
/// q8_0-quantized checkpoint's vendored decode gemv, "f16" for the dense f16
/// gemv, "f32-bypass" under XWEN_ATTN_F32). Grandfather "f32-bypass": like every
/// post-baseline field, the grandfather is the value the Reference oracle carries
/// (the gate enforces attn_decode on the reference side), and the oracle runs
/// under XWEN_ATTN_F32 — the whole attention block is the dequant-f32 QMatMul —
/// so every pre-v5 reference dump ran the f32-bypass decode path. Candidate dumps
/// are always freshly generated at the current version and never rely on the
/// grandfather; only the cached (committed) reference dumps do, and those are all
/// f32-bypass, so grandfathering to it keeps them valid.
/// Version 6: delta (the gated-DeltaNet layer path: "fused" for the vendored
/// conv/beta/scan/norm kernels, "classic" for the frozen reference scan under
/// XWEN_DELTA_CLASSIC). Grandfather "classic": every pre-v6 binary had only the
/// reference scan, which is also the value the oracle carries, so the cached
/// reference dumps stay valid. This field matters more than its siblings — the
/// fused delta scan is the one vendored family that is NOT bit-identical to the
/// chain it replaces, so a reference dump that quietly ran it would make the
/// bounded tiers grade the fused path against itself.
/// Version 7: dense_mm (the DENSE checkpoint's SwiGLU FFN prefill gemm: "fused"
/// for the vendored cooperative-tensor kernel, "classic" for candle's QMatMul
/// chain under XWEN_DENSE_MM_CLASSIC). Grandfather "classic": every pre-v7
/// binary had only the QMatMul chain, which is also what the oracle runs. Like
/// `delta`, this one is load-bearing rather than cosmetic — the vendored gemm is
/// NOT bit-identical to the chain it replaces (matmul2d's reduced-precision
/// path puts it ~4e-4 from the f32 oracle against QMatMul's ~1.9e-4), so a
/// reference dump that quietly ran it would grade the fused path against itself.
/// The field is env-derived, not observed: the 35B-A3B has no dense FFN layer at
/// all, so an observed field would have nothing to report on that model.
/// Version 8: mv_ext (the SMALL-BATCH token window, 2..=8: "fused" for the
/// vendored `mul_mv_ext` kernels, "classic" for the path each site had before,
/// under XWEN_MV_EXT_CLASSIC). Grandfather "classic": no pre-v8 binary had the
/// kernels at all, which is also what the oracle runs. Load-bearing like
/// `delta` and `dense_mm` — it is not bit-identical to what it replaces (a
/// different K-reduction order) — though it differs from both in DIRECTION:
/// nothing is narrowed, so it sits ~1e-6 from the f32 oracle, either better than
/// the displaced path (candle's half-staged `mul_mm`, ~1.8e-4, at the `QLinear`
/// sites) or level with it (the vendored q8_0 gemv, also ~1e-6, at the attention
/// and DeltaNet projections). Neither is bit-identical, so the pin is the same.
/// Env-derived, like `dense_mm`.
///
/// Version 9: act_l2 (the f16-tile rescale branch's fused activation glue —
/// silu*mul + per-row L2 + clamp + headroom scale in one pass, "fused" for
/// `ops::silu_mul_l2`, "classic" for the candle chain and the oracle, whose
/// ReferenceExperts never reaches the branch; XWEN_ACT_L2_CLASSIC, and
/// XWEN_ACT_CLASSIC disables it too), shexp_gemm (the shared expert's prefill
/// route: "fused" for the dense cooperative-tensor gemm via
/// QLinear::forward_gemm, "classic" for QMatMul; XWEN_SHEXP_QMATMUL or
/// XWEN_DENSE_MM_CLASSIC), and hc_gemm (the hyper-connection bottleneck's
/// prefill route: "classic"/"fused"/"down-only"/"up-only" per
/// XWEN_HC_GEMM_QMATMUL, "classic" under XWEN_DENSE_MM_CLASSIC). All three
/// grandfather "classic": no pre-v9 binary had the fold or the routes. All
/// three are load-bearing like `dense_mm` — the fold's sum order is not
/// candle's, and the gemm routes carry dense_mm's reduced-precision class.
pub const PROVENANCE_FIELDS: &[ProvenanceField] = &[
    ProvenanceField {
        name: "moe_impl",
        introduced: 1,
        grandfather: None,
    },
    ProvenanceField {
        name: "seq_len",
        introduced: 1,
        grandfather: None,
    },
    ProvenanceField {
        name: "mm_variant",
        introduced: 1,
        grandfather: None,
    },
    ProvenanceField {
        name: "mm_min_seq",
        introduced: 1,
        grandfather: None,
    },
    ProvenanceField {
        name: "no_mm_id",
        introduced: 1,
        grandfather: None,
    },
    ProvenanceField {
        name: "attn_dtype",
        introduced: 1,
        grandfather: None,
    },
    ProvenanceField {
        name: "combine",
        introduced: 1,
        grandfather: None,
    },
    ProvenanceField {
        name: "attn_mm",
        introduced: 1,
        grandfather: None,
    },
    ProvenanceField {
        name: "attn_glue",
        introduced: 1,
        grandfather: None,
    },
    ProvenanceField {
        name: "sdpa",
        introduced: 2,
        grandfather: Some("f16"),
    },
    ProvenanceField {
        name: "flash",
        introduced: 3,
        grandfather: Some("classic"),
    },
    ProvenanceField {
        name: "act",
        introduced: 4,
        grandfather: Some("classic"),
    },
    ProvenanceField {
        name: "attn_decode",
        introduced: 5,
        grandfather: Some("f32-bypass"),
    },
    ProvenanceField {
        name: "delta",
        introduced: 6,
        grandfather: Some("classic"),
    },
    ProvenanceField {
        name: "dense_mm",
        introduced: 7,
        grandfather: Some("classic"),
    },
    ProvenanceField {
        name: "mv_ext",
        introduced: 8,
        grandfather: Some("classic"),
    },
    ProvenanceField {
        name: "act_l2",
        introduced: 9,
        grandfather: Some("classic"),
    },
    ProvenanceField {
        name: "shexp_gemm",
        introduced: 9,
        grandfather: Some("classic"),
    },
    ProvenanceField {
        name: "hc_gemm",
        introduced: 9,
        grandfather: Some("classic"),
    },
];

/// Look up a field's introduction record by name.
pub fn field(name: &str) -> Option<&'static ProvenanceField> {
    PROVENANCE_FIELDS.iter().find(|f| f.name == name)
}

/// Resolve a field MISSING from a dump of the given `schema_version`:
/// `Some(grandfather)` when the dump predates the field's introduction (the
/// dump stays valid), `None` when the dump's version says the field should be
/// present (stale binary — the caller hard-fails). Unknown field names resolve
/// to `None` (fail closed).
pub fn resolve_missing(name: &str, schema_version: u32) -> Option<&'static str> {
    let f = field(name)?;
    if schema_version < f.introduced {
        f.grandfather
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Table invariants the resolution logic relies on: unique names, every
    /// introduction within the current version, versions starting at 1, and a
    /// grandfather value for every field introduced after the baseline (a
    /// post-baseline field without one could never be grandfathered, silently
    /// reintroducing the regen tax this module exists to kill).
    #[test]
    fn table_is_well_formed() {
        for (i, f) in PROVENANCE_FIELDS.iter().enumerate() {
            assert!(
                f.introduced >= 1,
                "{}: introduced {} < 1",
                f.name,
                f.introduced
            );
            assert!(
                f.introduced <= PROVENANCE_SCHEMA_VERSION,
                "{}: introduced {} > current version {PROVENANCE_SCHEMA_VERSION}",
                f.name,
                f.introduced
            );
            if f.introduced > 1 {
                assert!(
                    f.grandfather.is_some(),
                    "{}: post-baseline field needs a grandfather",
                    f.name
                );
            }
            for other in &PROVENANCE_FIELDS[..i] {
                assert_ne!(f.name, other.name, "duplicate field {}", f.name);
            }
        }
    }

    #[test]
    fn resolve_missing_grandfathers_only_predating_versions() {
        // sdpa introduced at v2: a v1 dump missing it resolves to "f16" ...
        assert_eq!(resolve_missing("sdpa", 1), Some("f16"));
        // ... while missing at/after v2 stays a hard failure.
        assert_eq!(resolve_missing("sdpa", 2), None);
        // flash introduced at v3: v1/v2 dumps missing it resolve to "classic"
        // (their binaries ran the candle sdpa prefill); missing at v3 fails.
        assert_eq!(resolve_missing("flash", 1), Some("classic"));
        assert_eq!(resolve_missing("flash", 2), Some("classic"));
        assert_eq!(resolve_missing("flash", 3), None);
        // act introduced at v4: v1..v3 dumps missing it resolve to "classic"
        // (their binaries ran candle's silu*mul chain); missing at v4 fails.
        assert_eq!(resolve_missing("act", 1), Some("classic"));
        assert_eq!(resolve_missing("act", 3), Some("classic"));
        assert_eq!(resolve_missing("act", 4), None);
        // attn_decode introduced at v5: v1..v4 dumps missing it resolve to
        // "f32-bypass" (the Reference oracle's XWEN_ATTN_F32 decode path — the
        // value every cached pre-v5 reference dump carries); missing at v5 fails.
        assert_eq!(resolve_missing("attn_decode", 1), Some("f32-bypass"));
        assert_eq!(resolve_missing("attn_decode", 4), Some("f32-bypass"));
        assert_eq!(resolve_missing("attn_decode", 5), None);
        // dense_mm introduced at v7: v1..v6 dumps missing it resolve to
        // "classic" (their binaries had only the QMatMul FFN chain, which is
        // also what the oracle runs); missing at v7 fails.
        assert_eq!(resolve_missing("dense_mm", 1), Some("classic"));
        assert_eq!(resolve_missing("dense_mm", 6), Some("classic"));
        assert_eq!(resolve_missing("dense_mm", 7), None);
        // mv_ext introduced at v8: v1..v7 dumps missing it resolve to "classic"
        // (no binary of that era had the small-batch kernels, which is also
        // what the oracle runs); missing at v8 fails.
        assert_eq!(resolve_missing("mv_ext", 1), Some("classic"));
        assert_eq!(resolve_missing("mv_ext", 7), Some("classic"));
        assert_eq!(resolve_missing("mv_ext", 8), None);
        // act_l2 / shexp_gemm / hc_gemm introduced at v9: v1..v8 dumps missing
        // them resolve to "classic" (no earlier binary had the fold or the
        // dense-gemm routes); missing at v9 fails.
        for f in ["act_l2", "shexp_gemm", "hc_gemm"] {
            assert_eq!(resolve_missing(f, 1), Some("classic"), "{f}");
            assert_eq!(resolve_missing(f, 8), Some("classic"), "{f}");
            assert_eq!(resolve_missing(f, 9), None, "{f}");
        }
        // Baseline fields are required at every version.
        assert_eq!(resolve_missing("attn_dtype", 1), None);
        assert_eq!(resolve_missing("attn_dtype", 2), None);
        // Unknown fields fail closed.
        assert_eq!(resolve_missing("not_a_field", 1), None);
    }
}
