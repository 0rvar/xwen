# Scope

One topic of [docs/decisions.md](../decisions.md), the index of decisions by topic; moved
here verbatim on 2026-09-06. Dated paragraphs, newest additions appended within their topic.


**xwen is a fork of ../laguna (crate `maxuna`) adapted to Qwen 3.6, serving exactly two
checkpoints: Qwen3.6-27B (dense) and Qwen3.6-35B-A3B (MoE).** Same design target as the
parent: maximum tok/s on this one machine (M5 Max, Metal), batch 1, GGUF weights, no
portability hedging. The 35B-A3B is the bring-up model — 20.4 GB Q4_K_M, 3B active,
fastest iteration loop; the 27B dense follows as a variant (its FFN is a strict subset
of the MoE machinery) (2026-07-28).

**The dependency set is laguna's, verbatim, and is not relitigated.** The candle git pin
(rev 21cca0b) ships the quantized indexed MoE matmuls and the residency-set APIs the
mmap loader needs; the objc2 crates stay `=`-pinned to what that rev resolves or cargo
duplicates them and the ObjC types stop interoperating (2026-07-28).

**Text-only.** Qwen 3.6 is multimodal upstream, but the GGUF conversions are text-only
(no vision tensors in the qwen35/qwen35moe arch lists; mmproj is a separate CLIP file we
do not load). The chat template's vision content items are rejected, not rendered
(2026-07-28).
