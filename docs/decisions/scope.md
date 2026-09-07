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

**Dense Qwen3-4B is a full checkpoint AND the conditioning encoder building block, and
it is not a tok/s target.** The design-target line in AGENTS.md names the three Qwen 3.6
and 3.8 GGUF checkpoints plus Flash-Next; `model_type: qwen3` joins the repo for two
reasons at once and neither of them is throughput. The user wants full inference on the
dense 4B (generate, chat, serve, batch), and the diffusion image transformers to be
implemented later need text conditioning, which for Z-Image-Turbo is exactly this model
called in-process for one hidden state. So the amendment is a scope amendment, not a
target amendment: the 4B is held to correctness bars and to not costing the shipped
checkpoints anything, and no arc of it is allowed to argue for a change to the hot path
on its behalf. The weight format follows from the role rather than from taste - the
encoder that a pipeline hands to a diffusion model is the HF BF16 safetensors set the
pipeline itself loads, so this is the first architecture here that is not GGUF
(2026-09-06).
