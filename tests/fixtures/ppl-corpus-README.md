# ppl-corpus.txt — perplexity-gate held-out corpus

`ppl-corpus.txt` is the frozen corpus for the perplexity-delta parity gate
(`ppl_parity` in `tests/parity.rs`; see `docs/parity.md` "Perplexity gate").

**Source**: the head of the *test* split of WikiText-2 (raw, `wikitext-2-raw-v1`),
the standard language-model perplexity corpus. Fetched verbatim from the
Hugging Face `Salesforce/wikitext` dataset (config `wikitext-2-raw-v1`, split
`test`) via the datasets-server rows API, concatenated in row order and
truncated at a paragraph boundary. Nothing is stripped or rewritten — the text
is as-is, including the dataset's ` @-@ ` / ` = heading = ` markers.

**License**: WikiText is distributed under the Creative Commons
Attribution-ShareAlike License (CC BY-SA 4.0), derived from Wikipedia. Retained
here under those terms for held-out evaluation.

**Why held-out**: it is not among the parity prompts
(`tests/fixtures/parity-prompts.json`), so scoring it does not overlap the
token-level gates. It is mixed-register prose, which exercises a broad token
distribution for the mean-NLL comparison.

**Size**: 4218 tokens under the Qwen 3.6 tokenizer (`add_special_tokens=false`,
nothing prepended — the vocabulary has no BOS), 4217 of them scored. Both official
checkpoints share the tokenizer, so the token stream is identical for each.

Regenerating or resizing the corpus invalidates every frozen
`reference-ppl-<checkpoint>.json` and `PPL_NLL_DELTA_MAX` — all must be
recalibrated (see docs/parity.md "Perplexity gate").
