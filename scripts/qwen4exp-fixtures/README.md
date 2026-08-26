# qwen4exp fixture generator

`generate.py` runs the HuggingFace transformers reference implementation of
`qwen4_exp` on tiny seeded configs and writes the golden fixtures under
`tests/fixtures/qwen4exp/` (see the README there for what each file pins).

## Venv recipe

transformers main requires Python >= 3.10; this machine's system python is
3.9, so use uv's managed interpreter:

```sh
uv venv --python 3.12 /path/to/venv
uv pip install --python /path/to/venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python /path/to/venv/bin/python "git+https://github.com/huggingface/transformers" safetensors numpy
/path/to/venv/bin/python scripts/qwen4exp-fixtures/generate.py
```

Keep the venv OUT of the repo. The committed fixtures were generated against
transformers `598d8ba8baaec7fec5a22da0e2844c7bf4ea20e1` (5.16.0.dev0, recorded
in each fixture's `meta`); regenerating against a newer main may legitimately
change values — diff `meta.transformers_commit` before assuming a bug.

## Determinism

Every weight the generator randomizes is dumped into the fixtures, so
consumers never reproduce torch's RNG. Fixed `torch.Generator` seeds per
fixture (41-45); the QSA above-budget case additionally SEARCHES its data seed
(recorded as `config.case_above_budget_data_seed`) so every query's block
top-k sits a clear margin (> 0.05) away from a score tie — relu'd scores tie
at 0.0 easily, and a tie would make the fixture depend on `torch.topk`'s
tie-breaking.

The generator asserts, at generation time: the intermediate replica of the PLE
forward is bit-identical to the module output; the QSA selection replica
reproduces the module's selected sets; below-budget selection equals dense;
the hyper-connection write-back anchors on the raw un-normed stream; and the
selected-count arithmetic (budget+1 / budget+ratio-1 / budget) for the three
tail cases.
