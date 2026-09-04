# memory-extract-bench

A deterministic quality + speed benchmark for **memory-extraction pipelines** — the LLM call that reads a conversation/document and emits structured facts as JSON (the "retain" step of agent memory systems like [Hindsight](https://github.com/vectorize-io/hindsight)).

Built out of frustration with the standard practice of judging extraction quality by vibes. Everything here is deterministic: frozen golden dataset, string/field-match scoring, **no LLM judge**, fingerprinted baselines, and a measured noise floor.

Design informed by [run-llama/ExtractBench](https://github.com/run-llama/ExtractBench) (their null-semantics, embedded-date acceptance, and fail-as-zero rules are adopted; their paper is worth reading). This bench applies those ideas to *conversational* memory extraction rather than enterprise PDFs.

## What it measures

Two harnesses, one scoring engine:

| Harness | Varies | Freezes | Answers |
|---|---|---|---|
| `bench/retain_bench.py` | the **model** | the prompt | "Is model B better than model A at extraction?" |
| `bench/prompt_bench.py` | the **prompt** | the model | "Does prompt variant X beat variant Y on the same model?" |

Scoring is **graded**, per ground-truth item:

```
item credit = 0.5·(core strings) + 0.25·(detail strings) + 0.25·(metadata checks)
              ────────────────────────────────────────────────────────────────
                                 achievable max
```

- **Core** — strings that must appear (with accepted spelling variants)
- **Details** — bonus strings (exact timestamps, transports, quantities)
- **Metadata** — `fact_type`, `fact_kind`, `occurred_start` (ISO date), **and `expect_absent`**: fields that must be null/absent. Invented values on should-be-null fields are *penalized* (ExtractBench's explicit-null semantics)
- **Composite** = `50% recall + 25% yield + 15% discipline + 10% consistency`

Where:

- **recall** — mean per-item credit (headroom: detail and metadata axes)
- **yield** — symmetric credit density `1−|1−credit/facts|`. Punishes *both* failure modes: junk/fragmentation (too many facts) **and** mega-merging (too few)
- **discipline** — schema violations + over-extraction past the scenario cap + banned patterns (unresolved relative dates like "yesterday" in event facts, wrong entity naming)
- **consistency** — run-to-run credit spread (a flaky model scores worse than a stable one)

Plus: streaming TTFT and tok/s on the real prompt, and an optional speed-suite whose prompts mirror `llama.cpp`'s standard endpoint bench for cross-comparability.

## The dataset (v2.1.0)

8 scenarios, ground truth by construction, all names/hosts/domains placeholders:

| Scenario | What it isolates |
|---|---|
| `rich_transcript` | mixed session: decisions, exact-time details, a deadline **retraction** |
| `noise_only` | pure smalltalk — correct output is **zero facts** |
| `consolidation_heavy` | two topics, each mentioned 3× — exactly one fact per topic |
| `temporal` | relative-date traps ("yesterday", "coming monday"), a moved appointment |
| `no_repeats` | five distinct single-mention facts — pure recall, numeric traps (60/40 ≠ 40/60) |
| `entity_naming` | naming rules + fact_kind=assistant classification + coreference |
| `retraction_update` | corrections and cancellations — belief-update discipline |
| `multi_chunk` | facts spread across **3 chunks**, extracted independently then consolidated — the real pipeline shape of chunked memory systems |

The prompt is **byte-verbatim production**: captured from the live pipeline (system prompt with embedded JSON schema + per-bank custom instructions + FOCUS mission in the user message), hash-pinned, with a splice round-trip check that refuses to run if anchors drift.

## Key findings so far

Measured on Gemma-4-E4B (QAT, Q4_K_XL, GTX 1080) vs `gpt-5.6-sol` (~2T, reasoning, via OpenAI-compatible gateway):

1. **The noise floor is real: ±0.74 composite points.** Same model, same params, one hour apart. Any challenger must win by >1.5 points or rerun at n=5. Benchmarks without a measured noise floor are vibes with decimal points.
2. **Bigger is not better at extraction selectivity.** The frontier model beat the 4B on recall and temporal resolution (+9 composite) — and emitted *twice* the junk facts from pure smalltalk, every run, perfectly reliably.
3. **Symmetric yield exposes opposite failure modes.** The 4B *under*-consolidates (fragments a 2-topic transcript into 4 facts); the frontier model *over*-consolidates (merges distinct topics into one mega-fact in the merge step, scoring 0.29 yield on the multi-chunk scenario vs 0.88 local). Same instruction, opposite failure — only a two-sided metric catches this.
4. **The multi-chunk pipeline (per-chunk extraction + merge) does not degrade a 4B model.** Unlike OSS models' collapse on long documents in document-extraction benchmarks, the chunk+consolidate shape scored 0.875/0.875 — and later chunks ran *faster* (prefix caching).
5. **Prompt-space near a small model can be flat.** Four instruction variants (anti-junk skip rules, worked consolidation examples, anti-invention field rules) all landed within the noise floor. The behaviors are capacity limits, not instruction limits. The anti-invention variant *cost* recall — "never guess" language suppressed genuine dates.
6. **Grammar is load-bearing.** With the server-side JSON grammar disabled, the 4B emits zero JSON on the real prompt — numbered analysis prose instead. Production reliability was a model+grammar symbiosis, not model virtue. Cloud `json_ok` is therefore a genuine capability axis (their endpoints vary in structured-output support).

## Usage

```bash
# model-vs-model (prompt frozen)
python3 bench/retain_bench.py 8091 <model.gguf> --runs 3 --speed-suite --save-baseline mymodel

# through any OpenAI-compatible endpoint
python3 bench/retain_bench.py 4000 provider/model --host 127.0.0.1 --api-key KEY \
    --save-baseline cloud-model

# OpenAI platform (handles max_completion_tokens, no-temperature models)
OPENAI_API_KEY=sk-... python3 bench/retain_bench.py 8091 gpt-5.6 --openai --runs 3

# prompt-vs-prompt (model frozen) — variants in prompt-variants/
python3 bench/prompt_bench.py 8091 <model.gguf> --runs 3

# compare against a saved baseline (cross-version comparisons are refused)
python3 bench/retain_bench.py 8091 <model.gguf> --runs 3 \
    --compare baselines/<baseline>.json
```

Baselines are fingerprinted: model + GGUF sha256, prompt hash, dataset + bench version, temperature, grammar mode, timestamp. `--compare` warns on grammar-mode mismatches (content scores comparable; `json_ok`/speed are not).

## Honest limitations

- Ground truth is by construction on a small dataset (8 scenarios, ~23 items). It measures *our* definition of good extraction: selective, consolidated, temporally resolved, schema-faithful. Your memory pipeline may weight differently — fork the dataset, it's one JSON file.
- One summarizer transcript per scenario. Synthetic transcripts are pipeline-shaped but not distributionally real; validate any model swap against your live pipeline before rewiring (we learned this the hard way — see finding 6).
- Deterministic scoring can't judge *prose quality* of extracted facts, only presence/absence of the graded signals.

## License

MIT
