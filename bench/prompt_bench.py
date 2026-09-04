"""Prompt A/B bench for Hindsight retain — prompt-vs-prompt, model frozen.

The companion to local-llm's retain_bench.py (which is model-vs-model with the
prompt frozen). This one varies the bank's `retain_custom_instructions` (the
block edited via the Hindsight UI box) while holding model, dataset, schema,
and all other prompt bytes identical — so score deltas are attributable to the
prompt variant alone.

Scoring is IMPORTED from local-llm's golden bench (single source of truth —
same scorer, same frozen dataset, no forked copies). Results live HERE in the
hindsight skill so model baselines and prompt experiments never mix.

Prompt splicing: the frozen production system prompt = base template +
`custom instructions` span. The span is bounded by the '═══ RULES ═══' anchor
(start of the bank's block) and the template's '════...══\nFACT FORMAT'
section. A variant file replaces that span. Correctness is proven at startup:
splicing the EXTRACTED current span back must reproduce the frozen prompt
byte-for-byte (sha256 equal), or the bench refuses to run.

Usage:
  python3 prompt_bench.py <port> <model> [--host H] [--runs N] [--temp T]
      [--variants all|name1,name2] [--dataset v1|v2]

Variant files: prompt-variants/<name>.txt — each is a FULL custom-instructions
block (RULES / WHAT TO EXTRACT / WHAT TO SKIP sections). `_production.txt` is
the extracted current block; it doubles as the control arm and the round-trip
proof. Results: prompt-bench-results/<stamp>_<variant>.json (prompt sha256 in
every fingerprint).
"""
import argparse
import hashlib
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))                      # <repo>/bench
REPO = os.path.dirname(HERE)
GOLDEN_DIR = os.path.join(REPO, "golden")
VARIANTS_DIR = os.path.join(REPO, "prompt-variants")
RESULTS_DIR = os.path.join(REPO, "results")

sys.path.insert(0, GOLDEN_DIR)
import retain_bench as rb  # noqa: E402  (scorer + stream_call + dataset loader)

CUSTOM_START = "═══ RULES ═══"
TEMPLATE_TAIL = "══════════════════════════════════════════════════════════════════════════\nFACT FORMAT - BE CONCISE"


def sha256_text(t):
    return hashlib.sha256(t.encode()).hexdigest()


def load_template():
    """Frozen production system prompt, split into (prefix, current_rules, suffix)."""
    with open(os.path.join(GOLDEN_DIR, "retain_system_prompt.txt")) as f:
        sys_prompt = f.read()
    i = sys_prompt.index(CUSTOM_START)
    j = sys_prompt.index(TEMPLATE_TAIL)
    prefix, span, suffix = sys_prompt[:i], sys_prompt[i:j], sys_prompt[j:]
    current = span.rstrip("\n")
    # round-trip proof: reassembling must be byte-identical
    rebuilt = prefix + current + "\n\n" + suffix
    if sha256_text(rebuilt) != sha256_text(sys_prompt):
        print("FATAL: prompt splice round-trip FAILED — anchors drifted. "
              "Re-capture the production prompt before running prompt A/Bs.")
        sys.exit(1)
    return prefix, current, suffix


def splice(prefix, suffix, variant_text):
    return prefix + variant_text.rstrip("\n") + "\n\n" + suffix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port")
    ap.add_argument("model")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--temp", type=float, default=0.1)
    ap.add_argument("--dataset", default="v2", choices=["v1", "v2"])
    ap.add_argument("--variants", default="all",
                    help="comma list of variant names in prompt-variants/, or 'all'")
    ap.add_argument("--only", metavar="SCEN1,SCEN2", help="limit scenarios (smoke tests)")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--thinking-off-kwargs", action="store_true")
    ap.add_argument("--no-grammar", action="store_true")
    args = ap.parse_args()

    prefix, current_rules, suffix = load_template()
    print(f"splice round-trip: PASS (frozen prompt sha256 {sha256_text(prefix + current_rules + chr(10) + chr(10) + suffix)[:16]})")

    with open(os.path.join(GOLDEN_DIR, ("retain_golden_v2.1.json" if args.dataset == "v2" else "retain_golden_v1.json"))) as f:
        GOLDEN = json.load(f)
    rb.GOLDEN = GOLDEN  # banned patterns etc. read the module global
    with open(os.path.join(GOLDEN_DIR, "retain_response_schema.json")) as f:
        SCHEMA = json.load(f)
    RESPONSE_FORMAT = {"type": "json_schema",
                       "json_schema": {"name": "FactExtractionResponse",
                                       "schema": SCHEMA, "strict": True}}

    # collect variants
    if not os.path.isdir(VARIANTS_DIR):
        sys.exit(f"no variants dir: {VARIANTS_DIR}")
    names = (sorted(os.listdir(VARIANTS_DIR)) if args.variants == "all"
             else [v.strip() + ".txt" for v in args.variants.split(",")])
    variants = {}
    for fname in names:
        if not fname.endswith(".txt"):
            continue
        with open(os.path.join(VARIANTS_DIR, fname)) as f:
            variants[fname[:-4]] = f.read()
    if not variants:
        sys.exit("no variants resolved")

    # warmup once (discarded)
    try:
        rb.stream_call(args.host, args.port, args.model,
                       [{"role": "user", "content": "Say OK."}], 16, args.temp, False)
        print("warmup: done (discarded)\n")
    except Exception as e:
        print(f"warmup: FAILED — {e}\n")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    only = set(args.only.split(",")) if args.only else None
    league = []
    EP = {"https": False, "omit_temp": False, "use_mct": False, "base_path": ""}

    for vname, vtext in variants.items():
        prompt = splice(prefix, suffix, vtext)
        is_control = sha256_text(prompt) == sha256_text(prefix + current_rules + "\n\n" + suffix)
        print(f"═══ variant: {vname}{' (CONTROL — identical to production)' if is_control else ''} ═══")
        scenario_scores, raw = {}, {}
        for scenario in GOLDEN["scenarios"]:
            sid = scenario["id"]
            if only and sid not in only:
                continue
            meta = {"platform": GOLDEN["platform"], "session_id": "prompt-bench",
                    "turn_index": "1", "retained_at": "2026-09-03T19:00:00.000Z",
                    "agent_identity": GOLDEN["agent_identity"],
                    "message_count": str(2 * len(scenario["turns"]))}
            chunks = (scenario["chunks"] if scenario.get("chunks")
                      else [[{"role": t[0], "content": t[1]} for t in scenario["turns"]]])
            chunk_user_msgs = [rb.build_user_message(c, GOLDEN, meta, chunk_index=i + 1,
                                                     total_chunks=len(chunks))
                               for i, c in enumerate(chunks)]
            scorings = []
            for i in range(args.runs):
                try:
                    final_text, fmt_used, last = rb.pipeline_pass(
                        prompt, chunk_user_msgs, args, EP, RESPONSE_FORMAT)
                except Exception as e:
                    print(f"  {sid} run{i+1}: FAILED ({type(e).__name__}: {str(e)[:120]})")
                    continue
                s = rb.score_run(scenario, final_text)
                s["_text"] = final_text[:4000]
                scorings.append(s)
            if scorings:
                scenario_scores[sid] = rb.score_scenario(scorings)
                raw[sid] = scorings
                ss = scenario_scores[sid]
                print(f"  {sid:<22} recall={ss['recall_mean']}  yield={ss['yield_mean']}"
                      f"  viol={ss['violations_total']}")
        comp = rb.composite_score(scenario_scores)
        league.append({"variant": vname, "control": is_control,
                       "prompt_sha256": sha256_text(prompt)[:16], **comp})
        out = os.path.join(RESULTS_DIR, f"{stamp}_{vname}.json")
        with open(out, "w") as f:
            json.dump({"variant": vname, "prompt_sha256": sha256_text(prompt),
                       "is_control": is_control, "composite": comp,
                       "scenarios": scenario_scores, "raw_runs": raw},
                      f, indent=1, ensure_ascii=False)
        print(f"  ⇒ SCORE {comp['score']}/100  (saved {os.path.basename(out)})\n")

    league.sort(key=lambda x: -x["score"])
    print("═══ PROMPT LEAGUE TABLE ═══")
    for row in league:
        tag = " ← production" if row["control"] else ""
        print(f"  {row['score']:>6.2f}  {row['variant']:<24} recall={row['recall']:<7}"
              f" yield={row['yield']:<7} discipline={row['discipline']:<7}{tag}")
    best = league[0]
    ctrl = next((r for r in league if r["control"]), None)
    if ctrl and best is not ctrl:
        delta = best["score"] - ctrl["score"]
        verdict = "DECISIVE" if delta > 1.5 else "within noise floor — rerun at n=5 before believing"
        print(f"\nwinner: {best['variant']} (+{delta:.2f} over production) — {verdict}")


if __name__ == "__main__":
    main()
