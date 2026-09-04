#!/usr/bin/env python3
"""Hindsight retain QUALITY + SPEED benchmark v2 — golden dataset, GRADED deterministic scoring.

v2.0.0 (2026-09-03): graded per-item credit replaces binary must-strings so
ceiling scenarios carry signal again:
  per-item credit = core 0.5 (all core strings present)
                  + details 0.25 * (matched/total detail groups)
                  + metadata 0.25 * (satisfied/total metadata checks)
  metadata checks: fact_type, fact_kind, occurred (occurred_start date prefix)
Composite axes: recall=mean per-item credit (has headroom now), yield=total
credit / facts emitted (junk facts + fragmentation dilute), discipline=1 -
violations/runs (max_facts over-extraction + banned patterns + schema),
consistency=1 - stdev/mean of scenario credit across runs (flaky costs points).
Composite = 100 * (0.50*recall + 0.25*yield + 0.15*discipline + 0.10*consistency).

v1 datasets (binary must/alt format) still load — scored at full credit per
item, no detail/metadata axis (for reproducing v1 baselines).

The real production retain prompt: byte-verbatim system prompt (captured from
Hindsight's llm_requests trace table), FOCUS mission block, real
FactExtractionResponse json_schema (grammar-enforced), prod temperature.
See references/benchmark-methodology.md. No LLM judge anywhere.

Usage:
  python3 retain_bench.py <port> <model> [--host H] [--runs N] [--temp T]
      [--dataset v1|v2] [--speed-suite] [--save-baseline LABEL] [--compare FILE]
      [--thinking-off-kwargs] [--only scen1,scen2]

Artifacts:
  scripts/golden/baselines/<label>_<date>.json   fingerprinted (GGUF sha256, prompt hash, dataset version)
  scripts/golden/results/<label>_<date>_raw.json every run's full model output
"""
import argparse
import hashlib
import json
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))          # <repo>/bench
REPO = os.path.dirname(HERE)
GOLDEN_DIR = os.path.join(REPO, "golden")
RESULTS_DIR = os.path.join(REPO, "results")
BASELINES_DIR = os.path.join(REPO, "baselines")

# Speed-suite prompts: IDENTICAL to llm-bench-endpoint.py for cross-comparability.
SHORT = "Give me three names for a pet capybara, one line each."
MEDIUM = ("Explain how Mixture-of-Experts routing works, with the tradeoffs vs "
          "dense models. About 250 words.")
LONGFORM = ("Write a detailed beginner's guide to choosing a homelab GPU: VRAM classes, "
            "used-market traps, power and cooling, and when an MoE offload setup beats a "
            "bigger card. Use headings and concrete numbers.")

W_CREDIT = {"core": 0.5, "details": 0.25, "metadata": 0.25}


def sha256_text(t):
    return hashlib.sha256(t.encode()).hexdigest()


def gguf_fingerprint(model):
    """Best-effort: find the GGUF on disk and hash it (None for cloud endpoints)."""
    candidates = [
        f"{model}",
        f"{model}.gguf",
        os.path.expanduser(f"~/models/{model}"),
        os.path.expanduser(f"~/models/gguf/{model}"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            h = hashlib.sha256()
            with open(c, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 22), b""):
                    h.update(chunk)
            return {"file": c, "sha256": h.hexdigest()}
    return {"file": None, "sha256": None}


def build_user_message(chunk_pairs, ds, run_meta, chunk_index=1, total_chunks=1):
    """Byte-faithful reconstruction of the production user message.

    Order (from the live llm_requests trace, 2026-09-03): JSON_MODE_USER_HINT,
    FOCUS mission block (bank retain_mission), then the standard extraction
    scaffold with Chunk/Event Date/Context/Metadata/Text.
    """
    chunk_text = json.dumps(chunk_pairs, ensure_ascii=False)
    meta_lines = "\n".join(f"  {k}: {v}" for k, v in run_meta.items())
    return (
        "Return valid json only.\n\n"
        "══════════════════════════════════════════════════════════════════════════\n"
        "FOCUS — What to retain for this bank (takes priority over the general guidelines)\n"
        "══════════════════════════════════════════════════════════════════════════\n\n"
        f"{ds['retain_mission']}\n\n"
        "Extract facts from the following text chunk.\n\n"
        f"Chunk: {chunk_index}/{total_chunks}\n"
        f"Event Date: {ds['event_date_human']} ({ds['event_date_iso']})\n"
        f"Context: {ds['conversation_context']}\n"
        f"Metadata:\n{meta_lines}\n\n"
        f"Text:\n{chunk_text}"
    )


def stream_call(host, port, model, messages, max_tokens, temperature, use_kwargs,
                response_format=None, timeout=600, api_key=None, https=False,
                omit_temp=False, use_mct=False, base_path=""):
    """One streaming completion. Returns measurements + text + finish reason.

    https=True for internet endpoints (auto-on for port 443). omit_temp drops
    temperature (required by OpenAI reasoning-family models). use_mct sends
    max_completion_tokens instead of max_tokens (same family). include_usage
    is attempted for accurate token counts (reasoning tokens included) and
    silently dropped for endpoints that reject it.
    """
    scheme = "https" if (https or str(port) == "443") else "http"
    body = {"model": model, "messages": messages, "stream": True}
    if use_mct:
        body["max_completion_tokens"] = max_tokens
    else:
        body["max_tokens"] = max_tokens
    if not omit_temp:
        body["temperature"] = temperature
    if use_kwargs:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    if response_format:
        body["response_format"] = response_format
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    def _post(with_usage):
        b = dict(body)
        if with_usage:
            b["stream_options"] = {"include_usage": True}
        req = urllib.request.Request(
            f"{scheme}://{host}:{port}{base_path}/v1/chat/completions",
            data=json.dumps(b).encode(), headers=headers)
        return urllib.request.urlopen(req, timeout=timeout)

    t0 = time.time()
    chunks, text, usage_n, finish, reasoning_n = [], "", None, None, None
    try:
        r = _post(True)
    except urllib.error.HTTPError:
        r = _post(False)  # endpoint rejected stream_options — retry plain
    with r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            try:
                d = json.loads(line[6:])
            except Exception:
                continue
            if d.get("usage"):
                u = d["usage"]
                if u.get("completion_tokens"):
                    usage_n = u["completion_tokens"]
                reasoning_n = (u.get("reasoning_tokens")
                               or (u.get("completion_tokens_details") or {}).get("reasoning_tokens")
                               or reasoning_n)
            ch = d.get("choices") or [{}]
            if ch[0].get("finish_reason"):
                finish = ch[0]["finish_reason"]
            delta = ch[0].get("delta", {}) or {}
            piece = delta.get("content") or ""
            if piece:
                chunks.append((time.time(), len(piece)))
                text += piece
    if not chunks:
        raise RuntimeError("no content chunks received")
    n = usage_n or len(chunks)
    ttft_ms = (chunks[0][0] - t0) * 1000
    span = chunks[-1][0] - chunks[0][0]
    gen_tps = (n - 1) / span if span > 0 and n > 1 else 0.0
    return {"ttft_ms": ttft_ms, "gen_tps": gen_tps, "n": n, "reasoning_tokens": reasoning_n,
            "wall": chunks[-1][0] - t0, "finish": finish, "text": text}


# ------------------------------------------------- cloud-compat machinery --

def extract_json_payload(raw):
    """Recover the JSON payload from arbitrary model output.

    Mirrors what Hindsight's own parsers survive in production (e.g. MiniMax
    bleeding <think> blocks into content). Order: direct parse → strip
    <think> blocks → ```json fenced block → outermost {..} brace span.
    Returns (payload_text, recovery_flags). Flags include the recovery path
    taken so 'clean JSON' vs 'recovered JSON' stays visible in results.
    """
    flags, text = [], raw
    if "<think>" in text:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
        flags.append("think_stripped")
    try:
        json.loads(text)
        return text, flags + ["direct"]
    except Exception:
        pass
    fm = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fm:
        try:
            json.loads(fm.group(1))
            return fm.group(1), flags + ["fenced"]
        except Exception:
            pass
    i, j = text.find("{"), text.rfind("}")
    if 0 <= i < j:
        span = text[i:j + 1]
        try:
            json.loads(span)
            return span, flags + ["brace_span"]
        except Exception:
            pass
    return text, flags + ["unparseable"]


# ---------------------------------------------------------------- scoring --

# ------------------------------------------------- value comparison (v2.1) --
# Informed by run-llama/ExtractBench confidence_scoped/value_matching.py:
# explicit-null semantics, temporal quarantine, embedded-date acceptance,
# and per-comparison reason strings for failure forensics.

_MONTHS = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"])}
for _m, _n in list(_MONTHS.items()):
    _MONTHS[_m[:3]] = _n

_NULL_STRINGS = {"", "n/a", "na", "none", "null", "unknown", "-"}


def is_effectively_null(v):
    """Null, blank, N/A-style strings, and all-null containers are equivalent."""
    if v is None:
        return True
    if isinstance(v, str):
        return v.strip().casefold() in _NULL_STRINGS
    if isinstance(v, dict):
        return all(is_effectively_null(x) for x in v.values())
    if isinstance(v, list):
        return all(is_effectively_null(x) for x in v)
    return False


def extract_dates(text):
    """All (y, m, d) dates found in a prose string. Deterministic, US month-day
    for slashed dates. Full dates only — day-only forms stay with substring matching."""
    found = set()
    for y, m, d in re.findall(r"\b(\d{4})-(\d{2})-(\d{2})\b", text):
        found.add((int(y), int(m), int(d)))
    for m, d, y in re.findall(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", text):
        found.add((int(y), int(m), int(d)))
    for mon, d, y in re.findall(
            r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
            r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?\s+"
            r"(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b", text, re.IGNORECASE):
        found.add((int(y), _MONTHS[mon[:3].casefold()], int(d)))
    return found


def entry_is_date(entry):
    """Does this ground-truth variant list contain a date-like canonical form?"""
    return any(re.search(r"\d{4}-\d{2}-\d{2}|\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", v,
                         re.IGNORECASE) for v in entry)


def _expected_date_tuple(entry, occurred_prefix):
    """Resolve an entry's date to (y, m, d) using the item's occurred year if needed."""
    y = int(str(occurred_prefix)[:4]) if occurred_prefix else None
    for v in entry:
        m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", v.strip())
        if m:
            return int(m.group(1)), int(m.group(2)), int(m.group(3))
        m = re.match(r"^(january|february|march|april|may|june|july|august|september|october|"
                     r"november|december|jan|feb|mar|apr|jun|jul|aug|sept|oct|nov|dec)\.?\s+"
                     r"(\d{1,2})(?:st|nd|rd|th)?$", v.strip(), re.IGNORECASE)
        if m and y:
            return y, _MONTHS[m.group(1)[:3].casefold()], int(m.group(2))
    return None


def _entry_hit(variants, hay):
    """A ground-truth entry is a list of accepted variants (any-match, lowercase)."""
    return any(v.lower() in hay for v in variants)


def _variants(entry):
    """Coerce a legacy entry to a variants list: 'x' | ['a','b'] | ['a', ['b']]."""
    if isinstance(entry, str):
        return [entry]
    out = []
    for v in entry:
        out.extend(v if isinstance(v, list) else [v])
    return out


def item_credit(exp, facts):
    """Graded credit [0..1] for one expected item against all extracted facts.

    v2.1: adds expect_absent (explicit-null) checks, embedded-date acceptance
    for date-bearing core entries, occurred-prefix lists, and failure reasons.
    Uses the best-matching fact (merged facts covering several topics score on
    each of their items independently). Returns (credit, matched, fact, reason).
    """
    if "must" in exp:  # legacy v1 item: binary, full credit
        for f in facts:
            hay = json.dumps(f, ensure_ascii=False).lower()
            musts = [m.lower() for m in exp["must"]]
            alts = [a.lower() for a in exp.get("alt", [])]
            if musts[0] in hay or any(a in hay for a in alts):
                if all(m in hay for m in musts[1:]):
                    return 1.0, True, f, "matched"
        return 0.0, False, None, "missed"

    occurred = exp.get("metadata", {}).get("occurred")
    occurred_prefixes = [occurred] if isinstance(occurred, str) else list(occurred or [])
    absent_fields = exp.get("expect_absent", [])
    best = (0.0, False, None, "missed")
    for f in facts:
        hay = json.dumps(f, ensure_ascii=False).lower()
        core_entries = [_variants(e) for e in exp["core"]]
        core_hits, date_resolved = [], False
        for e in core_entries:
            if _entry_hit(e, hay):
                core_hits.append(True)
                continue
            # embedded-date fallback (ExtractBench rule): date-bearing entries may
            # be satisfied by a full date embedded in prose — accepted only when
            # exactly ONE distinct date appears (shotgun answers earn nothing)
            if occurred_prefixes and entry_is_date(e):
                want = _expected_date_tuple(e, occurred_prefixes[0])
                embedded = extract_dates(json.dumps(f, ensure_ascii=False))
                if want and len(embedded) == 1 and next(iter(embedded)) == want:
                    core_hits.append(True)
                    date_resolved = True
                    continue
            core_hits.append(False)
        if not all(core_hits):
            miss = core_entries[core_hits.index(False)][0]
            continue
        credit = W_CREDIT["core"]
        reasons = ["matched:embedded-date" if date_resolved else "matched"]
        achievable = W_CREDIT["core"]
        details = [_variants(d) for d in exp.get("details", [])]
        if details:
            hits = sum(_entry_hit(d, hay) for d in details)
            achievable += W_CREDIT["details"]
            credit += W_CREDIT["details"] * hits / len(details)
            if hits < len(details):
                reasons.append(f"details:{hits}/{len(details)}")
        meta = exp.get("metadata", {})
        expect_absent = exp.get("expect_absent", [])
        meta_checks = len(meta) + len(absent_fields)
        if meta_checks:
            achievable += W_CREDIT["metadata"]
            m_ok = 0
            if "fact_type" in meta:
                m_ok += int(f.get("fact_type") == meta["fact_type"])
            if "fact_kind" in meta:
                m_ok += int(f.get("fact_kind") == meta["fact_kind"])
            if occurred_prefixes:
                os_start = str(f.get("occurred_start") or "")
                if any(os_start.startswith(p) for p in occurred_prefixes):
                    m_ok += 1
                elif os_start:
                    reasons.append("occurred:wrong-date")
                else:
                    reasons.append("occurred:missing")
            for field in absent_fields:
                if is_effectively_null(f.get(field)):
                    m_ok += 1
                else:
                    reasons.append(f"phantom:{field}")
            credit += W_CREDIT["metadata"] * m_ok / meta_checks
        credit /= achievable  # normalize by achievable max
        if credit > best[0]:
            best = (credit, True, f, "+".join(reasons))
    return best


def score_run(scenario, out_text):
    """Score one run. Returns metrics dict + per-item detail."""
    sc_checks = scenario.get("checks", {})
    expected = scenario.get("expected", [])
    m = {"json_ok": False, "recovery": [], "schema_field_violations": 0, "facts": [],
         "banned_hits": [], "over_extraction": 0, "item_credits": {},
         "recall": None, "yield": 0.0}
    payload_text, flags = extract_json_payload(out_text)
    m["recovery"] = flags
    try:
        parsed = json.loads(payload_text)
        facts = parsed.get("facts", []) if isinstance(parsed, dict) else None
        if facts is None or not isinstance(facts, list):
            return m
        m["json_ok"] = "unparseable" not in flags
    except Exception:
        return m

    required = {"what", "when", "where", "who", "why", "fact_type"}
    clean_facts = []
    for f in facts:
        if not isinstance(f, dict):
            m["schema_field_violations"] += 1
            continue
        if required - set(f.keys()):
            m["schema_field_violations"] += 1
        clean_facts.append(f)
        m["facts"].append(f)

    # banned patterns (naming + unresolved-relative-dates)
    for bp in GOLDEN.get("banned_output_patterns", []):
        if "scenarios" in bp and scenario["id"] not in bp["scenarios"]:
            continue
        flags = re.IGNORECASE if bp.get("ignore_case") else 0
        for f in m["facts"]:
            fields = bp.get("applies_to", "what").split("+")
            hay = " ".join(str(f.get(k.strip(), "")) for k in fields)
            if re.search(bp["pattern"], hay, flags):
                m["banned_hits"].append({"pattern": bp["pattern"], "fact": hay[:120]})

    # graded recall: mean per-item credit + failure forensics
    total_credit, useful = 0.0, 0
    m["item_reasons"] = {}
    for exp in expected:
        credit, matched, _f, reason = item_credit(exp, clean_facts)
        m["item_credits"][exp["id"]] = round(credit, 3)
        m["item_reasons"][exp["id"]] = reason if matched or credit == 0 else f"partial:{reason}"
        total_credit += credit
        useful += int(matched)
    m["recall"] = total_credit / len(expected) if expected else None

    # yield (precision proxy): symmetric credit density — both junk-fragmentation
    # (yield << 1) AND mega-merging multiple topics into one fact (yield > 1) lose points.
    if sc_checks.get("max_facts", 99) == 0:
        m["yield"] = 1.0 if not clean_facts else 0.0
    else:
        raw_yield = (total_credit / len(clean_facts)) if clean_facts else 0.0
        m["yield"] = max(0.0, 1.0 - abs(1.0 - raw_yield))

    # consolidation discipline
    max_facts = sc_checks.get("max_facts")
    if max_facts is not None and len(clean_facts) > max_facts:
        m["over_extraction"] = len(clean_facts) - max_facts
    return m


def score_scenario(scorings):
    """Aggregate run scores for one scenario."""
    runs = len(scorings) or 1
    recalls = [s["recall"] for s in scorings if s["recall"] is not None]
    yields_ = [s["yield"] for s in scorings]
    violations = [s["over_extraction"] + len(s["banned_hits"]) + s["schema_field_violations"]
                  for s in scorings]
    # per-run scenario totals for consistency (undefined when no items — e.g.
    # noise_only: treat as perfect stability, there is nothing to vary)
    run_totals = []
    for s in scorings:
        tot = sum(s["item_credits"].values())
        run_totals.append(tot)
    mean_tot = statistics.mean(run_totals) if run_totals else 0.0
    stdev_tot = statistics.stdev(run_totals) if len(run_totals) > 1 else 0.0
    if stdev_tot == 0:
        consistency = 1.0          # zero variance = fully consistent (incl. no-item scenarios)
    elif mean_tot > 0:
        consistency = max(0.0, 1.0 - (stdev_tot / mean_tot))
    else:
        consistency = 0.0
    return {
        "runs": runs,
        "json_ok_all": all(s["json_ok"] for s in scorings),
        "recall_mean": round(statistics.mean(recalls), 4) if recalls else None,
        "recall_runs": [round(r, 3) for r in recalls],
        "yield_mean": round(statistics.mean(yields_), 4),
        "violations_total": sum(violations),
        "consistency": round(consistency, 4),
    }


def composite_score(scenario_scores):
    """recall 50% / yield 25% / discipline 15% / consistency 10%."""
    recalls, yields_, disciplines, consistencies = [], [], [], []
    for s in scenario_scores.values():
        if s["recall_mean"] is not None:
            recalls.append(s["recall_mean"])
        yields_.append(s["yield_mean"])
        disciplines.append(max(0.0, 1.0 - s["violations_total"] / s["runs"]))
        consistencies.append(s["consistency"])
    recall = statistics.mean(recalls) if recalls else 0.0
    yield_m = statistics.mean(yields_) if yields_ else 0.0
    discipline = statistics.mean(disciplines) if disciplines else 0.0
    consistency = statistics.mean(consistencies) if consistencies else 0.0
    score = 100 * (0.50 * recall + 0.25 * yield_m + 0.15 * discipline + 0.10 * consistency)
    return {"recall": round(recall, 4), "yield": round(yield_m, 4),
            "discipline": round(discipline, 4), "consistency": round(consistency, 4),
            "score": round(score, 2)}


# -------------------------------------------------------------------- main --

# ------------------------------------------------- multi-chunk pipeline ----

MERGE_SYSTEM = (
    "You are consolidating fact-extraction outputs from consecutive chunks of the same "
    "conversation. Merge them into ONE final result following the same schema and rules: "
    "consolidate duplicate topics into single facts (fold in later details and corrections), "
    "never emit two facts about the same topic, drop facts superseded by later corrections, "
    "keep entity naming and temporal resolution. Output ONLY the final JSON.")


def _stream_with_fallback(args, messages, EP, RESPONSE_FORMAT):
    """One streaming call with the response_format fallback ladder. Returns (resp, fmt_used)."""
    formats = [None] if args.no_grammar else [RESPONSE_FORMAT, {"type": "json_object"}, None]
    resp, fmt_used = None, None
    for attempt, fmt in enumerate(formats):
        try:
            resp = stream_call(args.host, args.port, args.model, messages,
                               args.max_tokens, args.temp, args.thinking_off_kwargs,
                               response_format=fmt, **EP)
            fmt_used = ("json_schema" if fmt and fmt.get("type") == "json_schema"
                        else "json_object" if fmt else "plain")
            return resp, fmt_used
        except urllib.error.HTTPError as e:
            body_snip = e.read().decode(errors="replace")[:160]
            print(f"    {fmt and fmt.get('type') or 'plain'} rejected (HTTP {e.code}: {body_snip})")
            if e.code in (400, 404, 422) and attempt < len(formats) - 1:
                continue
            raise
    raise RuntimeError("unreachable")


def pipeline_pass(system_prompt, chunk_user_msgs, args, EP, RESPONSE_FORMAT):
    """One extraction pass. Single chunk = single call; multi-chunk = per-chunk
    calls (independent contexts, exactly like production) + a consolidation
    merge call mirroring Hindsight's extraction/consolidation split.
    Returns (final_text, fmt_used, last_call_metrics)."""
    texts, fmt_used, last = [], "json_schema", None
    for ci, user_content in enumerate(chunk_user_msgs):
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}]
        resp, fmt_used = _stream_with_fallback(args, messages, EP, RESPONSE_FORMAT)
        texts.append(resp["text"])
        last = resp
    if len(texts) == 1:
        return texts[0], fmt_used, last
    merge_user = ("Chunk outputs to consolidate:\n\n" + "\n\n---\n\n".join(texts) +
                  "\n\nProduce the final consolidated {\"facts\": [...]} JSON.")
    resp, fmt_used = _stream_with_fallback(
        args, [{"role": "system", "content": MERGE_SYSTEM},
               {"role": "user", "content": merge_user}], EP, RESPONSE_FORMAT)
    return resp["text"], fmt_used, resp


def main():
    global GOLDEN
    ap = argparse.ArgumentParser()
    ap.add_argument("port")
    ap.add_argument("model")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--temp", type=float, default=0.1, help="prod retain temp")
    ap.add_argument("--dataset", default="v2", choices=["v1", "v2"])
    ap.add_argument("--speed-suite", action="store_true")
    ap.add_argument("--save-baseline", metavar="LABEL")
    ap.add_argument("--compare", metavar="FILE")
    ap.add_argument("--thinking-off-kwargs", action="store_true")
    ap.add_argument("--api-key", default=os.environ.get("RETAIN_BENCH_API_KEY")
                    or os.environ.get("OPENAI_API_KEY"),
                    help="Bearer key for the endpoint (or RETAIN_BENCH_API_KEY / OPENAI_API_KEY env)")
    ap.add_argument("--base-url", metavar="URL",
                    help="Full base (e.g. https://api.openai.com or https://host:8443/hf) — "
                         "overrides --host/--port; /v1/chat/completions is appended")
    ap.add_argument("--openai", action="store_true",
                    help="OpenAI-platform preset: HTTPS api.openai.com, max_completion_tokens, "
                         "no temperature (gpt-5*/o*-family behavior), auto OPENAI_API_KEY")
    ap.add_argument("--max-tokens", type=int, default=8192,
                    help="generous by default so thinking models can finish (methodology rule)")
    ap.add_argument("--no-grammar", action="store_true",
                    help="skip json_schema/response_format entirely — test raw JSON discipline "
                         "(cloud parity mode; grammar-enforced local runs are NOT comparable)")
    ap.add_argument("--only", metavar="SCEN1,SCEN2")
    args = ap.parse_args()

    with open(os.path.join(GOLDEN_DIR, ("retain_golden_v2.1.json" if args.dataset == "v2" else "retain_golden_v1.json"))) as f:
        GOLDEN = json.load(f)
    with open(os.path.join(GOLDEN_DIR, "retain_system_prompt.txt")) as f:
        SYSTEM_PROMPT = f.read()
    with open(os.path.join(GOLDEN_DIR, "retain_response_schema.json")) as f:
        SCHEMA = json.load(f)
    # Production sends response_format json_schema (llama.cpp converts to GBNF
    # grammar) — verified in openai_compatible_llm.py:955-963 + live trace.
    RESPONSE_FORMAT = {"type": "json_schema",
                       "json_schema": {"name": "FactExtractionResponse",
                                       "schema": SCHEMA, "strict": True}}

    only = set(args.only.split(",")) if args.only else None

    # --- endpoint resolution -------------------------------------------------
    openai_preset = args.openai
    omit_temp = openai_preset
    use_mct = openai_preset
    https = openai_preset
    if openai_preset and not args.base_url:
        args.base_url = "https://api.openai.com"
        if not (args.api_key or os.environ.get("OPENAI_API_KEY")):
            sys.exit("--openai needs OPENAI_API_KEY (or --api-key)")
    omit_temp = omit_temp or bool(re.search(r"^(o\d|gpt-5)", args.model))
    if args.base_url:
        m = re.match(r"^(https?)://([^/:]+)(?::(\d+))?(/.*)?$", args.base_url)
        if not m:
            sys.exit(f"bad --base-url: {args.base_url}")
        https = (m.group(1) == "https") or https
        args.host, args.port = m.group(2), int(m.group(3) or (443 if https else 80))
        base_path = (m.group(4) or "").rstrip("/")
        BASE_PATH = base_path  # e.g. "" or "/hf"
    else:
        BASE_PATH = ""

    gguf_fp = None if (args.base_url or openai_preset) else gguf_fingerprint(args.model)
    if gguf_fp and not gguf_fp.get("file"):
        gguf_fp = None  # combo/cloud model — no local file to fingerprint
    ds_version = GOLDEN["dataset_version"]
    fp = {"model": args.model,
          "host": (args.base_url or f"{args.host}:{args.port}"),
          "gguf": gguf_fp,
          "temperature": (None if omit_temp else args.temp), "max_tokens": args.max_tokens,
          "runs": args.runs, "dataset_version": ds_version,
          "bench_version": "2.0.0",
          "system_prompt_sha256": sha256_text(SYSTEM_PROMPT),
          "schema_sha256": sha256_text(json.dumps(SCHEMA, sort_keys=True)),
          "response_format": ("plain (no-grammar)" if args.no_grammar
                              else "json_schema (grammar-enforced)"),
          "api_key_used": bool(args.api_key),
          "thinking_off_kwargs": args.thinking_off_kwargs,
          "date": time.strftime("%Y-%m-%d %H:%M")}

    # Common endpoint kwargs threaded into every stream_call
    EP = {"api_key": args.api_key, "https": https, "omit_temp": omit_temp,
          "use_mct": use_mct, "base_path": BASE_PATH}

    print(f"retain_bench v2: {args.model} @ {fp['host']}"
          f"{' (openai preset: mct, no-temp)' if use_mct else ''}  temp={0 if omit_temp else args.temp}"
          f" runs={args.runs}")
    print(f"dataset v{ds_version}  system_prompt sha256={fp['system_prompt_sha256'][:16]}")
    print(f"gguf: {(fp['gguf'] or {}).get('file') if fp['gguf'] else '(cloud endpoint — no local file)'}")
    print(f"gguf sha256: {((fp['gguf'] or {}).get('sha256')[:16] if fp['gguf'] else 'n/a')}\n")

    run_meta_base = {"platform": GOLDEN["platform"], "session_id": "retain-bench",
                     "turn_index": "1", "retained_at": "2026-09-03T19:00:00.000Z",
                     "agent_identity": GOLDEN["agent_identity"]}

    # warmup (discarded)
    try:
        stream_call(args.host, args.port, args.model,
                    [{"role": "user", "content": "Say OK."}], 16, args.temp,
                    args.thinking_off_kwargs, **EP)
        print("warmup: done (discarded)\n")
    except Exception as e:
        print(f"warmup: FAILED — {e}\n")

    scenario_scores, raw = {}, {}
    for scenario in GOLDEN["scenarios"]:
        sid = scenario["id"]
        if only and sid not in only:
            continue
        print(f"--- {sid}{' (multi-chunk)' if scenario.get('chunks') else ''}: "
              f"{scenario['description'][:60]}... ---")
        meta = dict(run_meta_base, message_count=str(2 * len(scenario["turns"])))
        chunks = (scenario["chunks"] if scenario.get("chunks")
                  else [[{"role": t[0], "content": t[1]} for t in scenario["turns"]]])
        chunk_user_msgs = [build_user_message(c, GOLDEN, meta, chunk_index=i + 1,
                                              total_chunks=len(chunks))
                           for i, c in enumerate(chunks)]
        scorings = []
        for i in range(args.runs):
            try:
                final_text, fmt_used, last = pipeline_pass(
                    SYSTEM_PROMPT, chunk_user_msgs, args, EP, RESPONSE_FORMAT)
            except Exception as e:  # timeouts, conn resets, provider hiccups: failed run, bench survives
                print(f"  run{i+1}: FAILED ({type(e).__name__}: {str(e)[:140]})")
                continue
            m = last
            s = score_run(scenario, final_text)
            s["response_format_used"] = fmt_used
            s["_text"] = final_text[:4000]  # persist raw output for forensics
            if m.get("reasoning_tokens"):
                s["reasoning_tokens"] = m["reasoning_tokens"]
            scorings.append(s)
            weak = [f"{k}:{c}" for k, c in s["item_credits"].items() if c < 1.0]
            note = (f"facts={len(s['facts'])} recall={'' if s['recall'] is None else format(s['recall'], '.2f')}"
                    f" yield={s['yield']:.2f} t/s={m['gen_tps']:.1f}")
            if fmt_used != "json_schema":
                note += f" fmt={fmt_used}"
            if s.get("recovery") and s["recovery"] != ["direct"]:
                note += " recovery=" + "+".join(f for f in s["recovery"] if f != "direct")
            if s.get("reasoning_tokens"):
                note += f" reason_tok={s['reasoning_tokens']}"
            flags = []
            if s["over_extraction"]:
                flags.append(f"OVER+{s['over_extraction']}")
            if s["banned_hits"]:
                flags.append(f"BANNED×{len(s['banned_hits'])}")
            if s["schema_field_violations"]:
                flags.append(f"SCHEMA×{s['schema_field_violations']}")
            if weak:
                flags.append("partial[" + " ".join(weak) + "]")
            print(f"  run{i+1}: {note}{'  ⚠ ' + ', '.join(flags) if flags else ''}")
        if scorings:
            scenario_scores[sid] = score_scenario(scorings)
            raw[sid] = scorings
            ss = scenario_scores[sid]
            json_fails = sum(1 for s in scorings if not s["json_ok"])
            json_note = f"  ⚠ JSON FAIL {json_fails}/{len(scorings)}" if json_fails else ""
            print(f"  ⇒ recall={ss['recall_mean']}  yield={ss['yield_mean']}"
                  f"  viol={ss['violations_total']}  consistency={ss['consistency']}{json_note}\n")
        # incremental save: keep partial results alive for long/flaky cloud runs
        if raw and not args.save_baseline:
            os.makedirs(RESULTS_DIR, exist_ok=True)
            _partial = os.path.join(RESULTS_DIR, f"run_{time.strftime('%Y%m%d_%H%M%S')}.json")
            with open(_partial, "w") as f:
                json.dump({"fingerprint": fp, "composite": composite_score(scenario_scores),
                           "scenarios": scenario_scores, "raw_runs": raw}, f,
                          indent=1, ensure_ascii=False)

    # optional speed-suite (cross-comparable with llm-bench-endpoint.py)
    speed_suite = {}
    if args.speed_suite:
        print("--- speed-suite (llm-bench-endpoint.py-identical prompts) ---")
        for name, prompt, maxtok in [("short", SHORT, 120), ("medium", MEDIUM, 400),
                                     ("longform", LONGFORM, 2048)]:
            tps_list, ttft_list = [], []
            for i in range(args.runs):
                m = stream_call(args.host, args.port, args.model,
                                [{"role": "user", "content": prompt}], maxtok,
                                args.temp, args.thinking_off_kwargs, **EP)
                tps_list.append(m["gen_tps"])
                ttft_list.append(m["ttft_ms"])
            speed_suite[name] = {"tps_median": statistics.median(tps_list),
                                 "ttft_ms_median": statistics.median(ttft_list)}
            print(f"  {name}: {speed_suite[name]['tps_median']:.1f} t/s  "
                  f"ttft {speed_suite[name]['ttft_ms_median']:.0f}ms")

    comp = composite_score(scenario_scores)
    print(f"\n═══ COMPOSITE (v2) ═══ recall={comp['recall']}  yield={comp['yield']}"
          f"  discipline={comp['discipline']}  consistency={comp['consistency']}"
          f"  SCORE={comp['score']}/100")

    result = {"fingerprint": fp, "composite": comp,
              "scenarios": scenario_scores, "speed_suite": speed_suite}

    if args.save_baseline:
        os.makedirs(BASELINES_DIR, exist_ok=True)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        stamp = time.strftime("%Y%m%d")
        bpath = os.path.join(BASELINES_DIR, f"{args.save_baseline}_{stamp}.json")
        rpath = os.path.join(RESULTS_DIR, f"{args.save_baseline}_{stamp}_raw.json")
        result["raw_runs"] = raw
        with open(bpath, "w") as f:
            json.dump(result, f, indent=1, ensure_ascii=False)
        print(f"baseline: {bpath}")
        print(f"raw runs: {rpath}")
    elif raw:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        rpath = os.path.join(RESULTS_DIR, f"run_{time.strftime('%Y%m%d_%H%M%S')}.json")
        with open(rpath, "w") as f:
            json.dump({**result, "raw_runs": raw}, f, indent=1, ensure_ascii=False)
        print(f"raw runs: {rpath}")

    if args.compare:
        with open(args.compare) as f:
            base = json.load(f)
        if base["fingerprint"].get("dataset_version") != ds_version or \
                base["fingerprint"].get("bench_version", "1.0.0") != "2.0.0":
            print(f"\n⚠ REFUSING comparison: baseline is dataset "
                  f"v{base['fingerprint'].get('dataset_version')} / bench "
                  f"v{base['fingerprint'].get('bench_version', '1.0.0')} — "
                  f"this run is dataset v{ds_version} / bench 2.0.0. Scores are not "
                  f"cross-version comparable. Re-baseline the incumbent under the new "
                  f"dataset version first (new baseline: this run's --save-baseline).")
        else:
            if base["fingerprint"].get("response_format") != fp["response_format"]:
                print(f"\n⚠ GRAMMAR MISMATCH: baseline was '{base['fingerprint'].get('response_format')}' "
                      f"but this run is '{fp['response_format']}'. json_ok/speed are NOT "
                      f"apples-to-apples across grammar modes — content scores are still comparable.")
            print(f"\n═══ COMPARISON vs {os.path.basename(args.compare)} ═══")
            print(f"baseline: {base['fingerprint']['model']} @ {base['fingerprint']['host']}"
                  f" ({base['fingerprint'].get('date', '?')})")
            print(f"{'scenario':<22}{'recall':>14}{'yield':>14}{'viol':>10}{'consist':>12}")
            all_ids = sorted(set(base["scenarios"]) | set(scenario_scores))
            for sid in all_ids:
                b = base["scenarios"].get(sid)
                n = scenario_scores.get(sid)
                if not (b and n):
                    print(f"{sid:<22}{'(missing side)':>50}")
                    continue
                dr = (n["recall_mean"] or 0) - (b["recall_mean"] or 0)
                dy = n["yield_mean"] - b["yield_mean"]
                dv = n["violations_total"] - b["violations_total"]
                dc = n["consistency"] - b["consistency"]
                print(f"{sid:<22}{dr:>+12.2f}  {dy:>+12.2f}  {dv:>+8}  {dc:>+10.2f}")
            db = comp["score"] - base["composite"]["score"]
            print(f"\ncomposite: {base['composite']['score']} → {comp['score']} ({db:+.2f})")


if __name__ == "__main__":
    main()
