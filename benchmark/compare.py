"""
Comparative accuracy + speed benchmark across four intent engines.

Engines
-------
padaos      – regex-based matcher (same family as padacioso, no fuzz)
padacioso   – regex-based matcher, fuzz=False
padacioso   – regex-based matcher, fuzz=True
padatious   – neural-network matcher (requires training pass)
nebulento   – fuzzy string matching engine

All engines use the same training templates and are evaluated on the same
natural-language test utterances (contractions, idioms, indirect phrasing).

Usage
-----
    uv run python benchmark/compare.py
"""
import time
import tempfile
import statistics
import logging
from collections import defaultdict

from benchmark.dataset import INTENTS, NO_MATCH_UTTERANCES

logging.disable(logging.CRITICAL)

try:
    from padacioso.bracket_expansion import expand_parentheses, normalize_example, normalize_utterance
except ImportError:
    import re as _re
    def normalize_utterance(s): return _re.sub(r"\s+", " ", _re.sub(r"[''ʼ]", "", s)).strip().lower()
    def normalize_example(s): return _re.sub(r"\s+", " ", _re.sub(r"[''ʼ]", "", s)).strip()
    def expand_parentheses(s): return [s]


# ── shared helpers ─────────────────────────────────────────────────────────

def all_cases():
    cases = []
    for name, data in INTENTS.items():
        for utt in data["test_match"]:
            cases.append((utt, name))
    for utt in NO_MATCH_UTTERANCES:
        cases.append((utt, None))
    return cases


def compute_metrics(results, cases):
    total     = len(cases)
    match_n   = sum(1 for _, e in cases if e is not None)
    nomatch_n = total - match_n
    tp = fp = fn = tn = 0
    per_tp = defaultdict(int)
    per_fn = defaultdict(int)
    per_fp = defaultdict(int)
    wrong  = []
    for (predicted, conf), (utt, expected) in zip(results, cases):
        if expected is not None:
            if predicted == expected:
                tp += 1; per_tp[expected] += 1
            else:
                fn += 1; per_fn[expected] += 1
                wrong.append((utt, expected, predicted, conf))
        else:
            if predicted is not None:
                fp += 1; per_fp[predicted] += 1
                wrong.append((utt, expected, predicted, conf))
            else:
                tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / match_n   if match_n   else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return dict(
        accuracy=(tp + tn) / total,
        precision=precision, recall=recall, f1=f1,
        fp=fp, fn=fn, match_n=match_n, nomatch_n=nomatch_n,
        per_tp=per_tp, per_fn=per_fn, per_fp=per_fp, wrong=wrong,
    )


import sys as _sys
_CI_MODE = "--ci" in _sys.argv


def _stats_lines(label, metrics, latencies, train_ms=None):
    s = sorted(latencies)
    total = metrics['match_n'] + metrics['nomatch_n']
    lines = []
    lines.append(f"{'='*64}")
    lines.append(f"  {label}")
    lines.append(f"{'='*64}")
    if train_ms is not None:
        lines.append(f"  Train time: {train_ms:.0f} ms")
    lines.append(f"  Accuracy  : {metrics['accuracy']:.1%}  ({int(metrics['accuracy']*total)}/{total})")
    lines.append(f"  Precision : {metrics['precision']:.1%}")
    lines.append(f"  Recall    : {metrics['recall']:.1%}")
    lines.append(f"  F1        : {metrics['f1']:.3f}")
    lines.append(f"  FP        : {metrics['fp']} / {metrics['nomatch_n']}  ({metrics['fp']/metrics['nomatch_n']:.0%} of no-match)")
    lines.append(f"  FN        : {metrics['fn']} / {metrics['match_n']}  ({metrics['fn']/metrics['match_n']:.0%} of match)")
    lines.append(f"  Latency   : median={statistics.median(latencies):.2f}ms  p95={s[int(len(s)*.95)]:.2f}ms  max={s[-1]:.2f}ms")
    issues = sorted(set(metrics['per_fn']) | set(metrics['per_fp']))
    if issues:
        lines.append("")
        lines.append("  Per-intent (issues only):")
        for i in sorted(INTENTS):
            fn = metrics['per_fn'].get(i, 0)
            fp = metrics['per_fp'].get(i, 0)
            tp = metrics['per_tp'].get(i, 0)
            if fn or fp:
                rec = tp / (tp + fn) if (tp + fn) else 0
                lines.append(f"    {i:<24}  recall={rec:.0%}  fn={fn}  fp={fp}")
    if metrics['wrong']:
        lines.append("")
        lines.append(f"  Mismatches ({len(metrics['wrong'])}):")
        for utt, exp, pred, conf in metrics['wrong']:
            lines.append(f"    [{exp or chr(8212)} -> {pred or chr(8212)}] ({conf:.2f})  \"{utt}\"")
    return lines


def print_report(label, metrics, latencies, train_ms=None):
    lines = _stats_lines(label, metrics, latencies, train_ms)
    if _CI_MODE:
        acc = metrics['accuracy']
        fp  = metrics['fp']
        med = statistics.median(latencies)
        print(f"<details>")
        print(f"<summary><b>{label}</b> &mdash; acc {acc:.1%} &middot; FP {fp} &middot; median {med:.2f}ms</summary>")
        print()
        print("```text")
        for line in lines:
            print(line)
        print("```")
        print()
        print("</details>")
        print()
    else:
        for line in lines:
            print(line)


# ── engine runners ─────────────────────────────────────────────────────────

def run_padaos(cases):
    import padaos
    c = padaos.IntentContainer()
    for name, data in INTENTS.items():
        # padaos uses the same template syntax as padacioso
        c.add_intent(name, data["train"])
    t0 = time.perf_counter()
    c.compile()
    train_ms = (time.perf_counter() - t0) * 1000

    results, latencies = [], []
    for utt, _ in cases:
        q = normalize_utterance(utt)
        t0 = time.perf_counter()
        r  = c.calc_intent(q)
        latencies.append((time.perf_counter() - t0) * 1000)
        results.append((r.get("name"), 1.0 if r.get("name") else 0.0))

    m = compute_metrics(results, cases)
    print_report("padaos  (regex, no fuzz)", m, latencies, train_ms)
    return m, statistics.median(latencies), statistics.mean(latencies), train_ms


def run_padacioso(cases, fuzz):
    from padacioso import IntentContainer
    c = IntentContainer(fuzz=fuzz)
    for name, data in INTENTS.items():
        c.add_intent(name, data["train"])

    results, latencies = [], []
    for utt, _ in cases:
        t0 = time.perf_counter()
        r  = c.calc_intent(utt)
        latencies.append((time.perf_counter() - t0) * 1000)
        results.append((r.get("name") if r else None, r.get("conf", 0.0) if r else 0.0))

    label = f"padacioso  fuzz={'True ' if fuzz else 'False'}"
    m = compute_metrics(results, cases)
    print_report(label, m, latencies)
    return m, statistics.median(latencies), statistics.mean(latencies), None


def run_padatious(cases, threshold=0.5):
    from padatious import IntentContainer as PC
    with tempfile.TemporaryDirectory() as d:
        c = PC(cache_dir=d)
        for name, data in INTENTS.items():
            c.add_intent(name, data["train"])
        t0 = time.perf_counter()
        c.train(single_thread=True, debug=False)
        train_ms = (time.perf_counter() - t0) * 1000

        results, latencies = [], []
        for utt, _ in cases:
            t0 = time.perf_counter()
            r  = c.calc_intent(normalize_utterance(utt))
            latencies.append((time.perf_counter() - t0) * 1000)
            predicted = r.name if (r and r.conf >= threshold) else None
            results.append((predicted, r.conf if r else 0.0))

    m = compute_metrics(results, cases)
    print_report(f"padatious  (neural, threshold={threshold})", m, latencies, train_ms)
    return m, statistics.median(latencies), statistics.mean(latencies), train_ms


def run_nebulento(cases, strategy_name, threshold=0.5):
    from nebulento import IntentContainer
    from nebulento.fuzz import MatchStrategy
    strategy = getattr(MatchStrategy, strategy_name)
    c = IntentContainer(fuzzy_strategy=strategy)
    for name, data in INTENTS.items():
        c.add_intent(name, data["train"])

    results, latencies = [], []
    for utt, _ in cases:
        t0 = time.perf_counter()
        r  = c.calc_intent(utt)
        latencies.append((time.perf_counter() - t0) * 1000)
        predicted = r.get("name") if (r and r.get("conf", 0) >= threshold) else None
        results.append((predicted, r.get("conf", 0.0) if r else 0.0))

    m = compute_metrics(results, cases)
    label = f"nebulento  {strategy_name.lower().replace('_', '-')}"
    print_report(label, m, latencies)
    return m, statistics.median(latencies), statistics.mean(latencies), None


# ── summary table ──────────────────────────────────────────────────────────

def summary(rows):
    """rows: list of (label, metrics, median_lat_ms, mean_lat_ms, train_ms_or_None)"""
    if _CI_MODE:
        print("## Summary\n")
        print("| Engine | Acc | Prec | Recall | F1 | FP | Median |")
        print("|---|---|---|---|---|---|---|")
        for label, m, median_lat, mean_lat, _ in rows:
            print(f"| {label} | {m['accuracy']:.1%} | {m['precision']:.1%} | {m['recall']:.1%} | {m['f1']:.3f} | {m['fp']} | {median_lat:.2f}ms |")
        print()
        print("_FP = false positives on no-match_")
    else:
        print(f"\n\n{'─'*84}")
        print(f"  {'Engine':<36} {'Acc':>6} {'Prec':>6} {'Recall':>7} {'F1':>6}  {'FP':>4}  {'Median':>8}  {'Mean':>8}")
        print(f"{'─'*84}")
        for label, m, median_lat, mean_lat, train_ms in rows:
            print(f"  {label:<36} {m['accuracy']:>5.1%} {m['precision']:>5.1%} "
                  f"{m['recall']:>6.1%} {m['f1']:>5.3f}  {m['fp']:>4}  {median_lat:>6.2f}ms  {mean_lat:>6.2f}ms")
        print(f"{'─'*84}")
        print(f"  FP = false positives on no-match | Median/Mean = query latency in ms")


# ── main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cases   = all_cases()
    match_n = sum(1 for _, e in cases if e is not None)
    print(f"\nDataset : {len(cases)} cases  ({match_n} match, {len(cases)-match_n} no-match)")
    print(f"Intents : {len(INTENTS)}")
    print(f"Note    : test utterances are natural human phrasing, NOT template fills.")

    rows = []
    m, lat, mean_lat, tr = run_padaos(cases)
    rows.append(("padaos  (regex)", m, lat, mean_lat, tr))

    m, lat, mean_lat, tr = run_padacioso(cases, fuzz=False)
    rows.append(("padacioso  fuzz=False", m, lat, mean_lat, tr))

    m, lat, mean_lat, tr = run_padacioso(cases, fuzz=True)
    rows.append(("padacioso  fuzz=True", m, lat, mean_lat, tr))

    m, lat, mean_lat, tr = run_padatious(cases, threshold=0.5)
    rows.append(("padatious  neural  threshold=0.5", m, lat, mean_lat, tr))

    from nebulento.fuzz import MatchStrategy
    for strategy in MatchStrategy:
        m, lat, mean_lat, tr = run_nebulento(cases, strategy_name=strategy.name, threshold=0.5)
        rows.append((f"nebulento  {strategy.name.lower().replace('_', '-')}", m, lat, mean_lat, tr))

    summary(rows)
