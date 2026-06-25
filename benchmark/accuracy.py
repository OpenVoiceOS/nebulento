"""
Accuracy benchmark for nebulento.

Runs every labelled utterance through IntentContainer and reports:
  - Per-intent precision / recall
  - Overall accuracy, false-positive rate, avg confidence
  - Confusion matrix for mismatches
  - Speed: median query latency

Usage:
    python benchmark/accuracy.py [--strategy TOKEN_SET_RATIO] [--dataset intents-for-eval]
"""
import argparse
import statistics
import time
from collections import defaultdict

from nebulento import IntentContainer
from nebulento.fuzz import MatchStrategy
from benchmark.dataset import DATASETS, load


def build_container(bundle, strategy: str = "TOKEN_SET_RATIO") -> IntentContainer:
    fuzzy_strategy = getattr(MatchStrategy, strategy)
    c = IntentContainer(fuzzy_strategy=fuzzy_strategy)
    for entity_name, samples in bundle.entities.items():
        c.add_entity(entity_name, samples)
    for intent_name, data in bundle.intents.items():
        c.add_intent(intent_name, data["train"])
    return c


def run(strategy: str = "TOKEN_SET_RATIO", dataset: str = "intents-for-eval"):
    bundle = load(dataset)
    container = build_container(bundle, strategy)

    # ── collect test cases ─────────────────────────────────────────────────
    # (utterance, expected_intent_or_None)
    cases = []
    for intent_name, data in bundle.intents.items():
        for utt in data["test_match"]:
            cases.append((utt, intent_name))
    for utt in bundle.no_match:
        cases.append((utt, None))

    total = len(cases)
    match_cases = sum(1 for _, e in cases if e is not None)
    nomatch_cases = total - match_cases

    # ── run ────────────────────────────────────────────────────────────────
    results = []
    latencies = []
    for utt, expected in cases:
        t0 = time.perf_counter()
        r = container.calc_intent(utt)
        latencies.append((time.perf_counter() - t0) * 1000)
        predicted = r.get("name") if r else None
        conf = r.get("conf", 0.0) if r else 0.0
        results.append((utt, expected, predicted, conf))

    # ── aggregate ──────────────────────────────────────────────────────────
    correct = sum(1 for _, e, p, _ in results if e == p)
    true_pos  = sum(1 for _, e, p, _ in results if e is not None and e == p)
    false_neg = sum(1 for _, e, p, _ in results if e is not None and p != e)
    false_pos = sum(1 for _, e, p, _ in results if e is None and p is not None)

    accuracy   = correct / total if total else 0
    precision  = true_pos / (true_pos + false_pos) if (true_pos + false_pos) else 0
    recall     = true_pos / match_cases if match_cases else 0
    f1         = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    fp_rate    = false_pos / nomatch_cases if nomatch_cases else 0

    # ── per-intent ─────────────────────────────────────────────────────────
    per_intent_tp = defaultdict(int)
    per_intent_fn = defaultdict(int)
    per_intent_fp = defaultdict(int)
    wrong_predictions = []  # (utt, expected, predicted, conf)

    for utt, expected, predicted, conf in results:
        if expected is not None:
            if predicted == expected:
                per_intent_tp[expected] += 1
            else:
                per_intent_fn[expected] += 1
                wrong_predictions.append((utt, expected, predicted, conf))
        else:
            if predicted is not None:
                per_intent_fp[predicted] += 1
                wrong_predictions.append((utt, expected, predicted, conf))

    # ── print report ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Accuracy benchmark  (dataset={dataset}, strategy={strategy})")
    print(f"{'='*60}")
    print(f"  Total cases     : {total}  ({match_cases} match, {nomatch_cases} no-match)")
    print(f"  Correct         : {correct}/{total}  ({accuracy:.1%})")
    print(f"  Precision       : {precision:.1%}")
    print(f"  Recall          : {recall:.1%}")
    print(f"  F1              : {f1:.3f}")
    print(f"  False positives : {false_pos}  ({fp_rate:.1%} of no-match cases)")
    fn_rate = false_neg / match_cases if match_cases else 0
    print(f"  False negatives : {false_neg}  ({fn_rate:.1%} of match cases)")

    lat_sorted = sorted(latencies)
    print(f"\n  Latency  median={statistics.median(latencies):.2f}ms  "
          f"p95={lat_sorted[int(len(lat_sorted)*.95)]:.2f}ms  "
          f"max={lat_sorted[-1]:.2f}ms")

    # per-intent recall table
    print(f"\n  {'Intent':<22} {'TP':>4} {'FN':>4} {'Recall':>8}  {'FP':>4}")
    print(f"  {'-'*48}")
    all_intents = sorted(bundle.intents.keys())
    for name in all_intents:
        tp = per_intent_tp[name]
        fn = per_intent_fn[name]
        fp = per_intent_fp[name]
        rec = tp / (tp + fn) if (tp + fn) else 0
        flag = " !" if rec < 1.0 or fp > 0 else ""
        print(f"  {name:<22} {tp:>4} {fn:>4} {rec:>7.0%}  {fp:>4}{flag}")

    if wrong_predictions:
        print(f"\n  Mismatches ({len(wrong_predictions)}):")
        for utt, expected, predicted, conf in wrong_predictions:
            exp_str = expected or "—"
            pre_str = predicted or "—"
            print(f"    [{exp_str} → {pre_str}] (conf={conf:.2f})  \"{utt}\"")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="TOKEN_SET_RATIO",
                        help="MatchStrategy name (default: TOKEN_SET_RATIO)")
    parser.add_argument("--dataset", default="intents-for-eval", choices=list(DATASETS),
                        help="benchmark dataset (default: intents-for-eval)")
    args = parser.parse_args()
    run(strategy=args.strategy, dataset=args.dataset)
