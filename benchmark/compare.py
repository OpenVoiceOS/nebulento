"""
Comparative accuracy + speed benchmark across intent engines.

Engines
-------
padaos      – regex-based matcher
padatious   – neural-network matcher (requires training pass)
nebulento   – fuzzy string matching engine (flat, every MatchStrategy, + hierarchical)

Every engine here is a template / sample matcher: it trains on example
sentences, not keyword vocabularies. They are evaluated on two OpenVoiceOS
datasets — ``intents-for-eval`` and ``massive`` — each engine training on the
``<lang>-templates`` config and evaluating on ``<lang>-test``. See
``benchmark/dataset.py``.

Usage
-----
    uv run python benchmark/compare.py
"""
import sys
import time
import tempfile
import statistics
import logging
from collections import defaultdict

from nebulento.bracket_expansion import normalize_utterance

from benchmark.dataset import DATASETS, load

def fbeta(precision, recall, beta=0.5):
    """F_β score. β<1 weights precision over recall — appropriate for voice
    assistants where a false positive (wrong skill fires) is unrecoverable
    while a false negative falls through to fallback handlers (LLM, etc.).
    """
    b2 = beta * beta
    denom = b2 * precision + recall
    return ((1 + b2) * precision * recall / denom) if denom else 0.0


def recall_at_precision(results_no_thresh, cases, p_floor=0.99, step=0.01):
    """Sweep the threshold and return the max recall achievable while
    keeping precision >= ``p_floor`` (and the threshold that gets there).
    """
    best_r, best_t = 0.0, None
    t = 0.0
    while t <= 1.0 + 1e-9:
        thresholded = [
            (lbl if c >= t else None, c) for (lbl, c) in results_no_thresh
        ]
        m = compute_metrics(thresholded, cases)
        if m["precision"] >= p_floor and m["recall"] > best_r:
            best_r, best_t = m["recall"], round(t, 4)
        t += step
    return best_r, best_t


def calibrate_threshold(results_no_thresh, cases, step=0.01, metric="f0.5"):
    """Sweep threshold and return (best_threshold, best_metric_value, best_metrics).

    ``metric`` selects the scalar to maximise: ``"f1"`` or ``"f0.5"``.
    """
    def score(m):
        return m["f1"] if metric == "f1" else fbeta(m["precision"], m["recall"], 0.5)

    best = (0.0, -1.0, None)
    t = 0.0
    while t <= 1.0 + 1e-9:
        thresholded = [
            (lbl if c >= t else None, c) for (lbl, c) in results_no_thresh
        ]
        m = compute_metrics(thresholded, cases)
        s = score(m)
        if s > best[1]:
            best = (round(t, 4), s, m)
        t += step
    return best


logging.disable(logging.CRITICAL)

_CI_MODE = "--ci" in sys.argv


# ── shared helpers ─────────────────────────────────────────────────────────

def all_cases(bundle):
    """Flatten a :class:`~benchmark.dataset.Bundle` into ``(utterance, expected)``."""
    cases = []
    for name, data in bundle.intents.items():
        for utt in data["test_match"]:
            cases.append((utt, name))
    for utt in bundle.no_match:
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
                tp += 1
                per_tp[expected] += 1
            else:
                fn += 1
                per_fn[expected] += 1
                wrong.append((utt, expected, predicted, conf))
        else:
            if predicted is not None:
                fp += 1
                per_fp[predicted] += 1
                wrong.append((utt, expected, predicted, conf))
            else:
                tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / match_n   if match_n   else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return dict(
        accuracy=(tp + tn) / total if total else 0.0,
        precision=precision, recall=recall, f1=f1,
        fp=fp, fn=fn, match_n=match_n, nomatch_n=nomatch_n,
        per_tp=per_tp, per_fn=per_fn, per_fp=per_fp, wrong=wrong,
    )


def _stats_lines(label, metrics, latencies, intents, train_ms=None):
    s = sorted(latencies)
    total = metrics['match_n'] + metrics['nomatch_n']
    nomatch_n = metrics['nomatch_n']
    match_n = metrics['match_n']
    fp_pct = f"  ({metrics['fp']/nomatch_n:.0%} of no-match)" if nomatch_n else ""
    fn_pct = f"  ({metrics['fn']/match_n:.0%} of match)" if match_n else ""
    lines = [
        f"{'='*64}",
        f"  {label}",
        f"{'='*64}",
    ]
    if train_ms is not None:
        lines.append(f"  Train time: {train_ms:.0f} ms")
    lines += [
        f"  Accuracy  : {metrics['accuracy']:.1%}  ({int(metrics['accuracy']*total)}/{total})",
        f"  Precision : {metrics['precision']:.1%}",
        f"  Recall    : {metrics['recall']:.1%}",
        f"  F1        : {metrics['f1']:.3f}",
        f"  FP        : {metrics['fp']} / {nomatch_n}{fp_pct}",
        f"  FN        : {metrics['fn']} / {match_n}{fn_pct}",
        f"  Latency   : median={statistics.median(latencies):.2f}ms  "
        f"p95={s[int(len(s)*.95)]:.2f}ms  max={s[-1]:.2f}ms",
    ]
    issues = sorted(set(metrics['per_fn']) | set(metrics['per_fp']))
    if issues:
        lines.append("")
        lines.append("  Per-intent (issues only):")
        for i in sorted(intents):
            fn = metrics['per_fn'].get(i, 0)
            fp = metrics['per_fp'].get(i, 0)
            tp = metrics['per_tp'].get(i, 0)
            if fn or fp:
                rec = tp / (tp + fn) if (tp + fn) else 0
                lines.append(f"    {i:<28}  recall={rec:.0%}  fn={fn}  fp={fp}")
    return lines


def print_report(label, metrics, latencies, intents, train_ms=None):
    lines = _stats_lines(label, metrics, latencies, intents, train_ms)
    if _CI_MODE:
        acc = metrics['accuracy']
        fp  = metrics['fp']
        med = statistics.median(latencies)
        print("<details>")
        print(f"<summary><b>{label}</b> &mdash; acc {acc:.1%} &middot; "
              f"FP {fp} &middot; median {med:.2f}ms</summary>")
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

def run_padaos(bundle, cases):
    import padaos
    c = padaos.IntentContainer()
    for entity_name, samples in bundle.entities.items():
        c.add_entity(entity_name, samples)
    for name, data in bundle.intents.items():
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
    print_report("padaos  (regex, no fuzz)", m, latencies, bundle.intents, train_ms)
    return m, statistics.median(latencies), statistics.mean(latencies), train_ms, None


def run_padatious(bundle, cases, threshold=0.5):
    from padatious import IntentContainer as PC
    with tempfile.TemporaryDirectory() as d:
        c = PC(cache_dir=d)
        for entity_name, samples in bundle.entities.items():
            c.add_entity(entity_name, samples)
        for name, data in bundle.intents.items():
            c.add_intent(name, data["train"])
        t0 = time.perf_counter()
        c.train(single_thread=True, debug=False)
        train_ms = (time.perf_counter() - t0) * 1000

        raw, latencies = [], []
        for utt, _ in cases:
            t0 = time.perf_counter()
            r  = c.calc_intent(normalize_utterance(utt))
            latencies.append((time.perf_counter() - t0) * 1000)
            raw.append((r.name if r else None, r.conf if r else 0.0))

    results = [(lbl if c >= threshold else None, c) for (lbl, c) in raw]
    m = compute_metrics(results, cases)
    print_report(f"padatious  (neural, threshold={threshold})", m, latencies,
                 bundle.intents, train_ms)
    return m, statistics.median(latencies), statistics.mean(latencies), train_ms, raw


def run_nebulento(bundle, cases, strategy_name, threshold=0.5):
    from nebulento import IntentContainer
    from nebulento.fuzz import MatchStrategy
    strategy = getattr(MatchStrategy, strategy_name)
    c = IntentContainer(fuzzy_strategy=strategy)
    try:
        for entity_name, samples in bundle.entities.items():
            c.add_entity(entity_name, samples)
        for name, data in bundle.intents.items():
            c.add_intent(name, data["train"])
    except Exception as exc:
        print(f"[SKIP] nebulento  {strategy_name.lower().replace('_', '-')}"
              f"  — registration failed: {exc}")
        return None

    raw, latencies = [], []
    for utt, _ in cases:
        t0 = time.perf_counter()
        r  = c.calc_intent(utt)
        latencies.append((time.perf_counter() - t0) * 1000)
        raw.append((r.get("name") if r else None, r.get("conf", 0.0) if r else 0.0))

    results = [(lbl if c >= threshold else None, c) for (lbl, c) in raw]
    m = compute_metrics(results, cases)
    label = f"nebulento  {strategy_name.lower().replace('_', '-')}"
    print_report(label, m, latencies, bundle.intents)
    return m, statistics.median(latencies), statistics.mean(latencies), None, raw


def run_nebulento_hierarchical(bundle, cases, strategy_name, threshold=0.5,
                               domain_threshold=0.5):
    from nebulento import HierarchicalIntentContainer
    from nebulento.fuzz import MatchStrategy
    strategy = getattr(MatchStrategy, strategy_name)
    c = HierarchicalIntentContainer(fuzzy_strategy=strategy,
                                    domain_threshold=domain_threshold)
    # use the dataset-authoritative ``domain`` field on each intent; parsing
    # ``intent_id`` with a hard-coded separator is fragile (datasets vary
    # between ``:`` and ``.``).
    try:
        for name, data in bundle.intents.items():
            c.register_domain_intent(data["domain"], name, data["train"])
        # register each entity in every domain whose intents reference it
        domain_entities = defaultdict(set)
        for name, data in bundle.intents.items():
            for entity_name in data["entities"]:
                domain_entities[data["domain"]].add(entity_name)
        for dom, entity_names in domain_entities.items():
            for entity_name in entity_names:
                c.register_domain_entity(dom, entity_name,
                                         bundle.entities[entity_name])
    except Exception as exc:
        print(f"[SKIP] nebulento-hierarchical  "
              f"{strategy_name.lower().replace('_', '-')}"
              f"  — registration failed: {exc}")
        return None

    raw, latencies = [], []
    for utt, _ in cases:
        t0 = time.perf_counter()
        r  = c.calc_intent(utt)
        latencies.append((time.perf_counter() - t0) * 1000)
        raw.append((r.get("name") if r else None, r.get("conf", 0.0) if r else 0.0))

    results = [(lbl if c >= threshold else None, c) for (lbl, c) in raw]
    m = compute_metrics(results, cases)
    label = f"nebulento-hierarchical  {strategy_name.lower().replace('_', '-')}"
    print_report(label, m, latencies, bundle.intents)
    return m, statistics.median(latencies), statistics.mean(latencies), None, raw


# ── summary table ──────────────────────────────────────────────────────────

def summary(title, rows):
    """rows: list of (label, metrics, median_lat_ms, mean_lat_ms, train_ms_or_None)"""
    if _CI_MODE:
        print(f"## {title}\n")
        print("| Engine | Acc | Prec | Recall | F1 | FP | Median |")
        print("|---|---|---|---|---|---|---|")
        for label, m, median_lat, mean_lat, _ in rows:
            print(f"| {label} | {m['accuracy']:.1%} | {m['precision']:.1%} | "
                  f"{m['recall']:.1%} | {m['f1']:.3f} | {m['fp']} | {median_lat:.2f}ms |")
        print()
        print("_FP = false positives on no-match_")
    else:
        print(f"\n\n{'─'*84}")
        print(f"  {title}")
        print(f"  {'Engine':<36} {'Acc':>6} {'Prec':>6} {'Recall':>7} {'F1':>6}  "
              f"{'FP':>4}  {'Median':>8}  {'Mean':>8}")
        print(f"{'─'*84}")
        for label, m, median_lat, mean_lat, train_ms in rows:
            print(f"  {label:<36} {m['accuracy']:>5.1%} {m['precision']:>5.1%} "
                  f"{m['recall']:>6.1%} {m['f1']:>5.3f}  {m['fp']:>4}  "
                  f"{median_lat:>6.2f}ms  {mean_lat:>6.2f}ms")
        print(f"{'─'*84}")
        print("  FP = false positives on no-match | Median/Mean = query latency in ms")


# ── main ───────────────────────────────────────────────────────────────────

def run_dataset(name):
    from nebulento.fuzz import MatchStrategy

    bundle = load(name)
    cases = all_cases(bundle)
    match_n = sum(1 for _, e in cases if e is not None)
    print(f"\nDataset : {bundle.repo}  ({bundle.lang})")
    print(f"Cases   : {len(cases)}  ({match_n} match, {len(cases)-match_n} no-match)")
    print(f"Intents : {len(bundle.intents)}  across {len(bundle.domains)} domains")
    print("Splits  : " + ", ".join(f"{k}={len(v)}" for k, v in bundle.splits.items()))

    rows = []
    cal_rows = []

    m, lat, mean_lat, tr, raw = run_padaos(bundle, cases)
    rows.append(("padaos  (regex)", m, lat, mean_lat, tr))
    # padaos has no conf knob — skip calibration

    m, lat, mean_lat, tr, raw = run_padatious(bundle, cases, threshold=0.5)
    rows.append(("padatious  neural  threshold=0.5", m, lat, mean_lat, tr))
    cal_rows.append(_calibrate_row("padatious", raw, cases, 0.5, m))

    # every flat MatchStrategy
    for strategy in MatchStrategy:
        result = run_nebulento(bundle, cases,
                               strategy_name=strategy.name, threshold=0.5)
        if result is None:
            continue
        m, lat, mean_lat, tr, raw = result
        label_short = strategy.name.lower().replace('_', '-')
        rows.append((f"nebulento  {label_short}", m, lat, mean_lat, tr))
        cal_rows.append(_calibrate_row(f"neb {label_short}"[:24], raw, cases, 0.5, m))

    # two-stage variant — intents grouped into the dataset's domains.
    # damerau-levenshtein with no gate isolates the cost of misrouting;
    # token-set-ratio with a 0.7 gate shows the off-topic rejection trade-off.
    for strategy, domain_threshold in (
        (MatchStrategy.DAMERAU_LEVENSHTEIN_SIMILARITY, 0.0),
        (MatchStrategy.TOKEN_SET_RATIO, 0.7),
    ):
        result = run_nebulento_hierarchical(
            bundle, cases, strategy_name=strategy.name, threshold=0.5,
            domain_threshold=domain_threshold)
        if result is None:
            continue
        m, lat, mean_lat, tr, raw = result
        label_short = strategy.name.lower().replace('_', '-')
        rows.append((f"nebulento-hierarchical  {label_short}",
                     m, lat, mean_lat, tr))
        cal_rows.append(_calibrate_row(
            f"hier {label_short} g={domain_threshold}"[:24],
            raw, cases, 0.5, m))

    summary(f"{name}  —  {bundle.repo}", rows)
    _print_calibration_table(cal_rows)


def _calibrate_row(label, raw, cases, default_thr, default_metrics):
    """Compute the calibration row for one engine."""
    df1 = default_metrics["f1"]
    dfp = default_metrics["fp"]
    df05 = fbeta(default_metrics["precision"], default_metrics["recall"], 0.5)
    opt_thr, _, opt_metrics = calibrate_threshold(raw, cases, step=0.01,
                                                  metric="f0.5")
    of1 = opt_metrics["f1"]
    ofp = opt_metrics["fp"]
    of05 = fbeta(opt_metrics["precision"], opt_metrics["recall"], 0.5)
    rec_at_p, rec_thr = recall_at_precision(raw, cases, p_floor=0.99, step=0.01)
    return (label, default_thr, df1, df05, dfp,
            opt_thr, of1, of05, ofp, rec_at_p, rec_thr)


def _print_calibration_table(rows):
    print(f"\n{'─'*108}")
    print("  Per-engine threshold calibration  (sweep 0..1 step 0.01, max F_0.5)")
    print(f"  {'Engine':<24} {'def_thr':>7} {'def_F1':>7} {'defF.5':>7} {'def_FP':>6}"
          f"  {'opt_thr':>7} {'opt_F1':>7} {'optF.5':>7} {'opt_FP':>6}"
          f"  {'R@P99':>6} {'thr':>5}")
    print(f"{'─'*108}")
    for r in rows:
        (label, dthr, df1, df05, dfp,
         othr, of1, of05, ofp, rec_at_p, rec_thr) = r
        rec_t = f"{rec_thr:.2f}" if rec_thr is not None else "--"
        print(f"  {label:<24} {dthr:>7.2f} {df1:>7.3f} {df05:>7.3f} {dfp:>6d}"
              f"  {othr:>7.2f} {of1:>7.3f} {of05:>7.3f} {ofp:>6d}"
              f"  {rec_at_p:>6.1%} {rec_t:>5}")
    print(f"{'─'*108}")
    print("  F_0.5 (β=0.5) weights precision 2x recall — the right summary metric")
    print("  for OVOS, where a wrong intent is unrecoverable but a missed intent")
    print("  falls through to fallback handlers. R@P99 = max recall achievable")
    print("  with the threshold tuned to keep precision >= 99%.")


if __name__ == "__main__":
    selected = [a for a in sys.argv[1:] if not a.startswith("-")]
    targets = selected or list(DATASETS)
    for dataset_name in targets:
        if dataset_name not in DATASETS:
            print(f"unknown dataset {dataset_name!r}; choose from {list(DATASETS)}")
            continue
        run_dataset(dataset_name)
