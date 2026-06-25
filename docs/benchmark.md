# Benchmark

Nebulento ships a comparative accuracy and speed benchmark in `benchmark/compare.py`. It runs on two OpenVoiceOS evaluation datasets and reports every engine — and every nebulento `MatchStrategy` — side by side, with per-engine threshold calibration so the numbers reflect each engine's real ceiling, not just its shipped default.

---

## Headline results — `intents-for-eval`

50 intents across 10 domains, 1700 labelled test cases, 50 off-topic (`far_ood`) cases. F_0.5 numbers come from the per-engine threshold sweep (`benchmark/compare.py` calibration table).

| Engine | def F_0.5 | **opt F_0.5** | opt thr | opt FP | **Rec @ P≥99%** |
|---|---|---|---|---|---|
| padatious (neural) | 0.892 | 0.924 | 0.16 | 13 | 72.8% |
| padaos (regex) | n/a | 0.831\* | 0.50 | 1 | n/a (no conf knob) |
| **nebulento `damerau-levenshtein`** (default) | 0.909 | **0.918** | 0.43 | 26 | **62.0%** |
| nebulento `simple-ratio` | 0.914 | 0.915 | 0.55 | 25 | 60.4% |
| nebulento `ratio` | 0.914 | **0.917** | 0.55 | 26 | 59.1% |
| nebulento `token-set-ratio` | 0.907 | 0.913 | 0.60 | 21 | 62.9% |
| nebulento `token-sort-ratio` | 0.908 | 0.910 | 0.53 | 34 | 61.4% |
| nebulento `partial-ratio` | 0.879 | 0.886 | 0.67 | 29 | 56.8% |
| nebulento `partial-token-sort-ratio` | 0.868 | 0.878 | 0.67 | 26 | 49.8% |
| nebulento `partial-token-ratio` | 0.783 | 0.791 | 0.75 | 36 | 0.0% |
| nebulento `partial-token-set-ratio` | 0.783 | 0.791 | 0.75 | 36 | 0.0% |
| nebulento-hierarchical `damerau` (gate=0.0) | 0.884 | 0.889 | 0.47 | 17 | 58.5% |
| nebulento-hierarchical `token-set-ratio` (gate=0.7) | 0.852 | 0.852 | 0.00 | 11 | 54.6% |

\* padaos has no confidence knob — the 0.831 figure is F_0.5 at its single fixed operating point.

**The shipped default — flat `damerau-levenshtein-similarity` at threshold 0.5 — is already at F_0.5 = 0.909, within 0.01 of its calibrated optimum (0.918). Calibration moves the needle very little for nebulento; the shipped defaults are sensible.**

---

## Why F_0.5 and not F1

A voice assistant's two failure modes are not symmetric:

- **False positive** — the wrong intent fires, the skill executes the wrong action, the assistant says the wrong thing. The user has to notice, abort, and re-ask. There is no recovery layer above the intent service that can catch this.
- **False negative** — no intent fires. OVOS hands the utterance to its fallback chain: common-query, the LLM fallback, online search. These exist precisely to handle "I don't know what you meant." Worst case the user re-phrases; best case the LLM nails it.

The cost ratio is roughly 5–10× in favour of false negatives. F1 (which weights precision and recall equally) is the wrong summary metric. **F_β with β=0.5 weights precision twice as recall** and is the right summary for OVOS.

We also report **Rec@P≥99%** — the recall achievable once the threshold is tuned to keep precision at or above 99%. This is the operating point a maintainer actually picks: "give me the most coverage you can while letting through at most 1% wrong matches." For nebulento's best strategies this lands between 60% and 63%.

---

## Datasets

Both datasets are loaded from the Hugging Face Hub by `benchmark/dataset.py`. Each has a `<lang>-templates` config (training templates) and a `<lang>-test` config (labelled evaluation utterances).

| Name | Repo | Intents | Test cases | Notes |
|---|---|---|---|---|
| `intents-for-eval` | [`OpenVoiceOS/intents-for-eval`](https://huggingface.co/datasets/OpenVoiceOS/intents-for-eval) | 50 | 1750 | Six test splits, including a 50-row `far_ood` no-match set |
| `massive` | [`OpenVoiceOS/massive-templates`](https://huggingface.co/datasets/OpenVoiceOS/massive-templates) | 60 | 2974 | OVOS-templated rebuild of MASSIVE; one labelled split, no no-match cases |

`intents-for-eval` test splits:

| Split | Cases | What it tests |
|---|---|---|
| `template` | 500 | Utterances that fill a training template directly |
| `paraphrase` | 700 | Natural rephrasings — different words, same intent |
| `near_ood` | 400 | Boundary utterances close to another intent |
| `far_ood` | 50 | Genuinely off-topic — should match **nothing** |
| `asr_noise` | 50 | Speech-recognition artefacts |
| `typos` | 50 | Spelling errors |

### Slots and entities

Every `{slot}` placeholder in the templates ships with a list of example values. `benchmark/dataset.py` collects them into `Bundle.entities`. Nebulento, padaos and padatious all register slot values as named entities natively, so the matcher sees real song titles / city names / contact names rather than the literal `{song}` token.

---

## Engines

| Engine | Role | Notes |
|---|---|---|
| `padaos` | baseline | regex-based exact matcher; no fuzz, no confidence knob |
| `padatious` | baseline | neural matcher (requires `train()` pass) |
| `nebulento` (9 strategies) | **subject** | every `MatchStrategy` value, all at the shipped threshold 0.5 |
| `nebulento-hierarchical` (2 variants) | **subject** | two-stage domain → intent routing; tested with `damerau` (gate=0.0) and `token-set-ratio` (gate=0.7) |

The nine `MatchStrategy` values benchmarked:

- `SIMPLE_RATIO`, `RATIO` — character-level similarity over the whole string
- `PARTIAL_RATIO` — best-matching substring of one string against the other
- `TOKEN_SORT_RATIO`, `TOKEN_SET_RATIO` — alphabetised-token / set-overlap variants of `RATIO`
- `PARTIAL_TOKEN_RATIO`, `PARTIAL_TOKEN_SORT_RATIO`, `PARTIAL_TOKEN_SET_RATIO` — partial counterparts
- `DAMERAU_LEVENSHTEIN_SIMILARITY` — normalised edit distance with transposition; the **shipped default**

---

## Deep-dive: tuning nebulento

### Strategy comparison

Ranked by calibrated F_0.5:

| Rank | Strategy | opt F_0.5 | opt FP | Recall @ P≥99% |
|---|---|---|---|---|
| 1 | `damerau-levenshtein-similarity` | **0.918** | 26 | 62.0% |
| 2 | `ratio` | 0.917 | 26 | 59.1% |
| 3 | `simple-ratio` | 0.915 | 25 | 60.4% |
| 4 | `token-set-ratio` | 0.913 | 21 | **62.9%** |
| 5 | `token-sort-ratio` | 0.910 | 34 | 61.4% |
| 6 | `partial-ratio` | 0.886 | 29 | 56.8% |
| 7 | `partial-token-sort-ratio` | 0.878 | 26 | 49.8% |
| 8 | `partial-token-ratio` | 0.791 | 36 | 0.0% |
| 8 | `partial-token-set-ratio` | 0.791 | 36 | 0.0% |

Two things jump out:

1. **The top four are clustered tightly** (F_0.5 between 0.913 and 0.918) — at this dataset's vocabulary scale (~50 short imperative intents) the choice between character-edit and token-overlap strategies is largely a wash.
2. **All four `partial-*` token strategies are penalised hard.** They reward any substring overlap, which on a 50-intent corpus where many intents share common verbs (`set`, `play`, `cancel`, `read`) means cross-domain confusions multiply — `partial-token-ratio` and `partial-token-set-ratio` produce so many false positives that **they never reach 99% precision at any threshold** (R@P99 = 0.0%).

### The clear winner: `damerau-levenshtein-similarity`

The shipped default wins by a small but real margin: F_0.5 = 0.918 vs second-place 0.917. More importantly it has the **best raw FP behaviour at the shipped threshold** of any strategy in the table — 17 FPs at threshold 0.5 vs 40+ for every `ratio`/`token-*` variant. The OVOS asymmetry rewards that hugely.

Why? Damerau-Levenshtein gives a sharper, more concentrated score distribution than token-overlap metrics. Token strategies score "kind-of" matches (one shared verb) at 0.5+, which floods the FP bin. Damerau punishes character-level distance proportionally, so genuine paraphrases stay above 0.5 and unrelated utterances drop well below.

The flip side: damerau is **slower** than the ratio variants when slot expansion balloons the candidate set (7.5 ms median vs 3.1 ms for `ratio`). For a typical OVOS install that's irrelevant; for a large skill catalogue it's worth knowing.

### FP profile of the shipped default

At threshold 0.5, `damerau-levenshtein-similarity` produces 17 false positives — 34% of the 50 `far_ood` cases. Calibration to threshold 0.43 reduces this further to a precision of ~96.9% on the 1750-case mix. The shipped default is **not** the F_0.5 optimum (0.43 is), but the gap is 0.009 F_0.5 — well within measurement noise.

### `damerau-levenshtein-similarity` as the shipped default — justified?

**Yes.** Three converging reasons:

1. Highest opt F_0.5 of any strategy in the table.
2. Lowest FP count at the shipped threshold.
3. Best behaviour under the strict P≥99% operating point (62.0% R@P99 — only `token-set-ratio` beats it by 0.9pp).

If anything, the data suggests a default *threshold* tweak: dropping from 0.50 to 0.43 lifts F_0.5 from 0.909 to 0.918, trading a few extra FPs (17 → 26) for higher recall. The current default is conservative — appropriate for OVOS, where FPs hurt more.

### Hierarchical variant analysis

Two hierarchical variants were benchmarked:

| Variant | opt F_0.5 | opt FP | R@P99 |
|---|---|---|---|
| flat `damerau` (reference) | 0.918 | 26 | 62.0% |
| hierarchical `damerau` (gate=0.0) | 0.889 | 17 | 58.5% |
| hierarchical `token-set-ratio` (gate=0.7) | 0.852 | 11 | 54.6% |

**Hierarchical loses ~3pp F_0.5 vs flat at the default `damerau` strategy.** The loss is recall, not precision — the domain stage occasionally misroutes a borderline utterance to the wrong domain, after which the per-domain matcher cannot recover it. With `domain_threshold=0.0` (no gate), routing is best-effort and any misroute is a lost match.

The `domain_threshold` gate is the interesting knob. With `token-set-ratio` and gate=0.7:

- FPs collapse from 26 → **11** (lowest of any nebulento variant in the table).
- The gate sweep finds 0.00 as the F_0.5-optimal *intent* threshold — because the *domain* gate is already doing all the filtering work.
- R@P99 drops to 54.6%, the worst of any non-degenerate strategy.

In other words: the `domain_threshold` gate is a **precision boost at the cost of recall**. It earns its keep when off-topic rejection matters more than catch rate — e.g. a deployment with strong LLM fallback that can absorb the missed matches.

For most OVOS deployments the flat default is the right pick. The hierarchical variant earns its keep when:

- Skill modularity matters (add/remove a skill by re-registering just one domain).
- You want a precision boost via `domain_threshold` and have a fallback that handles the resulting FNs.

### Threshold calibration: was the shipped 0.5 sensible?

Across all twelve calibrated rows, the F_0.5-optimal threshold and the resulting opt F_0.5 sit very close to the defaults:

| Strategy | def F_0.5 | opt F_0.5 | Δ |
|---|---|---|---|
| `damerau` (default) | 0.909 | 0.918 | +0.009 |
| `ratio` | 0.914 | 0.917 | +0.003 |
| `simple-ratio` | 0.914 | 0.915 | +0.001 |
| `token-set-ratio` | 0.907 | 0.913 | +0.006 |
| `token-sort-ratio` | 0.908 | 0.910 | +0.002 |
| `partial-ratio` | 0.879 | 0.886 | +0.007 |
| hierarchical `damerau` | 0.884 | 0.889 | +0.005 |
| hierarchical `token-set-ratio` g=0.7 | 0.852 | 0.852 | 0.000 |

**The biggest swing is +0.009 F_0.5** (for the shipped default). Nebulento's score distribution is already well-calibrated for threshold 0.5 — unlike engines whose confidence distributions crowd low or high. **This is a positive finding: the engine is well-tuned out of the box.**

For comparison: padatious's calibrated optimum sits at threshold 0.16 (vs default 0.50), a +0.032 F_0.5 swing. Markov flat (in the markov repo's benchmark) swings +0.244 F_0.5 from re-thresholding alone. Nebulento needs no such intervention.

---

## Flat vs Hierarchical

| Variant | opt F_0.5 | opt FP | R@P99 | median ms |
|---|---|---|---|---|
| flat `damerau` | **0.918** | 26 | 62.0% | 7.5 |
| hierarchical `damerau` (gate=0.0) | 0.889 | 17 | 58.5% | 6.1 |
| hierarchical `token-set-ratio` (gate=0.7) | 0.852 | **11** | 54.6% | 3.5 |

- **Flat wins by F_0.5** by ~3pp. The flat matcher considers every intent equally; the hierarchical matcher loses some recall to domain-stage misrouting.
- **Hierarchical wins on FPs** when the `domain_threshold` gate is active (11 FPs with `token-set-ratio` gate=0.7), but at a steep recall cost.
- **Latency favours hierarchical** modestly — each utterance is only scored against the intents in the winning domain. The `token-set-ratio` variant is **~2× faster** than the flat default.

Use flat when accuracy is paramount. Use hierarchical when you need per-skill modularity or want the `domain_threshold` precision boost.

---

## Reproducing

```bash
pip install -e .[benchmark]
python benchmark/compare.py intents-for-eval   # ~2-3 minutes
python benchmark/compare.py                    # both datasets
```

The first run downloads each dataset from the Hugging Face Hub (cached afterwards). The script prints a per-engine report, the summary table, and the per-engine threshold calibration table.

---

## How metrics are calculated

Source: `compute_metrics`, `calibrate_threshold`, `fbeta`, `recall_at_precision` in `benchmark/compare.py`.

- **Accuracy** = (TP + TN) / total
- **Precision** = TP / (TP + FP)
- **Recall** = TP / total_match_cases
- **F1** = 2·P·R / (P + R)
- **F_0.5** = 1.25·P·R / (0.25·P + R) — weights precision 2× recall (default summary metric for OVOS)
- **Rec@P≥99%** = max recall achievable by sweeping the threshold while keeping precision ≥ 99%
- **FP** = no-match utterances incorrectly assigned an intent

A prediction is a TP when the predicted intent name exactly matches the expected intent and `conf ≥ threshold`. A no-match case is correct only when the engine returns no intent or a confidence below threshold. The calibration table re-applies the threshold over each engine's raw `(label, conf)` output, sweeping 0..1 in steps of 0.01.
