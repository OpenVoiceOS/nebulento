# Benchmark

Nebulento includes a comparative accuracy and speed benchmark in `benchmark/compare.py`. It runs on two OpenVoiceOS evaluation datasets and reports every engine — and every nebulento `MatchStrategy` — side by side.

---

## Datasets

Both datasets are loaded from the Hugging Face Hub by `benchmark/dataset.py`. Each has a `<lang>-templates` config (training templates) and a `<lang>-test` config (labelled evaluation utterances). Every engine in this benchmark is a template / sample matcher, so it trains on `-templates` and is evaluated on `-test`.

| Name | Repo | Intents | Test cases | Notes |
|---|---|---|---|---|
| `intents-for-eval` | [`OpenVoiceOS/intents-for-eval`](https://huggingface.co/datasets/OpenVoiceOS/intents-for-eval) | 50 | 1750 | Six test splits, including a `far_ood` no-match set |
| `massive` | [`OpenVoiceOS/massive-templates`](https://huggingface.co/datasets/OpenVoiceOS/massive-templates) | 60 | 2974 | OVOS-templated rebuild of the MASSIVE corpus; one labelled split, no no-match cases |

`intents-for-eval` test splits:

| Split | Cases | What it tests |
|---|---|---|
| `template` | 500 | Utterances that fill a training template directly |
| `paraphrase` | 700 | Natural rephrasings — different words, same intent |
| `near_ood` | 400 | Boundary utterances close to another intent |
| `far_ood` | 50 | Genuinely off-topic — should match **nothing** |
| `asr_noise` | 50 | Speech-recognition artefacts |
| `typos` | 50 | Spelling errors |

`massive` has a single labelled `test` split and **no no-match cases** — so on `massive` every engine has zero false positives by construction, and accuracy equals recall.

### Entities

Each `{slot}` placeholder ships with example values. `benchmark/dataset.py` collects them into an `ENTITIES` map and every engine registers them (the equivalent of a padatious `.entity` file) before matching.

---

## Engines Compared

| Engine | Description |
|---|---|
| `padaos` | Regex-based exact matcher (no fuzzy) |
| `padatious` | Neural network matcher (requires a training pass) |
| `nebulento <strategy>` | Flat `IntentContainer`, one row per `MatchStrategy` value |
| `nebulento-hierarchical <strategy>` | Two-stage `HierarchicalIntentContainer` — intents grouped into the dataset's domains |

---

## Results — `intents-for-eval`

1750 cases (1700 match, 50 no-match), 50 intents across 10 domains.

| Engine | Accuracy | Precision | Recall | F1 | FP / 50 | Median lat |
|---|---|---|---|---|---|---|
| padaos (regex) | 51.4% | **99.9%** | 50.0% | 0.666 | **1** | **0.39 ms** |
| padatious (neural) | 66.1% | 99.7% | 65.2% | 0.789 | 3 | 3.63 ms |
| nebulento simple-ratio | **72.9%** | 96.9% | 74.4% | **0.842** | 40 | 77.2 ms |
| nebulento ratio | **72.9%** | 96.9% | **74.5%** | **0.842** | 40 | 4.04 ms |
| nebulento token-sort-ratio | 71.1% | 96.9% | 72.6% | 0.830 | 40 | 8.59 ms |
| nebulento token-set-ratio | 71.0% | 96.6% | 72.6% | 0.829 | 43 | 6.48 ms |
| nebulento damerau-levenshtein | 69.2% | 98.6% | 69.3% | 0.814 | **17** | 10.2 ms |
| nebulento partial-ratio | 64.0% | 95.8% | 65.8% | 0.780 | 49 | 10.0 ms |
| nebulento partial-token-sort-ratio | 61.4% | 95.6% | 63.2% | 0.761 | 49 | 9.15 ms |
| nebulento partial-token-ratio | 44.1% | 93.9% | 45.4% | 0.612 | 50 | 9.48 ms |
| nebulento partial-token-set-ratio | 44.1% | 93.9% | 45.4% | 0.612 | 50 | 8.89 ms |
| nebulento-hierarchical damerau-levenshtein | 62.9% | 98.4% | 62.8% | 0.767 | 17 | 8.63 ms |
| nebulento-hierarchical token-set-ratio | 55.6% | **98.8%** | 54.9% | 0.706 | **11** | 4.49 ms |

FP = false positives on the 50 `far_ood` no-match utterances. Latency varies run-to-run.

---

## Results — `massive`

2974 cases, 60 intents across 18 domains. The corpus has no no-match cases, so false positives are zero for every engine and accuracy equals recall — this dataset measures recall on a broad, diverse intent set with 13.5k training templates.

Run it with `python benchmark/compare.py massive`. The summary table has the same columns as above; because `massive` has no off-topic split, the `FP` column is zero for every engine and `Accuracy` equals `Recall`.

---

## How to Run

Install benchmark dependencies:

```bash
pip install nebulento[benchmark]
# installs: padaos, padatious, datasets
```

Run both datasets:

```bash
python benchmark/compare.py
```

Or one at a time:

```bash
python benchmark/compare.py intents-for-eval
python benchmark/compare.py massive
```

The first run downloads each dataset from the Hugging Face Hub (cached afterwards). Padatious requires a training pass; the other engines start immediately.

---

## How Metrics Are Calculated

Source: `compute_metrics` in `benchmark/compare.py`.

- **Accuracy** = (TP + TN) / total
- **Precision** = TP / (TP + FP)
- **Recall** = TP / total_match_cases
- **F1** = 2 × precision × recall / (precision + recall)
- **FP** = no-match utterances incorrectly assigned an intent

A prediction is a TP when the predicted intent name exactly matches the expected intent and `conf >= threshold` (0.5). A no-match case is correct only when the engine returns `None` or a confidence below threshold.

---

## Interpreting the Results

- **nebulento's fuzzy strategies lead on accuracy and recall.** `ratio` and `simple-ratio` reach 72.9% / 0.842 F1 on `intents-for-eval` — ahead of padatious (66.1%). The `paraphrase` split rewards fuzzy matching: rephrasings no regex template covers still score well by string similarity.
- **The cost is false positives.** The high-recall fuzzy strategies fire on 40+ of the 50 `far_ood` utterances. nebulento's `conf` should be gated by a downstream threshold — the pipeline's `conf_high` / `conf_med` / `conf_low` tiers do exactly that.
- **`ratio` is the strategy to use** — identical accuracy to `simple-ratio` at a fraction of the latency (4 ms vs 77 ms). `simple-ratio` is the difflib path and is kept only for reference.
- **`damerau-levenshtein` is the most balanced strategy** — 69.2% accuracy at 17 false positives, less than half the FP count of the other fuzzy strategies. It is the library default for this reason.
- **`partial-*` strategies saturate false positives** (49–50 / 50). They are substring matchers, not suitable for intent gating.
- **padaos and padatious are precise but lower-recall** — regex and neural matchers cannot match paraphrases as broadly as fuzzy string distance.

---

## Hierarchical variant

The two-stage `HierarchicalIntentContainer` groups intents into the dataset's domains, classifies the domain first, then resolves the intent only within it.

- **`nebulento-hierarchical damerau-levenshtein`** (no gate, `domain_threshold=0.0`) scores 62.9% vs flat damerau's 69.2% on `intents-for-eval`. The drop is the cost of misrouting: the top-level classifier is not perfect, and an utterance routed to the wrong domain cannot be recovered. False positives are unchanged (17).
- **`nebulento-hierarchical token-set-ratio`** (`domain_threshold=0.7`) cuts false positives from 43 to 11 and lifts precision to 98.8%, at a recall cost (72.6% → 54.9%). The `domain_threshold` gate rejects utterances no domain recognises before any intent is scored.

Two-stage routing is a **precision tool, not an accuracy tool**: it helps when off-topic rejection matters more than catching every command, and when domains are lexically distinct enough for the classifier to route reliably. With intents whose vocabulary overlaps across domains, the misrouting cost is real — the flat engine is the better default. See [Hierarchical Matching](hierarchical-matching.md#off-topic-rejection).
