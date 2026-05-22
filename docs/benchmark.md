# Benchmark

Nebulento includes a comparative accuracy and speed benchmark in `benchmark/compare.py`.

---

## Dataset

The benchmark runs on the English subset of [`OpenVoiceOS/intents-for-eval`](https://huggingface.co/datasets/OpenVoiceOS/intents-for-eval), loaded from the Hugging Face Hub by `benchmark/dataset.py`. Two of the dataset's configs are used:

- **`en-US-templates`** — 1000 training templates across 50 intents (e.g. `play {song} by {artist}`), each carrying example values for its `{slot}` placeholders.
- **`en-US-test`** — 1750 labelled evaluation utterances across six splits.

```
Dataset : OpenVoiceOS/intents-for-eval  (en-US)
Cases   : 1750  (1700 match, 50 no-match)
Intents : 50
Splits  : template=500, paraphrase=700, near_ood=400, far_ood=50, asr_noise=50, typos=50
```

The test splits exercise different robustness aspects:

| Split | Cases | What it tests |
|---|---|---|
| `template` | 500 | Utterances that fill a training template directly |
| `paraphrase` | 700 | Natural rephrasings — different words, same intent |
| `near_ood` | 400 | Boundary utterances close to another intent |
| `far_ood` | 50 | Genuinely off-topic — should match **nothing** |
| `asr_noise` | 50 | Speech-recognition artefacts (filler words, mishears) |
| `typos` | 50 | Spelling errors |

`far_ood` is the no-match set; every other split carries a real intent label.

The keyword config (`en-US-keywords`) is for keyword engines (Adapt, palavreado) and is not used here — every engine in this benchmark is a template / sample matcher.

### Entities

Each `{slot}` placeholder in the dataset ships with example values. `benchmark/dataset.py` collects them into an `ENTITIES` map and every engine registers them (the equivalent of a padatious `.entity` file) before matching, so slots can be filled and contribute to confidence.

---

## Engines Compared

Every engine here is a **template / sample matcher** — it trains on example sentences, not keyword vocabularies — so all of them train on `en-US-templates` and are evaluated on `en-US-test`.

| Engine | Description |
|---|---|
| `padaos` | Regex-based exact matcher (no fuzzy) |
| `padacioso fuzz=False` | Regex-based matcher, no fuzzy |
| `padacioso fuzz=True` | Regex with rapidfuzz augmentation |
| `padatious` | Neural network matcher (requires a training pass) |
| `nebulento <strategy>` | Flat `IntentContainer`, one row per `MatchStrategy` value |
| `nebulento-hierarchical <strategy>` | Two-stage `HierarchicalIntentContainer` — intents grouped into the dataset's 10 domains |

---

## Results Table

| Engine | Accuracy | Precision | Recall | F1 | FP / 50 | Median lat |
|---|---|---|---|---|---|---|
| padaos (regex) | 51.4% | **99.9%** | 50.0% | 0.666 | **1** | **0.34 ms** |
| padacioso fuzz=False | 55.1% | 99.6% | 54.1% | 0.701 | 4 | 1.06 ms |
| padacioso fuzz=True | 63.7% | 97.8% | 64.1% | 0.774 | 25 | 135 ms |
| padatious (neural) | 65.5% | 99.7% | 64.6% | 0.784 | 3 | 3.29 ms |
| nebulento ratio | **72.9%** | 96.9% | **74.5%** | **0.842** | 40 | 4.15 ms |
| nebulento simple-ratio | 72.9% | 96.9% | 74.4% | 0.842 | 40 | 78.9 ms |
| nebulento token-sort-ratio | 71.1% | 96.9% | 72.6% | 0.830 | 40 | 5.49 ms |
| nebulento token-set-ratio | 71.1% | 96.6% | 72.8% | 0.830 | 43 | 6.19 ms |
| nebulento damerau-levenshtein | 69.2% | 98.6% | 69.3% | 0.814 | 17 | 9.43 ms |
| nebulento partial-ratio | 64.0% | 95.8% | 65.8% | 0.780 | 49 | 6.97 ms |
| nebulento partial-token-sort-ratio | 61.4% | 95.6% | 63.2% | 0.761 | 49 | 9.56 ms |
| nebulento partial-token-(set\|)-ratio | 49.0% | 94.5% | 50.4% | 0.657 | 50 | ~10 ms |
| nebulento-hierarchical damerau-levenshtein | 62.8% | 98.4% | 62.7% | 0.766 | 17 | ~9 ms |
| nebulento-hierarchical token-set-ratio | 55.7% | **98.8%** | 55.0% | 0.707 | **11** | 4.27 ms |

FP = false positives on the 50 `far_ood` no-match utterances. Latency varies run-to-run; treat it as an order of magnitude.

---

## How to Run

Install benchmark dependencies:

```bash
pip install nebulento[benchmark]
# installs: padaos, padacioso, padatious, datasets
```

Run:

```bash
python benchmark/compare.py
```

The first run downloads the dataset from the Hugging Face Hub (cached afterwards). Padatious requires a training pass; all other engines start immediately.

---

## How Metrics Are Calculated

Source: `compute_metrics` in `benchmark/compare.py`.

- **Accuracy** = (TP + TN) / total
- **Precision** = TP / (TP + FP)
- **Recall** = TP / total_match_cases
- **F1** = 2 × precision × recall / (precision + recall)
- **FP** = number of `far_ood` utterances incorrectly assigned an intent

A prediction is a TP when the predicted intent name exactly matches the expected intent and `conf >= threshold` (0.5). A no-match case is correct only when the engine returns `None` or a confidence below threshold.

---

## Interpreting the Results

- **nebulento's fuzzy strategies lead on accuracy and recall.** `ratio` and `simple-ratio` reach 72.9% accuracy / 0.842 F1 — ahead of padatious (65.5% / 0.784). The `paraphrase` split rewards fuzzy matching: rephrasings that no regex template covers still score well by string similarity.
- **The cost is false positives.** The high-recall fuzzy strategies fire on 40+ of the 50 `far_ood` utterances. nebulento's `conf` should be gated by a downstream threshold — that is exactly what the pipeline's `conf_high` / `conf_med` / `conf_low` tiers do.
- **`damerau-levenshtein` is the most balanced nebulento strategy** — 69.2% accuracy at 17 false positives, the lowest FP count of any fuzzy strategy. It is the library default for this reason.
- **`partial-*` strategies saturate false positives** (49–50 / 50). They are substring matchers and are not suitable for intent gating.
- **padaos and padacioso are precise but low-recall** — pure regex cannot match paraphrases it was not given a template for.
- **padatious (neural)** lands between the regex engines and nebulento: better recall than regex, lower than fuzzy, with the highest precision after padaos.
- **Latency:** `ratio` is the fast nebulento strategy (~4 ms); `simple-ratio` (difflib-based) is far slower (~79 ms) for identical accuracy — prefer `ratio`. padacioso with fuzz is slowest (~135 ms).

---

## Hierarchical variant

The two-stage `HierarchicalIntentContainer` groups the 50 intents into the 10 domains the dataset already defines, classifies the domain first, then resolves the intent only within it.

- **`nebulento-hierarchical damerau-levenshtein`** (no gate, `domain_threshold=0.0`) scores 62.8% vs flat damerau's 69.2%. The drop is the cost of misrouting: the top-level classifier is not perfect, and an utterance routed to the wrong domain can no longer be recovered. False positives are unchanged (17) because the gate is off.
- **`nebulento-hierarchical token-set-ratio`** (`domain_threshold=0.7`) cuts false positives from 43 to 11 and lifts precision to 98.8%, at a large recall cost (72.8% → 55.0%). The `domain_threshold` gate rejects utterances no domain recognises before any intent is scored.

The lesson on this dataset: **two-stage routing is a precision tool, not an accuracy tool.** It helps when off-topic rejection matters more than catching every command, and when your domains are lexically distinct enough for the classifier to route reliably. With 50 intents whose vocabulary overlaps across domains, the misrouting cost is real — the flat engine is the better default. See [Hierarchical Matching](hierarchical-matching.md#off-topic-rejection).

The `domain_threshold` is strategy-dependent: `TOKEN_SET_RATIO` scores loosely and needs a high gate (0.7) before it rejects anything; `DAMERAU_LEVENSHTEIN_SIMILARITY` is strict and its `conf` already does most of the rejecting. With `domain_threshold=0.0` (the default) the hierarchical container routes every query and the only difference from the flat engine is domain-scoped vocabulary isolation.
