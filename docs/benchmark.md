# Benchmark

Nebulento includes a comparative accuracy and speed benchmark in `benchmark/compare.py`.

---

## Dataset

Source: `benchmark/dataset.py`.

- **268 total test cases**: 244 natural human utterances across 22 intents, plus 24 deliberate no-match cases.
- **22 intents** covering domains: media playback, home automation, calendar, timers, weather, general knowledge, alarms, and miscellaneous utility.
- **Test utterances are natural human phrasing** — contractions, filler words, politeness markers, regional phrasing, word-order variation, colloquialisms. They are not filled-in template strings.

```
Dataset : 268 cases  (244 match, 24 no-match)
Intents : 22
Note    : test utterances are natural human phrasing, NOT template fills.
```

The mismatch between test utterance style and template style is intentional and expected. Nebulento is a fuzzy pattern matcher, not an NLU engine; recall depends on how broadly templates are written.

---

## Engines Compared

| Engine | Description |
|---|---|
| `padaos` | Regex-based exact matcher (no fuzzy) |
| `padacioso fuzz=False` | Regex-based matcher, no fuzzy |
| `padacioso fuzz=True` | Regex with rapidfuzz augmentation |
| `padatious` | Neural network matcher (requires training) |
| `nebulento <strategy>` | One row per `MatchStrategy` value |

All engines use the same training templates from `INTENTS[name]["train"]`.

---

## Results Table

| Engine | Accuracy | Precision | Recall | F1 | FP / 24 | Median lat | Mean lat |
|---|---|---|---|---|---|---|---|
| padaos (regex) | 25.4% | 100% | 18.0% | 0.306 | 0 | 0.07 ms | 0.08 ms |
| padacioso fuzz=False | 30.2% | 100% | 23.4% | 0.379 | 0 | 0.48 ms | 0.51 ms |
| padacioso fuzz=True | 51.1% | 96.7% | 48.0% | 0.641 | 4 | 33 ms | 39 ms |
| padatious (neural) | 53.4% | 96.9% | 50.4% | 0.663 | 4 | 1.1 ms | 1.1 ms |
| nebulento token-set-ratio | 50.4% | 88.3% | **52.5%** | **0.658** | 17 | 6.3 ms | 6.5 ms |
| nebulento simple-ratio | 49.6% | 93.6% | 48.0% | 0.634 | 8 | 24 ms | 25 ms |
| nebulento ratio | 48.5% | 91.4% | 48.0% | 0.629 | 11 | 5.4 ms | 5.7 ms |
| nebulento token-sort-ratio | 43.3% | 89.0% | 43.0% | 0.580 | 13 | 6.0 ms | 6.2 ms |
| nebulento damerau-levenshtein | 38.8% | **100%** | 32.8% | 0.494 | **0** | 6.8 ms | 7.1 ms |
| nebulento partial-ratio | 40.3% | 81.8% | 44.3% | 0.574 | 24 | 6.0 ms | 6.2 ms |
| nebulento partial-token-* | ≤35% | ≤80% | ≤38% | ≤0.52 | 24 | ~6.5 ms | ~6.7 ms |

FP = false positives on the 24 no-match utterances (how many times an intent was returned when none should have been).

---

## How to Run

Install benchmark dependencies:

```bash
pip install nebulento[benchmark]
# installs: padaos, padacioso, padatious
```

Run:

```bash
python benchmark/compare.py
```

Or with uv:

```bash
uv run python benchmark/compare.py
```

Padatious requires a training pass (~several seconds). All other engines start immediately.

The script prints a per-engine report followed by a summary table:

```
Dataset : 268 cases  (244 match, 24 no-match)
Intents : 22

  padaos  (regex)                      25.4%  100.0%  18.0% 0.306     0    0.07ms    0.08ms
  padacioso  fuzz=False                30.2%  100.0%  23.4% 0.379     0    0.48ms    0.51ms
  ...
```

---

## How Metrics Are Calculated

Source: `benchmark/accuracy.py` (via `compute_metrics` in `benchmark/compare.py`).

- **Accuracy** = (TP + TN) / total
- **Precision** = TP / (TP + FP)
- **Recall** = TP / total_match_cases
- **F1** = 2 × precision × recall / (precision + recall)
- **FP** = number of no-match utterances incorrectly assigned an intent

A prediction is counted as TP when the predicted intent name exactly matches the expected intent name and `conf >= threshold` (0.5). A no-match case is correct only when the engine returns `None` or a confidence below threshold.

---

## Interpreting the Results

The benchmark tests how well each engine handles the gap between the training template style and natural human speech. Key observations:

- **No engine achieves high recall on this dataset without accepting some false positives.** The test utterances are deliberately challenging — real STT output, not template fills.
- **`token-set-ratio` achieves the highest recall (52.5%)** but 17 / 24 false positives make it unsuitable as a sole gating mechanism. Use with a downstream confidence threshold.
- **`damerau-levenshtein` is the only strategy with zero false positives.** It matches the precision of regex engines while handling spelling variation. The lower recall reflects strict string distance — it does not semantically understand paraphrases.
- **padatious (neural) achieves the best F1 (0.663)** among all tested engines and runs at 1.1 ms median. It requires a training step and a training corpus; nebulento requires neither.
- **`partial-*` strategies are not useful for intent gating** — they saturate false positives at 24/24, meaning every no-match utterance is incorrectly classified.
- **Latency:** all nebulento strategies run at 5–7 ms per utterance for this 22-intent set. padaos is fastest at 0.07 ms (pure regex). padacioso with fuzz is slowest at 33 ms.
