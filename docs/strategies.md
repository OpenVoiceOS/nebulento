# Match Strategies

Nebulento exposes nine similarity algorithms via the `MatchStrategy` enum (`nebulento/fuzz.py:15`). The strategy is selected at container construction time and applies to every intent scored by that container.

```python
from nebulento import IntentContainer, MatchStrategy

container = IntentContainer(fuzzy_strategy=MatchStrategy.TOKEN_SET_RATIO)
```

---

## Benchmark Summary

Evaluated on the English subset of `OpenVoiceOS/intents-for-eval` — 1750 test utterances across 50 intents (1700 match, 50 off-topic). See [Benchmark](benchmark.md) for the full methodology.

| Strategy | Accuracy | Precision | Recall | F1 | False Positives |
|---|---|---|---|---|---|
| `RATIO` | **72.9%** | 96.9% | **74.5%** | **0.842** | 40 / 50 |
| `SIMPLE_RATIO` | 72.9% | 96.9% | 74.4% | 0.842 | 40 / 50 |
| `TOKEN_SORT_RATIO` | 71.1% | 96.9% | 72.6% | 0.830 | 40 / 50 |
| `TOKEN_SET_RATIO` | 71.0% | 96.6% | 72.6% | 0.829 | 43 / 50 |
| `DAMERAU_LEVENSHTEIN_SIMILARITY` | 69.2% | **98.6%** | 69.3% | 0.814 | **17 / 50** |
| `PARTIAL_RATIO` | 64.0% | 95.8% | 65.8% | 0.780 | 49 / 50 |
| `PARTIAL_TOKEN_SORT_RATIO` | 61.4% | 95.6% | 63.2% | 0.761 | 49 / 50 |
| `PARTIAL_TOKEN_RATIO` | 44.1% | 93.9% | 45.4% | 0.612 | 50 / 50 |
| `PARTIAL_TOKEN_SET_RATIO` | 44.1% | 93.9% | 45.4% | 0.612 | 50 / 50 |

Latency: a few ms per utterance for most strategies; `SIMPLE_RATIO` (difflib) is far slower than `RATIO` for the same accuracy — prefer `RATIO`.

---

## Strategy Reference

### `SIMPLE_RATIO`

`nebulento/fuzz.py:23` — maps to `SequenceMatcher(None, x, against).ratio()` (difflib fallback path, i.e. when no other branch matches).

**What it measures:** Character-level Levenshtein ratio between two strings. Counts the number of matching characters as a fraction of the total characters in both strings.

**When to use:** Good general-purpose baseline. Handles spelling errors moderately well. Not sensitive to word order — transposed words lower the score.

**False positive risk:** Medium. Strings sharing many characters (common function words, short strings) can score surprisingly high.

**Benchmark:** Accuracy 72.9%, Precision 96.9%, F1 0.842, 40 / 50 false positives. Tied for the highest F1, but slow (difflib) — `RATIO` gives the same accuracy faster.

---

### `RATIO`

`nebulento/fuzz.py:63` — maps to `rapidfuzz.fuzz.ratio`.

**What it measures:** Identical to `SIMPLE_RATIO` in effect: character-level Levenshtein ratio, computed via rapidfuzz (faster than difflib for long strings).

**When to use:** Effectively interchangeable with `SIMPLE_RATIO` for intent matching purposes. Prefer `RATIO` over `SIMPLE_RATIO` for marginally better performance at scale.

**False positive risk:** Medium.

**Benchmark:** Accuracy 72.9%, Precision 96.9%, F1 0.842, 40 / 50 false positives. Highest F1, and fast — the recommended general-purpose strategy.

---

### `PARTIAL_RATIO`

`nebulento/fuzz.py:65` — maps to `rapidfuzz.fuzz.partial_ratio`.

**What it measures:** Best alignment of the shorter string as a substring within the longer string. A short query can match a long template with a high score even if most of the template's words are absent from the query.

**When to use:** Rarely appropriate for intent gating. Can be useful when intent templates contain long optional context phrases and the query is expected to be short.

**False positive risk:** Very high. Saturates at 24/24 false positives on the benchmark (every no-match utterance is incorrectly matched).

---

### `TOKEN_SORT_RATIO`

`nebulento/fuzz.py:67` — maps to `rapidfuzz.fuzz.token_sort_ratio`.

**What it measures:** Both strings are tokenised and tokens are sorted alphabetically before applying the character-level ratio. Identical sets of words in different orders receive the same score.

**When to use:** When the training templates and expected utterances use the same vocabulary but users may vary word order significantly. Example: "play song jazz" vs "jazz song play" would score identically.

**False positive risk:** Medium. Sorting tokens discards word-order information, which helps recall but also means that unrelated short strings with common words can score higher than expected.

**Benchmark:** Accuracy 71.1%, Precision 96.9%, F1 0.830, 40 / 50 false positives.

---

### `TOKEN_SET_RATIO`

`nebulento/fuzz.py:69` — maps to `rapidfuzz.fuzz.token_set_ratio`.

**What it measures:** Splits both strings into token sets. The score is the maximum of several comparisons: the intersection alone, intersection + sorted remainder of each string. Very tolerant of extra or missing words.

**When to use:** When tolerance to extra or missing words matters and a downstream confidence gate filters false positives. Tolerant of word-order and filler words, at the cost of the highest false-positive count.

**False positive risk:** High. 43 / 50 false positives on the benchmark dataset. Not suitable as a sole gating mechanism without a confidence threshold.

**Benchmark:** Accuracy 71.0%, Precision 96.6%, Recall 72.6%, F1 0.829, 43 / 50 false positives.

---

### `PARTIAL_TOKEN_RATIO`

`nebulento/fuzz.py:75` — maps to `rapidfuzz.fuzz.partial_token_ratio`.

**What it measures:** Token-split variant of `PARTIAL_RATIO`. Applies partial substring matching to the token-level representation.

**When to use:** Not recommended for intent classification. The partial approach means almost any short utterance produces a high score against any template.

**False positive risk:** Very high (50 / 50 on the benchmark).

---

### `PARTIAL_TOKEN_SORT_RATIO`

`nebulento/fuzz.py:71` — maps to `rapidfuzz.fuzz.partial_token_sort_ratio`.

**What it measures:** Token-sort then partial substring match.

**When to use:** Not recommended for intent gating.

**False positive risk:** Very high (49 / 50).

---

### `PARTIAL_TOKEN_SET_RATIO`

`nebulento/fuzz.py:73` — maps to `rapidfuzz.fuzz.partial_token_set_ratio`.

**What it measures:** Token-set then partial substring match.

**When to use:** Not recommended for intent gating.

**False positive risk:** Very high (50 / 50).

---

### `DAMERAU_LEVENSHTEIN_SIMILARITY`

`nebulento/fuzz.py:77` — maps to `rapidfuzz.distance.DamerauLevenshtein.normalized_similarity`.

**What it measures:** Edit distance between two strings counting insertions, deletions, substitutions, and transpositions (adjacent character swaps). Normalised to `[0.0, 1.0]` by string length. Transposition awareness means "teh" vs "the" scores higher than with plain Levenshtein.

**When to use:** Production deployments where false positives matter. This is the default strategy. Handles spelling errors and typos while keeping the lowest false-positive count of any fuzzy strategy.

**False positive risk:** Low. 17 / 50 on the benchmark dataset — far below the other fuzzy strategies (40–50 / 50).

**Benchmark:** Accuracy 69.2%, Precision 98.6%, Recall 69.3%, F1 0.814, 17 / 50 false positives.

**Note:** Slightly lower recall and F1 than `RATIO`, traded for roughly half the false positives. Pick `RATIO` for maximum recall, `DAMERAU_LEVENSHTEIN_SIMILARITY` when off-topic rejection matters.

---

## Decision Table

| Situation | Recommended strategy |
|---|---|
| Production deployment, no downstream confidence filter | `DAMERAU_LEVENSHTEIN_SIMILARITY` |
| General use, balanced starting point | `SIMPLE_RATIO` or `RATIO` |
| Users may transpose words or change phrasing significantly | `TOKEN_SORT_RATIO` |
| Maximum recall, downstream filter available | `TOKEN_SET_RATIO` |
| Known spelling errors / STT noise dominant | `DAMERAU_LEVENSHTEIN_SIMILARITY` |
| Partial keyword presence detection | `PARTIAL_RATIO` (not for gating) |
| Substring match not needed | Avoid all `PARTIAL_*` strategies |

---

## Strategy Selection in Code

```python
from nebulento import IntentContainer, MatchStrategy

# Default (zero FP, lower recall):
c = IntentContainer()

# By name (useful when reading from config):
strategy = MatchStrategy["TOKEN_SET_RATIO"]
c = IntentContainer(fuzzy_strategy=strategy)

# As IntEnum value — strategies can be compared and sorted:
print(MatchStrategy.DAMERAU_LEVENSHTEIN_SIMILARITY.name)
# 'DAMERAU_LEVENSHTEIN_SIMILARITY'
```

The `NebulentoPipeline` OVOS plugin reads strategy from `mycroft.conf` by name:

```json
{
  "intents": {
    "nebulento": {
      "strategy": "TOKEN_SET_RATIO"
    }
  }
}
```

An unknown strategy name falls back to `DAMERAU_LEVENSHTEIN_SIMILARITY` with a warning. Source: `nebulento/opm.py:72`.
