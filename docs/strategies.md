# Match Strategies

Nebulento exposes nine similarity algorithms via the `MatchStrategy` enum (`nebulento/fuzz.py:15`). The strategy is selected at container construction time and applies to every intent scored by that container.

```python
from nebulento import IntentContainer, MatchStrategy

container = IntentContainer(fuzzy_strategy=MatchStrategy.TOKEN_SET_RATIO)
```

---

## Benchmark Summary

268 test cases: 244 natural human utterances across 22 intents, plus 24 deliberate no-match cases. Test utterances use contractions, idioms, and indirect phrasing — not template fills.

| Strategy | Accuracy | Precision | Recall | F1 | False Positives |
|---|---|---|---|---|---|
| `TOKEN_SET_RATIO` | 50.4% | 88.3% | **52.5%** | **0.658** | 17 / 24 |
| `SIMPLE_RATIO` | 49.6% | 93.6% | 48.0% | 0.634 | 8 / 24 |
| `RATIO` | 48.5% | 91.4% | 48.0% | 0.629 | 11 / 24 |
| `TOKEN_SORT_RATIO` | 43.3% | 89.0% | 43.0% | 0.580 | 13 / 24 |
| `DAMERAU_LEVENSHTEIN_SIMILARITY` | 38.8% | **100%** | 32.8% | 0.494 | **0 / 24** |
| `PARTIAL_RATIO` | 40.3% | 81.8% | 44.3% | 0.574 | 24 / 24 |
| `PARTIAL_TOKEN_RATIO` | ≤35% | ≤80% | ≤38% | ≤0.52 | 24 / 24 |
| `PARTIAL_TOKEN_SORT_RATIO` | ≤35% | ≤80% | ≤38% | ≤0.52 | 24 / 24 |
| `PARTIAL_TOKEN_SET_RATIO` | ≤35% | ≤80% | ≤38% | ≤0.52 | 24 / 24 |

Latency: median 5–7 ms per utterance for all nebulento strategies on this dataset size.

---

## Strategy Reference

### `SIMPLE_RATIO`

`nebulento/fuzz.py:23` — maps to `SequenceMatcher(None, x, against).ratio()` (difflib fallback path, i.e. when no other branch matches).

**What it measures:** Character-level Levenshtein ratio between two strings. Counts the number of matching characters as a fraction of the total characters in both strings.

**When to use:** Good general-purpose baseline. Handles spelling errors moderately well. Not sensitive to word order — transposed words lower the score.

**False positive risk:** Medium. Strings sharing many characters (common function words, short strings) can score surprisingly high.

**Benchmark:** Accuracy 49.6%, Precision 93.6%, F1 0.634, 8 / 24 false positives.

---

### `RATIO`

`nebulento/fuzz.py:63` — maps to `rapidfuzz.fuzz.ratio`.

**What it measures:** Identical to `SIMPLE_RATIO` in effect: character-level Levenshtein ratio, computed via rapidfuzz (faster than difflib for long strings).

**When to use:** Effectively interchangeable with `SIMPLE_RATIO` for intent matching purposes. Prefer `RATIO` over `SIMPLE_RATIO` for marginally better performance at scale.

**False positive risk:** Medium.

**Benchmark:** Accuracy 48.5%, Precision 91.4%, F1 0.629, 11 / 24 false positives.

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

**Benchmark:** Accuracy 43.3%, Precision 89.0%, F1 0.580, 13 / 24 false positives.

---

### `TOKEN_SET_RATIO`

`nebulento/fuzz.py:69` — maps to `rapidfuzz.fuzz.token_set_ratio`.

**What it measures:** Splits both strings into token sets. The score is the maximum of several comparisons: the intersection alone, intersection + sorted remainder of each string. Very tolerant of extra or missing words.

**When to use:** When recall is the priority and a downstream confidence gate filters false positives. Achieves the highest recall of all strategies on the benchmark.

**False positive risk:** High. 17 / 24 false positives on the benchmark dataset. Not suitable as a sole gating mechanism without a confidence threshold.

**Benchmark:** Accuracy 50.4%, Precision 88.3%, Recall 52.5%, F1 0.658, 17 / 24 false positives. Highest F1 among all strategies.

---

### `PARTIAL_TOKEN_RATIO`

`nebulento/fuzz.py:75` — maps to `rapidfuzz.fuzz.partial_token_ratio`.

**What it measures:** Token-split variant of `PARTIAL_RATIO`. Applies partial substring matching to the token-level representation.

**When to use:** Not recommended for intent classification. The partial approach means almost any short utterance produces a high score against any template.

**False positive risk:** Very high (24 / 24 on the benchmark).

---

### `PARTIAL_TOKEN_SORT_RATIO`

`nebulento/fuzz.py:71` — maps to `rapidfuzz.fuzz.partial_token_sort_ratio`.

**What it measures:** Token-sort then partial substring match.

**When to use:** Not recommended for intent gating.

**False positive risk:** Very high (24 / 24).

---

### `PARTIAL_TOKEN_SET_RATIO`

`nebulento/fuzz.py:73` — maps to `rapidfuzz.fuzz.partial_token_set_ratio`.

**What it measures:** Token-set then partial substring match.

**When to use:** Not recommended for intent gating.

**False positive risk:** Very high (24 / 24).

---

### `DAMERAU_LEVENSHTEIN_SIMILARITY`

`nebulento/fuzz.py:77` — maps to `rapidfuzz.distance.DamerauLevenshtein.normalized_similarity`.

**What it measures:** Edit distance between two strings counting insertions, deletions, substitutions, and transpositions (adjacent character swaps). Normalised to `[0.0, 1.0]` by string length. Transposition awareness means "teh" vs "the" scores higher than with plain Levenshtein.

**When to use:** Production deployments where false positives are unacceptable. This is the default strategy. Handles spelling errors and typos without generating false matches on semantically unrelated utterances.

**False positive risk:** Zero on the benchmark dataset (0 / 24). The only nebulento strategy matching the precision of pure regex engines.

**Benchmark:** Accuracy 38.8%, Precision 100%, Recall 32.8%, F1 0.494, 0 / 24 false positives.

**Note:** The lower recall compared to `TOKEN_SET_RATIO` reflects the nature of the test corpus (natural human phrasing far from template wording). Recall can be improved by adding more diverse training templates rather than switching strategies.

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
