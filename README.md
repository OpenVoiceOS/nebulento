# Nebulento

A lightweight fuzzy-matching intent parser built on [rapidfuzz](https://github.com/maxbachmann/rapidfuzz).

Finds the closest matching intent by comparing the utterance against all training sentences using configurable fuzzy similarity strategies. Handles spelling errors, word-order variation, contractions, and natural phrasing that exact-match parsers would miss. Best suited for small-to-medium intent sets (dozens to hundreds of training sentences per intent).

## Install

```bash
pip install nebulento
```

## Quick start

```python
from nebulento import IntentContainer, MatchStrategy

container = IntentContainer(fuzzy_strategy=MatchStrategy.TOKEN_SET_RATIO)

container.add_intent("hello", ["hello", "hi", "how are you", "what's up"])
container.add_intent("buy", ["buy {item}", "purchase {item}", "get {item} for me"])
container.add_entity("item", ["milk", "cheese"])

container.calc_intent("hello")
# {'name': 'hello', 'conf': 1.0, 'entities': {}, 'best_match': 'hello',
#  'utterance': 'hello', 'utterance_consumed': 'hello', 'utterance_remainder': '',
#  'match_strategy': 'TOKEN_SET_RATIO'}

container.calc_intent("buy milk")
# {'name': 'buy', 'conf': 0.719, 'entities': {'item': ['milk']},
#  'best_match': 'buy {item}', ...}
```

### Template syntax

| Syntax | Meaning |
|---|---|
| `(one\|of\|these)` | Alternation — expands to one variant per combination |
| `[optional]` | Optional word or phrase |
| `{entity}` | Capture group — matched against registered entity samples |

```python
container.add_intent("look_at_thing", ["I see {thing} (in|on) {place}"])
container.add_entity("place", ["floor", "table"])
container.add_entity("thing", ["food", "trash"])

container.calc_intent("I see food on the table")
# {'name': 'look_at_thing', 'conf': 0.701,
#  'entities': {'place': ['table'], 'thing': ['food']},
#  'utterance_consumed': 'i see in table food', 'utterance_remainder': 'the', ...}
```

### Result fields

| Field | Description |
|---|---|
| `name` | Matched intent name (`None` if no match) |
| `conf` | Confidence score in [0, 1] |
| `entities` | Dict of entity name → list of matched values |
| `best_match` | The training sentence that scored highest |
| `utterance` | Normalised input query |
| `utterance_consumed` | Words accounted for by the match |
| `utterance_remainder` | Words left over after matching |
| `match_strategy` | Strategy name used for this result |

## Match strategies

Choose a strategy via `IntentContainer(fuzzy_strategy=MatchStrategy.X)`.

| Strategy | Best for | FP risk |
|---|---|---|
| `TOKEN_SET_RATIO` | Natural phrasing, word-order variation | High — permissive |
| `SIMPLE_RATIO` | General use, balanced recall/precision | Medium |
| `RATIO` | Close paraphrases, moderate variation | Medium |
| `TOKEN_SORT_RATIO` | Same words, different order | Medium |
| `DAMERAU_LEVENSHTEIN_SIMILARITY` | Spelling errors, zero false positives | Low — strict |
| `PARTIAL_RATIO` | Substring presence | Very high — avoid for intent gating |
| `PARTIAL_TOKEN_*` | Not recommended for intent classification | Very high |

The default strategy is `DAMERAU_LEVENSHTEIN_SIMILARITY` (zero false positives on the benchmark dataset).

## Accuracy

Run `python benchmark/compare.py` to reproduce. 268 test cases: 244 natural human utterances across 22 intents, 24 deliberate no-match cases. All engines use the same training templates.

### Natural language benchmark (268 cases, 22 intents)

| Engine | Accuracy | Precision | Recall | F1 | False positives | Median | Mean |
|---|---|---|---|---|---|---|---|
| padaos (regex) | 25.4% | **100%** | 18.0% | 0.306 | 0 / 24 | **0.07 ms** | 0.08 ms |
| padacioso `fuzz=False` | 30.2% | **100%** | 23.4% | 0.379 | 0 / 24 | 0.48 ms | 0.51 ms |
| padacioso `fuzz=True` | 51.1% | 96.7% | 48.0% | 0.641 | 4 / 24 | 33 ms | 39 ms |
| padatious (neural) | 53.4% | 96.9% | 50.4% | 0.663 | 4 / 24 | 1.1 ms | 1.1 ms |
| nebulento `token-set-ratio` | 50.4% | 88.3% | **52.5%** | **0.658** | 17 / 24 | 6.3 ms | 6.5 ms |
| nebulento `simple-ratio` | 49.6% | 93.6% | 48.0% | 0.634 | 8 / 24 | 24 ms | 25 ms |
| nebulento `ratio` | 48.5% | 91.4% | 48.0% | 0.629 | 11 / 24 | 5.4 ms | 5.7 ms |
| nebulento `token-sort-ratio` | 43.3% | 89.0% | 43.0% | 0.580 | 13 / 24 | 6.0 ms | 6.2 ms |
| nebulento `damerau-levenshtein` | 38.8% | **100%** | 32.8% | 0.494 | **0 / 24** | 6.8 ms | 7.1 ms |
| nebulento `partial-ratio` | 40.3% | 81.8% | 44.3% | 0.574 | 24 / 24 | 6.0 ms | 6.2 ms |
| nebulento `partial-token-*` | ≤35% | ≤80% | ≤38% | ≤0.52 | 24 / 24 | ~6.5 ms | ~6.7 ms |

Test utterances are real human phrasing — contractions, idioms, indirect requests — not template fills. This is expected and by design: nebulento is a **fuzzy pattern matcher**, not an NLU engine. Recall depends on how broadly the training templates are written.

**Strategy guidance:**
- `token-set-ratio` achieves the highest recall (52.5%) but generates 17 false positives — use only when a downstream confidence gate can filter them.
- `damerau-levenshtein` is the only nebulento strategy with zero false positives, matching the precision of pure regex engines while handling spelling variation.
- `partial-*` strategies saturate false positives (24/24) and are not suitable for intent gating.
- For production deployments without a downstream filter, `damerau-levenshtein` (default) or `simple-ratio` offer the best precision/recall tradeoff.

## OVOS plugin

Nebulento ships as an OVOS pipeline plugin (`ovos-nebulento-pipeline-plugin`).

```json
// ~/.config/mycroft/mycroft.conf
{
  "intents": {
    "pipeline": [
      "ovos-nebulento-pipeline-plugin"
    ]
  }
}
```

## Credits

Originally an experimental research project by [**TigreGoticoLda**](https://tigregotico.pt), polished
and donated to OpenVoiceOS as part of the NLnet
[NGI0 Commons Fund](https://nlnet.nl/project/OpenVoiceOS) under grant
agreement No [101135429](https://cordis.europa.eu/project/id/101135429).

![NGI0 / NLnet](./ngi.png)

## License

Apache 2.0
