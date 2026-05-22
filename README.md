# Nebulento

A lightweight fuzzy-matching intent parser built on [rapidfuzz](https://github.com/maxbachmann/rapidfuzz).

Finds the closest matching intent by comparing the utterance against all training sentences using configurable fuzzy similarity strategies. Handles spelling errors, word-order variation, contractions, and natural phrasing that exact-match parsers would miss. Best suited for small-to-medium intent sets (dozens to hundreds of training sentences per intent).

---

## Install

```bash
pip install nebulento
```

For the OVOS pipeline plugin:

```bash
pip install "nebulento[ovos]"
```

---

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

---

## Match strategies

Choose a strategy via `IntentContainer(fuzzy_strategy=MatchStrategy.X)`.

| Strategy | Best for | FP risk |
|---|---|---|
| `DAMERAU_LEVENSHTEIN_SIMILARITY` | Spelling errors, lowest false-positive rate | Low — **default** |
| `RATIO` | Highest recall and F1, fast | High |
| `TOKEN_SET_RATIO` | Natural phrasing, word-order variation | High |
| `TOKEN_SORT_RATIO` | Same words, different order | High |
| `PARTIAL_RATIO` | Substring presence — avoid for intent gating | Very high |

See [docs/strategies.md](docs/strategies.md) for the full comparison table and benchmark rows.

---

## OVOS pipeline plugin

Nebulento ships as an OVOS pipeline plugin (`ovos-nebulento-pipeline-plugin`).

```json
{
  "intents": {
    "pipeline": [
      "ovos-nebulento-pipeline-plugin"
    ]
  }
}
```

Configure the fuzzy strategy and confidence thresholds:

```json
{
  "intents": {
    "nebulento": {
      "strategy": "TOKEN_SET_RATIO",
      "conf_high": 0.95,
      "conf_med":  0.80,
      "conf_low":  0.50
    }
  }
}
```

Entry point: `nebulento.opm:NebulentoPipeline`

---

## Documentation

| Page | Description |
|---|---|
| [Quickstart](docs/quickstart.md) | 5-minute guide: intents, entities, strategies |
| [Intent API](docs/intent-api.md) | Full `IntentContainer` and `HierarchicalIntentContainer` reference |
| [Match Strategies](docs/strategies.md) | All 9 strategies with benchmark data and decision table |
| [Template Syntax](docs/template-syntax.md) | `(a\|b)`, `[opt]`, `{slot}`, `:0` padatious syntax, expansion rules |
| [Entity Extraction](docs/entity-extraction.md) | Registration, confidence boost, result fields |
| [Normalisation](docs/normalisation.md) | Apostrophes, whitespace, case handling |
| [Hierarchical Matching](docs/hierarchical-matching.md) | `HierarchicalIntentContainer` two-stage matching |
| [OVOS Pipeline Plugin](docs/ovos-plugin.md) | Bus events, confidence tiers, comparison with Padatious |
| [Configuration](docs/configuration.md) | All config keys with types, defaults, and effect |
| [Benchmark](docs/benchmark.md) | Full accuracy results across all strategies |
| [Troubleshooting](docs/troubleshooting.md) | False positives, low recall, entity issues, lru_cache gotchas |

---

## Benchmark

Evaluated on the English subset of [`OpenVoiceOS/intents-for-eval`](https://huggingface.co/datasets/OpenVoiceOS/intents-for-eval) — 1750 test utterances across 50 intents (1700 match, 50 off-topic).

| Engine | Accuracy | Precision | Recall | F1 | False positives | Median |
|---|---|---|---|---|---|---|
| padaos (regex) | 51.4% | **99.9%** | 50.0% | 0.666 | **1 / 50** | **0.34 ms** |
| padatious (neural) | 65.5% | 99.7% | 64.6% | 0.784 | 3 / 50 | 3.3 ms |
| nebulento `ratio` | **72.9%** | 96.9% | **74.5%** | **0.842** | 40 / 50 | 4.2 ms |
| nebulento `damerau-levenshtein` | 69.2% | 98.6% | 69.3% | 0.814 | 17 / 50 | 9.4 ms |

```bash
python benchmark/compare.py
```

See [docs/benchmark.md](docs/benchmark.md) for the full table, all nine strategies, and the hierarchical variant.

---

## Credits

Originally an experimental research project by [**TigreGoticoLda**](https://tigregotico.pt), polished
and donated to OpenVoiceOS as part of the NLnet
[NGI0 Commons Fund](https://nlnet.nl/project/OpenVoiceOS) under grant
agreement No [101135429](https://cordis.europa.eu/project/id/101135429).

![NGI0 / NLnet](./ngi.png)

---

## License

Apache 2.0
