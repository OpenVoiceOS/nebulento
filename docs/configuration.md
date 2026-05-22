# Configuration

Nebulento's OVOS pipeline plugin is configured under the `nebulento` key inside `intents` in `mycroft.conf` (`~/.config/mycroft/mycroft.conf`).

The two-stage [`HierarchicalNebulentoPipeline`](hierarchical-matching.md) is configured the same way under the `nebulento_hierarchical` key, and accepts every key below plus `domain_threshold`. The two plugins can run side by side in one OVOS instance.

---

## Full Example

```json
{
  "lang": "en-US",
  "secondary_langs": ["es-ES", "fr-FR"],
  "intents": {
    "pipeline": [
      "ovos-nebulento-pipeline-plugin"
    ],
    "nebulento": {
      "conf_high": 0.95,
      "conf_med": 0.80,
      "conf_low": 0.50,
      "max_words": 50,
      "strategy": "DAMERAU_LEVENSHTEIN_SIMILARITY"
    }
  }
}
```

---

## Config Key Reference

### `strategy`

| Property | Value |
|---|---|
| Type | `str` |
| Default | `"DAMERAU_LEVENSHTEIN_SIMILARITY"` |
| Source | `nebulento/opm.py:72` |

Name of the `MatchStrategy` enum value to use for all intent matching. Must match an enum member name exactly (case-sensitive).

Valid values:

| Value | Notes |
|---|---|
| `"SIMPLE_RATIO"` | Character-level Levenshtein ratio (difflib) |
| `"RATIO"` | rapidfuzz ratio — effectively same as `SIMPLE_RATIO` |
| `"PARTIAL_RATIO"` | Substring match — high FP risk |
| `"TOKEN_SORT_RATIO"` | Sorted tokens, word-order insensitive |
| `"TOKEN_SET_RATIO"` | Set intersection — highest recall, most FP |
| `"PARTIAL_TOKEN_RATIO"` | Not recommended for intent gating |
| `"PARTIAL_TOKEN_SORT_RATIO"` | Not recommended for intent gating |
| `"PARTIAL_TOKEN_SET_RATIO"` | Not recommended for intent gating |
| `"DAMERAU_LEVENSHTEIN_SIMILARITY"` | Default — zero FP on benchmark |

An unrecognised value falls back to `"DAMERAU_LEVENSHTEIN_SIMILARITY"` with a warning logged.

---

### `conf_high`

| Property | Value |
|---|---|
| Type | `float` |
| Default | `0.95` |
| Source | `nebulento/opm.py:65` |

Confidence threshold for `match_high`. Intents with `conf > conf_high` are returned at this tier. The OVOS pipeline tries this tier first; a high-confidence match prevents lower-priority engines from running.

---

### `conf_med`

| Property | Value |
|---|---|
| Type | `float` |
| Default | `0.80` |
| Source | `nebulento/opm.py:66` |

Confidence threshold for `match_medium`. Intents with `conf > conf_med` are returned at this tier (after `match_high` returned nothing).

---

### `conf_low`

| Property | Value |
|---|---|
| Type | `float` |
| Default | `0.50` |
| Source | `nebulento/opm.py:67` |

Confidence threshold for `match_low`. This is the lowest tier; matches above this threshold are returned only if `match_high` and `match_medium` both returned nothing. Setting this too low increases the risk of spurious intent matches.

---

### `max_words`

| Property | Value |
|---|---|
| Type | `int` |
| Default | `50` |
| Source | `nebulento/opm.py:68` |

Maximum number of space-separated tokens an utterance may contain. Utterances exceeding this limit are silently dropped before matching. This guards against extremely long inputs (e.g. dictated paragraphs) which would produce meaningless fuzzy scores.

If all utterances in a request exceed `max_words`, `calc_intent` returns `None` and an error is logged.

---

### `domain_threshold` (hierarchical plugin only)

| Property | Value |
|---|---|
| Type | `float` |
| Default | `0.0` |
| Source | `nebulento/opm.py:284` |

Only read by `HierarchicalNebulentoPipeline` under the `nebulento_hierarchical` key. Minimum confidence the top-level domain classifier must reach for a query to be routed to a domain. When the best domain scores below this value the query is rejected with no intent match. `0.0` (default) disables the gate — every query is routed to its best domain and rejection is left to the `conf_*` tiers. Raising it trades recall for precision. See [Hierarchical Matching](hierarchical-matching.md#off-topic-rejection).

---

### `lang` (global key)

| Property | Value |
|---|---|
| Type | `str` (BCP-47) |
| Default | `"en-US"` |
| Source | `nebulento/opm.py:59` |

The primary language. One `IntentContainer` is created for this language. Read from the top-level `ovos_config.config.Configuration()`, not from the `nebulento` sub-key.

---

### `secondary_langs` (global key)

| Property | Value |
|---|---|
| Type | `List[str]` (BCP-47) |
| Default | `[]` |
| Source | `nebulento/opm.py:60-63` |

Additional languages to support. One `IntentContainer` is created per language. Skills that register intents with a `lang` field matching a secondary language are handled by the corresponding container. Intents registered with a language not in this list (or not close enough by `langcodes.closest_match`) are silently ignored.

Read from the top-level `ovos_config.config.Configuration()`.

---

## Standalone Library

When using `IntentContainer` directly (outside OVOS), there is no config file. Strategy and case-folding are set via constructor arguments:

```python
from nebulento import IntentContainer, MatchStrategy

c = IntentContainer(
    fuzzy_strategy=MatchStrategy.TOKEN_SET_RATIO,
    ignore_case=True,
)
```

See [Intent API](intent-api.md) for the full constructor reference.
