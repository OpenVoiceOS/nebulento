# OVOS Pipeline Plugin

Nebulento ships as a first-class OVOS pipeline plugin. The plugin class is `NebulentoPipeline` (`nebulento/opm.py:51`) and is registered under the entry point name `ovos-nebulento-pipeline-plugin`.

---

## Entry Point

Defined in `pyproject.toml`:

```toml
[project.entry-points."opm.pipeline"]
"ovos-nebulento-pipeline-plugin" = "nebulento.opm:NebulentoPipeline"
```

`ovos-plugin-manager` discovers this automatically when nebulento is installed. No manual registration is required.

---

## Class Hierarchy

```
ConfidenceMatcherPipeline  (ovos-plugin-manager)
        │
        └── NebulentoPipeline
```

`ConfidenceMatcherPipeline` provides `match_high`, `match_medium`, and `match_low` at three configurable confidence thresholds, integrating with the OVOS pipeline priority system.

---

## Initialisation

`nebulento/opm.py:54`

```python
NebulentoPipeline(bus=None, config=None)
```

On startup the plugin:

1. Reads `lang` and `secondary_langs` from `ovos_config.config.Configuration()`.
2. Reads confidence thresholds and strategy from `self.config` (the `nebulento` section of `mycroft.conf`).
3. Creates one `IntentContainer` per language (primary + secondary).
4. Registers message bus event handlers.

---

## Bus Events

| Event | Handler | Description |
|---|---|---|
| `padatious:register_intent` | `register_intent` | Register an intent from a skill's `.intent` file or inline samples |
| `padatious:register_entity` | `register_entity` | Register an entity from a skill's `.entity` file or inline samples |
| `detach_intent` | `handle_detach_intent` | Unregister a single intent by name |
| `detach_skill` | `handle_detach_skill` | Unregister all intents and entities belonging to a skill |
| `mycroft.skills.train` | `train` | Emit `mycroft.skills.trained` immediately (no training needed) |

Source: `nebulento/opm.py:81-85`.

---

## Intent Registration

`nebulento/opm.py:137`

`register_intent` handles `padatious:register_intent` events. The message data may contain:

- `file_name` — path to a `.intent` file with one template per line
- `samples` — inline list of template strings (takes precedence over `file_name`)
- `name` — intent identifier (typically `skill_id:intent_name`)
- `lang` — BCP-47 language tag (defaults to the plugin's primary lang)

If the language tag does not match any registered container (primary or secondary), the intent is silently ignored.

On skill reload, a duplicate `add_intent` call would raise `RuntimeError`. The handler catches this and re-raises only if the intent is not already present in the container:

```python
except RuntimeError:
    if name not in self.containers[lang].registered_intents:
        raise
```

Source: `nebulento/opm.py:145-148`.

---

## Entity Registration

`nebulento/opm.py:150`

Follows the same pattern as intent registration: reads from `file_name` or inline `samples`, checks language, suppresses duplicate `RuntimeError` on skill reload.

---

## Skill Detach

`nebulento/opm.py:174-184`

`handle_detach_skill` removes all intents whose name starts with `skill_id` and all entities whose name starts with `skill_id:`. This matches the OVOS convention where both intents and entities are prefixed with the skill's ID.

---

## No-Training Handshake

`nebulento/opm.py:91`

When `mycroft.skills.train` is received, the plugin emits `mycroft.skills.trained` immediately without any computation. This satisfies the OVOS pipeline boot handshake without blocking.

---

## Intent Calculation

`nebulento/opm.py:188`

`NebulentoPipeline.calc_intent(utterances, lang, message)`:

1. Filters out utterances exceeding `max_words` tokens (default 50).
2. Resolves the closest matching language container via `langcodes.closest_match`.
3. Scores each utterance via the cached `_calc_nebulento_intent` function.
4. Returns the result with the highest confidence.

`_calc_nebulento_intent` is decorated with `@lru_cache(maxsize=128)` (`nebulento/opm.py:223`) to avoid re-scoring identical utterances across multiple ASR hypotheses in a single request. Because `IntentContainer` is mutable and `lru_cache` requires hashable arguments, the container itself is the cache key — this works because `IntentContainer` identity is stable per language after startup.

Session blacklists are applied after scoring:

```python
if result["name"] in sess.blacklisted_intents:
    return None
if result["name"].split(":")[0] in sess.blacklisted_skills:
    return None
```

Source: `nebulento/opm.py:231-234`.

---

## Confidence Tiers

The plugin implements three match levels by comparing `intent.conf` against a threshold:

| Method | Threshold config key | Default |
|---|---|---|
| `match_high` | `conf_high` | `0.95` |
| `match_medium` | `conf_med` | `0.80` |
| `match_low` | `conf_low` | `0.50` |

Source: `nebulento/opm.py:65-67, 110-117`.

The OVOS pipeline calls `match_high` first, then `match_medium`, then `match_low`, stopping at the first non-None result. Nebulento is typically placed in the pipeline after deterministic engines (Adapt, Padatious) but before generic fallbacks.

---

## `NebulentoIntent` Result Object

`nebulento/opm.py:21`

The pipeline returns `NebulentoIntent` instances (not raw dicts):

```python
class NebulentoIntent:
    name: str     # Matched intent name
    sent: str     # Input utterance
    conf: float   # Confidence in [0.0, 1.0]
    matches: dict # Entity name → extracted value
```

Supports `[]` access, `in` containment, and `.get()` against `matches`.

---

## mycroft.conf Configuration

```json
{
  "lang": "en-US",
  "secondary_langs": ["es-ES"],
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

See [Configuration](configuration.md) for the full key reference.

---

## Language Handling

`nebulento/opm.py:59-63`

The plugin creates one `IntentContainer` per language listed in the OVOS configuration. The primary `lang` is always included. Any language in `secondary_langs` that is not already in the list is added.

At query time, `langcodes.closest_match` is used to find the container whose BCP-47 tag is closest to the requested language. A match score ≥ 10 is treated as "no matching container" and returns `None`.

---

## Comparison with Padatious / Padacioso

| Property | Nebulento | Padacioso | Padatious |
|---|---|---|---|
| Matching approach | Fuzzy string similarity | Regex (+ optional fuzz) | Neural network |
| Training step | None | None | Required |
| Configurable strategy | Yes (9 options) | Fuzz on/off | No |
| False positives (benchmark) | 0–24/24 depending on strategy | 0–4/24 | 4/24 |
| Recall (benchmark, best) | 52.5% (`TOKEN_SET_RATIO`) | 48.0% | 50.4% |
| Spell-error handling | Yes (Damerau-Levenshtein) | Partial (with fuzz) | Limited |
| Entry point type | `opm.pipeline` | `opm.pipeline` | `opm.pipeline` |
| Bus events used | `padatious:*` | `padatious:*` | `padatious:*` |

Nebulento reuses the `padatious:register_intent` / `padatious:register_entity` bus events so it is a drop-in replacement for Padatious from the skill's perspective.

---

## Shutdown

`nebulento/opm.py:215`

`shutdown()` removes all four bus event handlers. Called by OVOS when the pipeline plugin is unloaded.
