# Troubleshooting

---

## False Positives with `partial-*` Strategies

**Symptom:** Unrelated utterances match an intent with high confidence.

**Cause:** `PARTIAL_RATIO`, `PARTIAL_TOKEN_RATIO`, `PARTIAL_TOKEN_SORT_RATIO`, and `PARTIAL_TOKEN_SET_RATIO` score the shorter string as a substring of the longer string. A short training template like `"play"` will score 1.0 against any utterance containing the word "play", including `"display something"` or `"replay the video"`.

**Fix:** Use a non-partial strategy. For production use without a downstream confidence filter, use `DAMERAU_LEVENSHTEIN_SIMILARITY` (default). If recall is the priority, use `TOKEN_SET_RATIO` and set `conf_high`/`conf_med` thresholds aggressively. Never use `partial-*` strategies as the sole gating mechanism.

See [Match Strategies](strategies.md#decision-table) for the decision table.

---

## Low Recall with `DAMERAU_LEVENSHTEIN_SIMILARITY`

**Symptom:** Intent not matched even when the utterance is clearly related to a registered intent.

**Cause:** Damerau-Levenshtein measures character-level edit distance. It does not understand synonyms or paraphrases. An utterance like "chuck some music on" scores poorly against templates like "play {song}" because the words are different, not misspelled.

**Fix:** Add more diverse training templates that cover the phrasing your users actually produce:

```python
container.add_intent("play_music", [
    "play {song}",
    "put on {song}",
    "chuck on {song}",
    "stick {song} on",
    "can you play {song}",
    "I want to hear {song}",
    "start {song}",
])
```

Alternatively, switch to `TOKEN_SET_RATIO` if false positives can be tolerated and a downstream confidence gate is applied.

---

## Duplicate Registration RuntimeError

**Symptom:** `RuntimeError: Intent already registered: 'my_intent'. Call remove_intent first.`

**Cause:** `add_intent` (and `add_entity`) raise `RuntimeError` when called with a name that is already registered. This can happen when a skill is reloaded without the old intent being detached first, or when `add_intent` is called twice in the same code path.

**Fix (standalone library):**

```python
if "my_intent" in container.intent_names:
    container.remove_intent("my_intent")
container.add_intent("my_intent", samples)
```

**Fix (OVOS plugin):** The `NebulentoPipeline.register_intent` handler already catches this case and suppresses the error if the intent is present in the container (`nebulento/opm.py:145-148`). If you are seeing the error from the plugin, it means the intent was registered by a different container, or the `registered_intents` tracking list is out of sync. File a bug.

---

## Entity Not Extracted

**Symptom:** `result["entities"]` is empty even though the utterance contains a registered entity value.

**Possible causes and fixes:**

1. **Entity name does not match slot name.** The slot in the intent template must exactly match the entity name (case-insensitive after lowercasing):

    ```python
    container.add_intent("buy", ["buy {item}"])   # slot is "item"
    container.add_entity("item", ["milk", "eggs"]) # name must be "item"
    ```

2. **Entity value not in registered samples.** `match_entities` only finds values that were registered. If "oat milk" was not in the entity samples, it will not be extracted even if "milk" was.

3. **Entity value present but template has no `{slot}` for it.** Entity confidence boost and extraction only apply when the best-matched training template contains the `{entity_name}` slot. If the highest-scoring template for that intent does not have the slot, the entity is not included.

4. **quebra_frases tokenisation mismatch.** `match_entities` uses `quebra_frases.chunk` for substring detection. Multi-word entity values require the exact token sequence to be present in the utterance. Check that normalisation (apostrophe dropping, lowercasing) has not split or altered the entity value.

5. **Entity registered after intent is already matched.** Entities must be registered before `calc_intent` is called. The matching pipeline uses whatever is registered at call time.

---

## Language Mismatch in OVOS Plugin

**Symptom:** Skills register intents successfully but `calc_intent` always returns `None`.

**Cause:** The intent was registered with a `lang` value that does not appear in `self.containers`. This happens when `secondary_langs` in `mycroft.conf` does not include the skill's language, or when the BCP-47 tags differ in format (e.g. `"en-US"` vs `"en_US"`).

**Fix:** Ensure `secondary_langs` in `mycroft.conf` lists all languages your skills use:

```json
{
  "lang": "en-US",
  "secondary_langs": ["es-ES"]
}
```

The plugin normalises tags via `ovos_utils.lang.standardize_lang_tag` and uses `langcodes.closest_match` at query time with a score threshold of 10. Tags that are too distant are rejected. Use the canonical BCP-47 form (e.g. `"en-US"` not `"english"`).

---

## `lru_cache` and Mutable Containers

**Symptom (OVOS plugin):** After registering new intents at runtime (e.g. dynamic skill loading), `calc_intent` returns stale results that do not reflect the new intents.

**Cause:** `_calc_nebulento_intent` is decorated with `@lru_cache(maxsize=128)` (`nebulento/opm.py:223`). The cache key includes the `IntentContainer` object. If the same utterance is queried again before the cache evicts the entry, the cached result is returned even though new intents were registered.

**Cause explained:** The cache is designed for burst ASR hypothesis deduplication within a single request. Across requests, LRU eviction (128 entries) and the variety of utterances means stale hits are rare in practice.

**Fix if this is a problem:** Call `_calc_nebulento_intent.cache_clear()` after bulk intent registration. This is not exposed as a public API. It is an implementation detail.

---

## `utterance_remainder` Contains Expected Words

**Symptom:** Words that should match are appearing in `utterance_remainder` instead of `utterance_consumed`.

**Cause:** Consumed/remainder classification is word-based using `quebra_frases.word_tokenize` and set membership (`if w not in sent_tokens`). Words in the utterance that are not also in the best-matching training template are classified as remainder, even if they are semantically part of the intent.

This is expected behaviour. The remainder field indicates words not covered by the winning template. Use it as a signal that the templates may need to be broadened, not as a bug indicator.

---

## Context Gate Not Suppressing Intent

**Symptom:** An intent fires even though its required context is not active.

**Cause:** `require_context` and `exclude_context` operate per intent name and per context key. The check is:

```python
avail = self.available_contexts.get(intent_name, {})
if any(c not in avail for c in contexts):
    excluded.append(intent_name)
```

Common mistakes:

- Calling `require_context("my_intent", "ctx")` but checking `calc_intent` before `set_context("my_intent", "ctx")` is called. This should suppress correctly.
- Using the wrong `intent_name` key. Must match the name used in `add_intent` exactly.
- Forgetting that `available_contexts` is keyed by intent name, not globally. Setting a context for one intent does not affect other intents.

---
[← Benchmark](benchmark.md) · [Home](index.md)
