# Intent API Reference

## `IntentContainer`

Defined in `nebulento/container.py:17`.

Fuzzy intent matcher backed by rapidfuzz similarity strategies. Registers intent templates and entity samples, then scores an utterance against every registered intent.

### Constructor

```python
IntentContainer(
    fuzzy_strategy: MatchStrategy = MatchStrategy.DAMERAU_LEVENSHTEIN_SIMILARITY,
    ignore_case: bool = True,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `fuzzy_strategy` | `MatchStrategy` | `DAMERAU_LEVENSHTEIN_SIMILARITY` | Similarity algorithm used for all comparisons. See [Match Strategies](strategies.md). |
| `ignore_case` | `bool` | `True` | When `True`, utterances and templates are lowercased before comparison. |

---

### Intent Registration

#### `add_intent(name, lines)`

`nebulento/container.py:74`

Register an intent with one or more training templates.

Templates support `(a|b)` alternation, `[optional]` syntax, and `{entity}` capture placeholders. All variants are expanded at registration time via `expand_template(normalize_example(line))` and duplicates are discarded.

```python
container.add_intent(
    name="play",
    lines=[
        "play {song}",
        "(put on|start) [some] {song}",
        "I want to listen to {song}",
    ]
)
```

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | Unique intent identifier. Convention for OVOS skills: `"skill_id:intent_name"`. |
| `lines` | `List[str]` | Training templates. May use full template syntax. |

Raises `RuntimeError` if `name` is already registered. Call `remove_intent` first if re-registration is needed.

---

#### `remove_intent(name)`

`nebulento/container.py:98`

Unregister an intent. Silently does nothing if `name` is not registered.

```python
container.remove_intent("play")
```

---

#### `intent_names` (property)

`nebulento/container.py:55`

Read-only list of all currently registered intent names.

```python
names = container.intent_names
# ['hello', 'goodbye', 'play']
```

---

### Entity Registration

#### `add_entity(name, lines)`

`nebulento/container.py:106`

Register an entity with sample values. When a template containing `{name}` is the best match and a registered sample value appears in the utterance, the confidence score is boosted and the value is included in `entities`.

Entity names are stored lowercase regardless of input casing.

```python
container.add_entity("song", ["jazz", "rock", "classical music", "the blues"])
```

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | Entity name. Case-insensitive; stored lowercase. |
| `lines` | `List[str]` | Sample values. Alternation syntax `(a\|b)` is expanded. |

Raises `RuntimeError` if `name` is already registered.

---

#### `remove_entity(name)`

`nebulento/container.py:129`

Unregister an entity. Case-insensitive. Silently does nothing if not registered.

```python
container.remove_entity("song")
```

---

#### `registered_entities` (attribute)

`nebulento/container.py:46`

Dict mapping entity name (lowercase) → list of expanded sample strings.

```python
samples = container.registered_entities["song"]
```

---

### Matching

#### `calc_intent(query) -> MatchResult`

`nebulento/container.py:341`

Return the single best-matching intent for `query`.

Tie-breaking order when multiple intents share the highest score:

1. Higher `conf` score.
2. More words consumed (more specific template).
3. Intent name alphabetically (deterministic tie-break).

When no intents are registered or all are suppressed by context/keyword gates, returns a result dict with `name=None` and all other keys present with empty/zero values.

```python
result = container.calc_intent("play some jazz")
```

| Parameter | Type | Description |
|---|---|---|
| `query` | `str` | Raw utterance to evaluate. |

**Returns** `MatchResult` — a `Dict[str, object]` with the following keys:

| Key | Type | Description |
|---|---|---|
| `name` | `str \| None` | Matched intent name. `None` if no match. |
| `conf` | `float` | Confidence score in `[0.0, 1.0]`. |
| `entities` | `Dict[str, List[str]]` | Entity name → list of matched values (lowercase keys). |
| `best_match` | `str \| None` | Training template that scored highest. |
| `utterance` | `str` | Normalised input query. |
| `utterance_consumed` | `str` | Space-joined words accounted for by the match. |
| `utterance_remainder` | `str` | Space-joined words not accounted for. |
| `match_strategy` | `str` | Name of the `MatchStrategy` enum value used. |

```python
result = container.calc_intent("play some jazz")
# {
#   'name': 'play',
#   'conf': 0.752,
#   'entities': {'song': ['jazz']},
#   'best_match': 'play {song}',
#   'utterance': 'play some jazz',
#   'utterance_consumed': 'play jazz',
#   'utterance_remainder': 'some',
#   'match_strategy': 'DAMERAU_LEVENSHTEIN_SIMILARITY'
# }
```

---

#### `calc_intents(query) -> Iterator[MatchResult]`

`nebulento/container.py:327`

Yield a scored `MatchResult` for every registered intent that is not suppressed by context or keyword gates. Results are in registration order.

Useful for debugging confidence distributions or implementing custom ranking on top of the raw scores.

```python
for r in container.calc_intents("play some jazz"):
    print(f"{r['name']:20s} conf={r['conf']:.3f}")
```

---

### Context Gating

Context gating allows intents to be conditionally suppressed based on named application state. The context system has two axes:

- **Required contexts** — the intent only fires when all required contexts are active.
- **Excluded contexts** — the intent is suppressed when any excluded context is active.

Contexts are stored per intent name. Multiple requirements or exclusions can be stacked.

#### `set_context(intent_name, context_name, context_val=None)`

`nebulento/container.py:139`

Mark a context as active for `intent_name`.

```python
container.set_context("confirm_delete", "delete_pending")
container.set_context("confirm_delete", "user_id", "alice")
```

| Parameter | Type | Description |
|---|---|---|
| `intent_name` | `str` | Intent whose context availability is updated. |
| `context_name` | `str` | Context key. |
| `context_val` | `object` | Optional associated value. Not used by the matcher itself. |

---

#### `unset_context(intent_name, context_name)`

`nebulento/container.py:150`

Remove an active context for `intent_name`.

```python
container.unset_context("confirm_delete", "delete_pending")
```

---

#### `require_context(intent_name, context_name)`

`nebulento/container.py:160`

Gate `intent_name` so it only matches when `context_name` is active. Multiple calls accumulate — all named contexts must be satisfied simultaneously.

```python
container.require_context("confirm_delete", "delete_pending")
container.require_context("confirm_delete", "user_authenticated")
```

---

#### `unrequire_context(intent_name, context_name)`

`nebulento/container.py:171`

Remove a context requirement from `intent_name`. Other requirements are unaffected.

```python
container.unrequire_context("confirm_delete", "user_authenticated")
```

---

#### `exclude_context(intent_name, context_name)`

`nebulento/container.py:183`

Suppress `intent_name` whenever `context_name` is active.

```python
container.exclude_context("help", "expert_mode")
```

---

#### `unexclude_context(intent_name, context_name)`

`nebulento/container.py:193`

Lift a context-based suppression from `intent_name`.

```python
container.unexclude_context("help", "expert_mode")
```

---

### Keyword Exclusion

#### `exclude_keywords(intent_name, samples)`

`nebulento/container.py:204`

Suppress `intent_name` when any keyword in `samples` appears in the query.

Single-word keywords use whole-word matching against the tokenised query. Multi-word keywords use a word-boundary regex (`\b`) against the lowercased query string. Keyword matching is case-insensitive.

```python
container.add_intent("play_music", ["play some music", "start the music"])
container.exclude_keywords("play_music", ["video", "movie", "film"])

# Suppressed — "video" appears:
result = container.calc_intent("play a video")
print(result["name"])  # some other intent or None

# Not suppressed:
result = container.calc_intent("play some music")
print(result["name"])  # 'play_music'
```

| Parameter | Type | Description |
|---|---|---|
| `intent_name` | `str` | Intent to suppress. |
| `samples` | `List[str]` | Keywords or phrases that block the intent. |

---

### Internal Methods

These are documented for completeness; prefer the public API above.

#### `match_entities(sentence) -> Dict[str, List[str]]`

`nebulento/container.py:251`

Find registered entity values present in `sentence` using `quebra_frases.chunk` for token-aware substring detection. Returns a dict mapping entity name → list of matched sample values found in the sentence.

#### `match_fuzzy(sentence) -> Iterator[MatchResult]`

`nebulento/container.py:269`

Core scoring loop. Yields one result dict per registered intent (excluding suppressed ones). Calls `match_one` for the best template score, applies entity confidence boost (`score = 0.25 + score * 0.75` when an entity slot is filled), and constructs the full result dict.

---

## `HierarchicalIntentContainer`

Defined in `nebulento/hierarchical.py:10`.

Two-stage intent engine: domain classification followed by intent matching. Intents are grouped into named domains. At query time the engine first selects the most likely domain, then runs the domain-specific `IntentContainer`.

See [Hierarchical Matching](hierarchical-matching.md) for a full guide.

### Constructor

```python
HierarchicalIntentContainer(
    fuzzy_strategy: MatchStrategy = MatchStrategy.DAMERAU_LEVENSHTEIN_SIMILARITY,
    ignore_case: bool = True,
    domain_threshold: float = 0.0,
)
```

`fuzzy_strategy` and `ignore_case` are forwarded to every `IntentContainer` created internally, including `domain_engine`. `domain_threshold` is the minimum confidence the top-level classifier must reach for a query to be routed at all — below it `calc_intent` returns a no-match. `0.0` (default) disables the gate.

### Public Attributes

| Attribute | Type | Description |
|---|---|---|
| `domain_engine` | `IntentContainer` | Top-level classifier mapping queries to domain names. |
| `domains` | `Dict[str, IntentContainer]` | Per-domain containers keyed by domain name. |
| `training_data` | `Dict[str, List[str]]` | Raw training samples per domain (accumulated at registration). |
| `domain_threshold` | `float` | Minimum classifier confidence to route a query; `0.0` disables the gate. |

---

### Domain Management

#### `remove_domain(domain_name)`

`nebulento/hierarchical.py:77`

Remove a domain and all its intents, entities, and training data.

```python
d.remove_domain("media")
```

---

### Intent Management

#### `register_domain_intent(domain_name, intent_name, intent_samples)`

`nebulento/hierarchical.py:90`

Register an intent inside a domain. Creates the domain's `IntentContainer` on first use, accumulates `intent_samples` into `training_data[domain_name]`, and retrains the top-level domain classifier — no separate classifier setup is needed.

```python
d.register_domain_intent("media", "play", ["play {song}", "put on {song}"])
d.register_domain_intent("media", "pause", ["pause", "stop the music"])
```

| Parameter | Type | Description |
|---|---|---|
| `domain_name` | `str` | Target domain. Created if it does not exist. |
| `intent_name` | `str` | Unique intent name within the domain. |
| `intent_samples` | `List[str]` | Training templates. |

---

#### `remove_domain_intent(domain_name, intent_name)`

`nebulento/hierarchical.py:110`

Remove a specific intent from a domain. Silently does nothing if domain or intent does not exist.

---

### Entity Management

#### `register_domain_entity(domain_name, entity_name, entity_samples)`

`nebulento/hierarchical.py:122`

Register an entity inside a domain. Creates the domain's container on first use.

```python
d.register_domain_entity("media", "song", ["jazz", "rock", "blues"])
```

---

#### `remove_domain_entity(domain_name, entity_name)`

`nebulento/hierarchical.py:139`

Remove a specific entity from a domain.

---

### Query API

#### `calc_domain(query) -> MatchResult`

`nebulento/hierarchical.py:151`

Classify `query` into the best-matching domain using `domain_engine.calc_intent`. The `domain_engine` is trained automatically as intents are registered with `register_domain_intent`.

```python
match = d.calc_domain("play some jazz")
print(match["name"])  # 'media'
```

---

#### `calc_intent(query, domain=None) -> MatchResult`

`nebulento/hierarchical.py:163`

Return the best-matching intent for `query`.

If `domain` is `None`, the domain is inferred by calling `calc_domain`; when that domain scores below `domain_threshold`, a no-match result is returned. If the resolved or supplied domain has no registered intents, a no-match result is returned. Passing `domain` explicitly bypasses both the classifier and the threshold gate.

```python
# Auto-infer domain:
result = d.calc_intent("play some jazz")

# Force a specific domain:
result = d.calc_intent("play some jazz", domain="media")
```

| Parameter | Type | Description |
|---|---|---|
| `query` | `str` | Raw utterance to evaluate. |
| `domain` | `str \| None` | Domain to restrict matching to. `None` triggers automatic domain classification. |

Returns a `MatchResult` dict (same structure as `IntentContainer.calc_intent`). `name` is `None` when no domain or intent matched.
