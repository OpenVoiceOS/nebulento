# Hierarchical Matching

`HierarchicalIntentContainer` (`nebulento/hierarchical.py:10`) provides a two-stage matching architecture. Intents are grouped into named *domains*. Matching proceeds in two stages:

1. **Domain classification**: the utterance is scored against a top-level `IntentContainer` that represents domains, not individual intents.
2. **Intent matching**: once a domain is selected, the utterance is scored only against intents registered within that domain.

---

## Architecture

```
utterance
    │
    ▼
domain_engine.calc_intent(query)
    │
    │  returns domain name, e.g. "media"
    ▼
domains["media"].calc_intent(query)
    │
    ▼
MatchResult with intent name, conf, entities
```

This contrasts with `IntentContainer`, which scores the utterance against every registered intent globally. With large intent sets (hundreds of intents across many skills), domain-scoped matching limits the comparison set and isolates each domain's vocabulary.

### Domain vs hierarchical

Two-stage routing is one of two ways to organise intents by topic:

- **Domain (parallel)**: every domain is scored independently and the global best wins. No domain is excluded up front.
- **Hierarchical (two-stage)**: a top-level classifier picks one domain, and only that domain's intents are scored.

`HierarchicalIntentContainer` is the two-stage variant, named to match Adapt's `HierarchicalIntentDeterminationEngine` and palavreado's `HierarchicalIntentContainer`. nebulento does not ship a separate parallel-Domain container. Its flat `IntentContainer` already scores every intent independently, so a parallel-domain grouping would behave identically to it.

---

## When to Use `HierarchicalIntentContainer`

Use it when:

- You have many intents spanning clearly separable topics (media, home automation, calendar, etc.).
- The training templates in different domains contain overlapping vocabulary. For example, "set" appears in both timer and thermostat intents. Scoping to one domain stops cross-domain bleed.
- You want a confidence gate that rejects utterances no domain recognises (see [Off-topic rejection](#off-topic-rejection)).

Avoid it when:

- Intent boundaries do not map cleanly to domains.
- Each domain has too few samples for the top-level classifier to learn from. Sparse domains classify poorly.
- You have fewer than ~10 intents total. The overhead is unnecessary.

---

## Setup

### 1. Create the container

```python
from nebulento import HierarchicalIntentContainer, MatchStrategy

d = HierarchicalIntentContainer(fuzzy_strategy=MatchStrategy.TOKEN_SET_RATIO)
```

### 2. Register domain intents

```python
d.register_domain_intent("media", "play",  ["play {song}", "put on {song}", "start {song}"])
d.register_domain_intent("media", "pause", ["pause", "stop the music", "hold on"])
d.register_domain_intent("media", "next",  ["next track", "skip this", "next song"])

d.register_domain_intent("home", "lights_on",  ["lights on", "turn on the lights", "switch on lights"])
d.register_domain_intent("home", "lights_off", ["lights off", "turn off the lights"])
d.register_domain_intent("home", "thermostat", ["set thermostat to {temp}", "change temperature to {temp}"])
```

The top-level domain classifier is **trained automatically**: every sample passed to `register_domain_intent` is also fed to `domain_engine` under its domain name (`nebulento/hierarchical.py:90`). There is no separate training step. The container works standalone as soon as intents are registered.

### 3. Register domain entities (optional)

```python
d.register_domain_entity("media", "song", ["jazz", "rock", "blues", "classical"])
d.register_domain_entity("home",  "temp", ["20 degrees", "22 degrees", "18"])
```

---

## Querying

### `calc_intent(query, domain=None)`

`nebulento/hierarchical.py:163`

```python
result = d.calc_intent("play some jazz")
print(result["name"])     # 'play'
print(result["conf"])     # float in [0, 1]
print(result["entities"]) # {'song': ['jazz']}
```

With explicit domain override (skips domain classification and the threshold gate):

```python
result = d.calc_intent("turn off the lights", domain="home")
print(result["name"])  # 'lights_off'
```

### `calc_domain(query)`

`nebulento/hierarchical.py:151`

Run only the domain classification step:

```python
domain_match = d.calc_domain("play some jazz")
print(domain_match["name"])  # 'media'
print(domain_match["conf"])  # confidence for the domain match
```

---

## Off-topic rejection

By default (`domain_threshold=0.0`) every query is routed to its best-scoring domain. There is no rejection at the domain stage, and a confident intent match still depends on the per-intent `conf`.

Set `domain_threshold` to reject utterances no domain recognises well:

```python
d = HierarchicalIntentContainer(domain_threshold=0.6)
```

When the top-level classifier's best domain scores below the threshold, `calc_intent` returns a no-match (`name=None`) without resolving any intent. This trades recall for precision. A real command whose domain is misclassified becomes unrecoverable, but off-topic speech that shares words with a domain is filtered out.

The right threshold depends on the `MatchStrategy`. Lenient strategies such as `TOKEN_SET_RATIO` score loosely and need a higher gate. Strict strategies such as `DAMERAU_LEVENSHTEIN_SIMILARITY` already reject off-topic input via low `conf` and gain little from the gate. See [Benchmark](benchmark.md) for measured precision/recall trade-offs.

---

## Worked Example

```python
from nebulento import HierarchicalIntentContainer, MatchStrategy

d = HierarchicalIntentContainer(fuzzy_strategy=MatchStrategy.DAMERAU_LEVENSHTEIN_SIMILARITY)

# Register intents. The domain classifier is trained automatically
d.register_domain_intent("media", "play",      ["play {artist}", "put on {artist}"])
d.register_domain_intent("media", "pause",     ["pause", "stop"])
d.register_domain_intent("calendar", "add",    ["add event {title}", "schedule {title}"])
d.register_domain_intent("calendar", "check",  ["what's next", "what do I have today"])

# Register entities
d.register_domain_entity("media",    "artist", ["the beatles", "pink floyd", "bach"])
d.register_domain_entity("calendar", "title",  ["meeting", "dentist", "lunch"])

# Query
result = d.calc_intent("put on some pink floyd")
print(result)
# {
#   'name': 'play',
#   'conf': 0.78,
#   'entities': {'artist': ['pink floyd']},
#   'best_match': 'put on {artist}',
#   'utterance': 'put on some pink floyd',
#   ...
# }

result = d.calc_intent("add dentist to my calendar")
print(result["name"])  # 'add'
```

---

## Removing Domains and Intents

```python
# Remove a specific intent from a domain:
d.remove_domain_intent("media", "pause")

# Remove a specific entity from a domain:
d.remove_domain_entity("media", "artist")

# Remove an entire domain (all intents, entities, training data):
d.remove_domain("media")
```

After removing a domain via `remove_domain`, the domain name is also removed from `domain_engine.registered_intents`, so it will no longer be selected during classification.

---

## Internal Structure

`nebulento/hierarchical.py:47-59`

```
HierarchicalIntentContainer
├── domain_engine: IntentContainer       ← top-level classifier
├── domains: Dict[str, IntentContainer]  ← per-domain intent containers
└── training_data: Dict[str, List[str]]  ← raw samples per domain
```

Each per-domain `IntentContainer` is independent. Context gating, keyword exclusions, and entity registrations are scoped to the domain container where they are registered.

---
[← OVOS Pipeline Plugin](ovos-plugin.md) · [Home](index.md) · [Configuration →](configuration.md)
