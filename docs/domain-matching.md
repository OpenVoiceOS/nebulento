# Domain Matching

`DomainIntentContainer` (`nebulento/domain_engine.py:10`) provides a two-level hierarchical matching architecture. Intents are grouped into named *domains*. Matching proceeds in two stages:

1. **Domain classification** — the utterance is scored against a top-level `IntentContainer` that represents domains, not individual intents.
2. **Intent matching** — once a domain is selected, the utterance is scored only against intents registered within that domain.

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

This contrasts with `IntentContainer`, which scores the utterance against every registered intent globally. With large intent sets (hundreds of intents across many skills), domain-scoped matching limits the comparison set and can reduce false positive rates.

---

## When to Use `DomainIntentContainer`

Use it when:

- You have many intents spanning clearly separable topics (media, home automation, calendar, etc.).
- You want to reduce cross-domain false positives.
- The training templates in different domains contain overlapping vocabulary (e.g. "set" appears in both timer and thermostat intents).

Avoid it when:

- Intent boundaries do not map cleanly to domains.
- The domain classifier cannot be trained with representative utterances (it performs poorly with sparse training data).
- You have fewer than ~10 intents total — the overhead is unnecessary.

---

## Setup

### 1. Create the container

```python
from nebulento import DomainIntentContainer, MatchStrategy

d = DomainIntentContainer(fuzzy_strategy=MatchStrategy.TOKEN_SET_RATIO)
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

### 3. Register domain entities (optional)

```python
d.register_domain_entity("media", "song", ["jazz", "rock", "blues", "classical"])
d.register_domain_entity("home",  "temp", ["20 degrees", "22 degrees", "18"])
```

### 4. Train the domain classifier

The `domain_engine` is a plain `IntentContainer` where each intent name is a domain name. You must populate it with representative utterances so the top-level classification works:

```python
d.domain_engine.add_intent("media", [
    "play music", "next track", "pause", "put on some jazz",
    "skip this song", "turn up the volume",
])
d.domain_engine.add_intent("home", [
    "lights on", "thermostat", "lock the door", "turn off the lights",
    "set temperature", "open the blinds",
])
```

The `training_data` attribute accumulates all `intent_samples` passed to `register_domain_intent`. You can use this as a seed for the domain classifier instead of writing separate training data:

```python
for domain, samples in d.training_data.items():
    d.domain_engine.add_intent(domain, samples)
```

---

## Querying

### `calc_intent(query, domain=None)`

`nebulento/domain_engine.py:143`

```python
result = d.calc_intent("play some jazz")
print(result["name"])     # 'play'
print(result["conf"])     # float in [0, 1]
print(result["entities"]) # {'song': ['jazz']}
```

With explicit domain override (skips domain classification):

```python
result = d.calc_intent("turn off the lights", domain="home")
print(result["name"])  # 'lights_off'
```

### `calc_domain(query)`

`nebulento/domain_engine.py:128`

Run only the domain classification step:

```python
domain_match = d.calc_domain("play some jazz")
print(domain_match["name"])  # 'media'
print(domain_match["conf"])  # confidence for the domain match
```

---

## Worked Example

```python
from nebulento import DomainIntentContainer, MatchStrategy

d = DomainIntentContainer(fuzzy_strategy=MatchStrategy.DAMERAU_LEVENSHTEIN_SIMILARITY)

# Register intents
d.register_domain_intent("media", "play",      ["play {artist}", "put on {artist}"])
d.register_domain_intent("media", "pause",     ["pause", "stop"])
d.register_domain_intent("calendar", "add",    ["add event {title}", "schedule {title}"])
d.register_domain_intent("calendar", "check",  ["what's next", "what do I have today"])

# Register entities
d.register_domain_entity("media",    "artist", ["the beatles", "pink floyd", "bach"])
d.register_domain_entity("calendar", "title",  ["meeting", "dentist", "lunch"])

# Train domain classifier from accumulated data
for domain, samples in d.training_data.items():
    d.domain_engine.add_intent(domain, samples)

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

`nebulento/domain_engine.py:40-51`

```
DomainIntentContainer
├── domain_engine: IntentContainer       ← top-level classifier
├── domains: Dict[str, IntentContainer]  ← per-domain intent containers
└── training_data: Dict[str, List[str]]  ← raw samples per domain
```

Each per-domain `IntentContainer` is independent. Context gating, keyword exclusions, and entity registrations are scoped to the domain container where they are registered.
