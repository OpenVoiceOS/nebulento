# Entity Extraction

Entities in Nebulento are named value sets. Registering an entity gives the matcher a vocabulary to look for inside the utterance. When a value from that vocabulary appears, two things happen: the matched entity value is included in the result, and the confidence score is boosted.

---

## Registration

### `add_entity(name, lines)`

`nebulento/container.py:106`

```python
container.add_entity("colour", ["red", "green", "blue", "dark blue", "light green"])
container.add_entity("room",   ["kitchen", "living room", "bedroom", "bathroom"])
```

`lines` is a list of sample strings. Alternation syntax `(a|b)` is expanded at registration time:

```python
container.add_entity("direction", ["(turn|go) left", "(turn|go) right"])
# stored samples: ['go left', 'go right', 'turn left', 'turn right']
```

Entity names are stored **lowercase** regardless of input:

```python
container.add_entity("COLOUR", ["red"])
# stored as: container.registered_entities["colour"]
```

Raises `RuntimeError` if the entity name (after lowercasing) is already registered.

---

## Detection During Matching

`nebulento/container.py:251` — `match_entities(sentence)`

Before scoring intents, `match_fuzzy` calls `match_entities` which scans the normalised utterance for registered entity samples using `quebra_frases.chunk`:

```python
entities = self.match_entities(sentence)
# {'colour': ['red'], 'room': [], 'direction': []}
```

`quebra_frases.chunk` performs token-aware substring detection — it finds multi-word samples while respecting token boundaries so "green" does not match inside "evergreen".

---

## Confidence Boost

`nebulento/container.py:307-311`

After the best training template is found for an intent, the code checks whether:

1. The template contains `{entity_name}` (the slot is declared in this intent), **and**
2. The entity detection found at least one value in the utterance.

If both conditions are true:

```python
score = 0.25 + score * 0.75
```

This formula always increases the score (assuming it was below 1.0). A fuzzy score of 0.6 on the template match becomes `0.25 + 0.6 × 0.75 = 0.70`. A score of 0.9 becomes `0.25 + 0.9 × 0.75 = 0.925`. The result is capped at 1.0.

The boost only applies when the specific entity slot appears in the intent's templates. An entity registered under a different name does not boost this intent even if the value is present.

---

## Entity Values in Results

`nebulento/container.py:308-311`

Matched entity values are collected into the `entities` dict of the result. Keys are entity names lowercased. Values are lists (a single entity slot can match multiple registered values if several appear in the utterance):

```python
container.add_intent("lights", ["turn {colour} lights on in the {room}"])
container.add_entity("colour", ["red", "green", "blue"])
container.add_entity("room", ["kitchen", "bedroom"])

result = container.calc_intent("turn red lights on in the kitchen")
print(result["entities"])
# {'colour': ['red'], 'room': ['kitchen']}
```

Only entities whose slot name appears in `{...}` in the matched intent's templates are included in the result. The entity scan runs over all registered entities, but only matching ones are promoted to `tagged`.

---

## `utterance_consumed` and `utterance_remainder`

`nebulento/container.py:301-311`

After entity matching, the words from matched entity values are added to `utterance_consumed` and removed from `utterance_remainder`:

```python
result = container.calc_intent("turn red lights on in the kitchen")
print(result["utterance_consumed"])   # 'turn on in the red kitchen'
print(result["utterance_remainder"])  # 'lights'
```

The consumed/remainder split is word-based (via `quebra_frases.word_tokenize`), not position-based.

---

## `expand_slots` Utility

`nebulento/bracket_expansion.py:148`

If you want to pre-generate all concrete training strings with entity values substituted in, use `expand_slots`:

```python
from nebulento.bracket_expansion import expand_slots

expand_slots("turn {colour} lights on", {
    "colour": ["red", "green", "blue"],
})
# ['turn blue lights on', 'turn green lights on', 'turn red lights on']
```

This is useful for inspecting what the fully-expanded training set looks like, or for feeding data to a different engine.

---

## Key-Lowercasing in Results

`nebulento/container.py:313`

The entity keys in `result["entities"]` are always lowercase, regardless of how the entity was registered:

```python
container.add_entity("COLOUR", ["red", "green"])
result = container.calc_intent("turn red lights on")
print(result["entities"])
# {'colour': ['red']}   ← lowercase key
```

---

## No Entity Registered — Slot Left Unmatched

If a template contains `{slot}` but no entity named `slot` is registered, the slot is simply not filled. The confidence is not boosted and `entities` does not include that key:

```python
container.add_intent("buy", ["buy {item}"])
# No add_entity("item", ...) call

result = container.calc_intent("buy milk")
print(result["entities"])  # {}  — slot present in template but no entity registered
```

The fuzzy match still runs against `"buy {item}"` as a literal template string. This works but the `{item}` substring reduces the similarity score compared to a template with the literal word.
