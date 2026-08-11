# Quick Start

This guide shows the most common patterns in under 5 minutes.

## 1. Create a Container

```python
from nebulento import IntentContainer, MatchStrategy

container = IntentContainer(fuzzy_strategy=MatchStrategy.TOKEN_SET_RATIO)
```

The default strategy is `DAMERAU_LEVENSHTEIN_SIMILARITY`. `TOKEN_SET_RATIO` is used here for higher recall. See [Match Strategies](strategies.md) for the trade-offs.

## 2. Register Intents

```python
container.add_intent("hello", [
    "hello",
    "hi",
    "hey there",
    "how are you",
    "(good|hey) morning",
])

container.add_intent("goodbye", [
    "goodbye",
    "bye",
    "see you later",
    "take care",
])
```

Templates support:

- `(a|b|c)`: alternation. Expands to one variant per combination at registration time.
- `[word]`: optional word or phrase (equivalent to `(word|)`)
- `{entity}`: capture slot matched against registered entity samples

All variants are expanded and stored immediately. There is no separate training step.

## 3. Register Entities

```python
container.add_entity("item", ["milk", "eggs", "bread", "cheese"])

container.add_intent("buy", [
    "buy {item}",
    "purchase {item}",
    "get me some {item}",
    "I need {item}",
])
```

When a registered entity value appears in the utterance the confidence is boosted and the value is returned under `entities` in the result.

## 4. Match an Utterance

```python
result = container.calc_intent("hello")
print(result)
# {
#   'name': 'hello',
#   'conf': 1.0,
#   'entities': {},
#   'best_match': 'hello',
#   'utterance': 'hello',
#   'utterance_consumed': 'hello',
#   'utterance_remainder': '',
#   'match_strategy': 'TOKEN_SET_RATIO'
# }
```

For an entity-bearing intent:

```python
result = container.calc_intent("I'd like to buy some milk")
print(result["name"])    # 'buy'
print(result["conf"])    # ~0.7
print(result["entities"])  # {'item': ['milk']}
```

## 5. Iterate All Scores

`calc_intents` yields one result per registered intent, useful for debugging or custom ranking:

```python
for r in container.calc_intents("good morning"):
    print(r["name"], r["conf"])
# hello  0.857
# goodbye  0.222
# buy  0.143
```

## 6. Choosing a Strategy

```python
# High recall, more false positives:
c1 = IntentContainer(fuzzy_strategy=MatchStrategy.TOKEN_SET_RATIO)

# Zero false positives on benchmark dataset, lower recall:
c2 = IntentContainer(fuzzy_strategy=MatchStrategy.DAMERAU_LEVENSHTEIN_SIMILARITY)

# Balanced starting point:
c3 = IntentContainer(fuzzy_strategy=MatchStrategy.SIMPLE_RATIO)
```

See [Match Strategies](strategies.md) for a full decision table.

## 7. Context Gating

Restrict an intent so it only fires when a condition is active:

```python
container.require_context("goodbye", "conversation_started")

# Before context is set, intent is suppressed:
result = container.calc_intent("goodbye")
print(result["name"])  # None

# Set the context:
container.set_context("goodbye", "conversation_started")

# Now it matches:
result = container.calc_intent("goodbye")
print(result["name"])  # 'goodbye'
```

## 8. Hierarchical Matching

For larger sets of intents, use `HierarchicalIntentContainer` to classify the
domain first and limit the search to that domain's intents:

```python
from nebulento import HierarchicalIntentContainer

d = HierarchicalIntentContainer()
d.register_domain_intent("media", "play", ["play {song}", "put on {song}"])
d.register_domain_intent("home",  "lights_on", ["lights on", "turn on the lights"])

# the domain classifier is trained automatically, no extra step needed
result = d.calc_intent("turn on the lights please")
print(result["name"])  # 'lights_on'
```

See [Hierarchical Matching](hierarchical-matching.md) for a detailed guide.

---
[← Installation](installation.md) · [Home](index.md) · [Intent API →](intent-api.md)
