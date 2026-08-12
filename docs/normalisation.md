# Normalisation

Normalisation is the first step in both template storage and utterance matching. It ensures that superficial textual differences (apostrophe variants, double spaces, case) do not prevent a correct fuzzy match. All normalisation utilities are in `nebulento/bracket_expansion.py`.

---

## Functions

### `normalize_utterance(text)`

`nebulento/bracket_expansion.py:86`

Normalise a plain query utterance before matching. Applied to every utterance by `IntentContainer._norm()`.

Steps:

1. Replace all apostrophe variants with a single space (`_drop_apostrophes`).
2. Collapse runs of whitespace to a single space and strip leading/trailing whitespace (`_normalize_whitespace`).

Does **not** touch entity placeholder syntax. There are no `{...}` tokens in a plain utterance.

```python
from nebulento.bracket_expansion import normalize_utterance

normalize_utterance("it's  fine")
# "it s fine"

normalize_utterance("  hello   world  ")
# "hello world"
```

---

### `normalize_example(example)`

`nebulento/bracket_expansion.py:69`

Normalise a training template for storage. Applied to each line before `expand_template` in `add_intent`.

Steps (in order):

1. `clean_braces(example)`: `{{entity}}` becomes `{entity}`
2. `translate_padatious(example)`: `:0` becomes `{word0}`, `{word1}`, and so on
3. `_drop_apostrophes(text)`: all apostrophe variants become a space
4. `_normalize_whitespace(text)`: collapse whitespace

Entity placeholders (`{name}`) are preserved through all steps.

```python
from nebulento.bracket_expansion import normalize_example

normalize_example("it's {{item}}  please")
# "it s {item} please"

normalize_example("set :0 timer")
# "set {word0} timer"
```

---

### `translate_padatious(example)`

`nebulento/bracket_expansion.py:45`

Convert Padatious `:0` word-slot tokens to numbered `{wordN}` entity placeholders. Numbering is per-line; each `:0` within a single call increments the counter.

```python
from nebulento.bracket_expansion import translate_padatious

translate_padatious("set a timer for :0 minutes")
# 'set a timer for {word0} minutes'

translate_padatious("play :0 by :0")
# 'play {word0} by {word1}'

# No-op when :0 is absent:
translate_padatious("hello world")
# 'hello world'
```

---

### `clean_braces(example)`

`nebulento/bracket_expansion.py:33`

Normalise accidental double-braces. This is a guard against template strings copied from Python f-string contexts where `{{` is the escaped form of `{`.

```python
from nebulento.bracket_expansion import clean_braces

clean_braces("buy {{item}} today")
# 'buy {item} today'
```

---

## Apostrophe Normalisation

`nebulento/bracket_expansion.py:9-19`

All eight apostrophe-like Unicode characters are replaced with a single ASCII space:

| Character | Unicode | Name |
|---|---|---|
| `'` | U+0027 | ASCII apostrophe |
| `'` | U+2019 | RIGHT SINGLE QUOTATION MARK |
| `'` | U+2018 | LEFT SINGLE QUOTATION MARK |
| `ʼ` | U+02BC | MODIFIER LETTER APOSTROPHE |
| `ʹ` | U+02B9 | MODIFIER LETTER PRIME |
| `` ` `` | U+0060 | GRAVE ACCENT |
| `´` | U+00B4 | ACUTE ACCENT |
| `＇` | U+FF07 | FULLWIDTH APOSTROPHE |

The effect is that contractions like "it's", "I'm", "don't" are split into two tokens: "it s", "i m", "don t". Both the training template and the utterance go through identical normalisation, so contractions match consistently regardless of which apostrophe variant the user typed or the STT produced.

```python
from nebulento.bracket_expansion import normalize_utterance

normalize_utterance("I don't want it")
# 'I don t want it'

normalize_utterance("it’s fine")   # RIGHT SINGLE QUOTATION MARK
# 'it s fine'
```

---

## Case Handling

`nebulento/container.py:61-69`: `IntentContainer._norm(text)`

`_norm` applies `normalize_utterance` and then optionally lowercases:

```python
def _norm(self, text: str) -> str:
    text = normalize_utterance(text)
    if self.ignore_case:
        text = text.lower()
    return text
```

`ignore_case=True` (the default) means all comparisons are case-insensitive. Template strings are stored lowercase; utterances are lowercased before comparison.

To preserve case sensitivity (e.g. for code or acronym matching):

```python
container = IntentContainer(ignore_case=False)
```

---

## Whitespace Collapsing

`nebulento/bracket_expansion.py:28-30`: `_normalize_whitespace(text)`

Any run of one or more whitespace characters (spaces, tabs, newlines) is collapsed to a single ASCII space, and leading/trailing whitespace is stripped.

```python
from nebulento.bracket_expansion import _normalize_whitespace

_normalize_whitespace("  hello   world\n")
# 'hello world'
```

This happens after apostrophe replacement, so the space introduced by `'` → ` ` is subject to collapse:

```python
normalize_utterance("it's  great")
# Step 1: "it s  great"  (apostrophe → space)
# Step 2: "it s great"   (double space collapsed)
```

---
[← Entity Extraction](entity-extraction.md) · [Home](index.md) · [OVOS Plugin →](ovos-plugin.md)
