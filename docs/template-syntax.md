# Template Syntax

Templates are strings passed to `add_intent` and `add_entity`. They are processed at registration time by functions in `nebulento/bracket_expansion.py`.

---

## Processing Pipeline

When `add_intent(name, lines)` is called, each line is processed as follows:

```
raw template string
        │
        ▼
  normalize_example()
   ├── clean_braces()       {{entity}} → {entity}
   ├── translate_padatious() :0 → {word0}
   ├── _drop_apostrophes()  ' → space (all Unicode variants)
   └── _normalize_whitespace() collapse runs, strip ends
        │
        ▼
  expand_template()
   ├── expand_optional()    [word] → (word|)
   └── fully_expand()       (a|b) × (c|d) → {ac, ad, bc, bd}
        │
        ▼
  _norm() per variant      lowercase if ignore_case=True
        │
        ▼
  set of normalised concrete strings stored in registered_intents[name]
```

Source: `nebulento/container.py:91-96`, `nebulento/bracket_expansion.py`.

---

## Alternation: `(a|b|c)`

`nebulento/bracket_expansion.py:123`

A parenthesised group of pipe-separated alternatives expands to one variant per combination. Groups can appear anywhere in the template and multiple groups in a single template produce a Cartesian product.

```python
from nebulento.bracket_expansion import expand_template

expand_template("(play|start) music")
# ['play music', 'start music']

expand_template("(turn|switch) (on|off) the lights")
# ['switch off the lights', 'switch on the lights',
#  'turn off the lights', 'turn on the lights']
```

Nesting is handled by repeated application of the expansion loop until the result stabilises (`fully_expand`). Empty alternatives are valid: `(a|)` expands to `['', 'a']` (i.e. the phrase with and without that word).

---

## Optional Words: `[word]`

`nebulento/bracket_expansion.py:120`

Square brackets mark optional words or phrases. `[word]` is internally translated to `(word|)` before alternation expansion runs.

```python
expand_template("play [some] music")
# ['play music', 'play some music']

expand_template("[please] turn [the] lights on")
# ['please turn lights on', 'please turn the lights on',
#  'turn lights on', 'turn the lights on']
```

---

## Entity Slots: `{name}`

Curly-brace tokens are entity capture placeholders. They survive template expansion unchanged — `expand_template` does not fill them. The fuzzy matcher uses them in two ways:

1. **Confidence boosting** (`nebulento/container.py:308-311`): when the matched template contains `{name}` and a registered entity sample is found in the utterance, the confidence formula `score = 0.25 + score * 0.75` is applied, capping at 1.0.
2. **Entity return** (`nebulento/container.py:313`): matched entity values are included in `result["entities"]`.

```python
container.add_intent("buy", ["buy {item}", "I need {item} please"])
container.add_entity("item", ["milk", "eggs"])

result = container.calc_intent("I need milk please")
# result["entities"] == {'item': ['milk']}
```

Entity slot names are stored and returned lowercase (`{k.lower(): v for k, v in tagged.items()}`).

---

## Double-Brace Normalisation

`nebulento/bracket_expansion.py:33` — `clean_braces(example)`

Accidental double braces (`{{entity}}`) are normalised to single braces before any further processing. This is applied by `normalize_example`.

```python
from nebulento.bracket_expansion import clean_braces
clean_braces("buy {{item}} now")
# 'buy {item} now'
```

---

## Padatious Colon Syntax: `:0`

`nebulento/bracket_expansion.py:45` — `translate_padatious(example)`

Padatious intent files use `:0` as a word-slot token. `translate_padatious` converts each `:0` occurrence to a sequentially numbered `{wordN}` placeholder, so intent files written for Padatious work with Nebulento without modification.

```python
from nebulento.bracket_expansion import translate_padatious

translate_padatious("set a timer for :0 minutes")
# 'set a timer for {word0} minutes'

translate_padatious("remind me to :0 at :0")
# 'remind me to {word0} at {word1}'
```

Numbering starts at 0 and increments per `:0` occurrence within the line.

---

## Deduplication

`nebulento/container.py:91-95`

After expansion, all variants are collected into a Python `set` before being stored. This discards duplicates that arise when templates with overlapping alternatives produce the same concrete string.

```python
# "(play|play) music" → only one "play music" is stored
container.add_intent("play", ["(play|play) music", "play music"])
# stored: {'play music'}
```

---

## `expand_slots` — Filling Slots with Entity Values

`nebulento/bracket_expansion.py:148` — `expand_slots(template, slots)`

A utility function that first runs `expand_template` and then substitutes `{slot}` placeholders with every provided value, producing the Cartesian product.

This is not used internally by `IntentContainer` (which stores raw templates with slots intact), but is available for pre-generating concrete training sentences.

```python
from nebulento.bracket_expansion import expand_slots

expand_slots("buy {item} from {shop}", {
    "item": ["milk", "eggs"],
    "shop": ["Tesco"],
})
# ['buy eggs from Tesco', 'buy milk from Tesco']
```

Slots absent from the `slots` dict are left as-is (`{name}` unchanged in output).

---

## Summary Table

| Syntax | Meaning | Processed by |
|---|---|---|
| `(a\|b\|c)` | Alternation — Cartesian product of all groups | `expand_template` |
| `[word]` | Optional word/phrase | `expand_template` (via `expand_optional`) |
| `{name}` | Entity capture slot | Preserved through expansion; used at match time |
| `{{name}}` | Accidental double brace | Normalised to `{name}` by `clean_braces` |
| `:0` | Padatious word slot | Translated to `{word0}`, `{word1}`, … by `translate_padatious` |
