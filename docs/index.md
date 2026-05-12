# Nebulento

A lightweight fuzzy-matching intent parser built on [rapidfuzz](https://github.com/maxbachmann/rapidfuzz).

## Overview

Nebulento finds the closest matching intent by comparing an utterance against all registered training sentences using configurable fuzzy similarity strategies. It handles spelling errors, word-order variation, contractions, and natural phrasing that exact-match parsers miss.

Key design choices:

- **No training step** — intents are registered as template strings and are available immediately. There is no model to train or reload.
- **Template expansion at registration time** — `(a|b)` alternation and `[optional]` syntax are fully expanded into concrete strings when `add_intent` is called.
- **Configurable similarity strategy** — nine `MatchStrategy` values map to distinct rapidfuzz scorers (plus a difflib fallback), each with a different precision/recall trade-off.
- **Entity extraction** — `{slot}` placeholders in templates are paired with registered entity samples. When a registered value appears in the utterance the confidence is boosted and the value is returned in `entities`.
- **Context gating** — intents can be conditionally suppressed or required based on named active contexts, or blocked when specific keywords appear in the query.

The package ships both a standalone Python library and an OVOS pipeline plugin (`NebulentoPipeline`) that integrates with the OpenVoiceOS skill framework.

---

## Architecture

```
                   ┌─────────────────────────────────────┐
                   │           IntentContainer            │
                   │                                      │
  utterance ──────►│  normalize_utterance()               │
                   │         │                            │
                   │         ▼                            │
                   │  _filter()  ◄── context gates        │
                   │             ◄── keyword exclusions   │
                   │         │                            │
                   │         ▼                            │
                   │  match_entities()  ◄── entity store  │
                   │         │                            │
                   │         ▼                            │
                   │  match_one() per intent              │
                   │   (rapidfuzz scorer)                 │
                   │         │                            │
                   │         ▼                            │
                   │  confidence boost if entity found    │
                   │         │                            │
                   │         ▼                            │
  result ◄─────────│  tie-break → best MatchResult        │
                   └─────────────────────────────────────┘
```

```
                   ┌─────────────────────────────────────┐
                   │       DomainIntentContainer          │
                   │                                      │
  utterance ──────►│  domain_engine.calc_intent()         │
                   │         │                            │
                   │         ▼                            │
                   │  domains[name].calc_intent()         │
                   │         │                            │
  result ◄─────────│  MatchResult                         │
                   └─────────────────────────────────────┘
```

---

## Key Classes

| Class | Purpose | Source |
|---|---|---|
| `IntentContainer` | Register intents/entities, score utterances | `nebulento/container.py:17` |
| `DomainIntentContainer` | Two-level domain→intent matching | `nebulento/domain_engine.py:10` |
| `MatchStrategy` | Enum of nine fuzzy similarity algorithms | `nebulento/fuzz.py:15` |
| `NebulentoPipeline` | OVOS pipeline plugin wrapping `IntentContainer` | `nebulento/opm.py:51` |
| `NebulentoIntent` | Result object returned by the pipeline plugin | `nebulento/opm.py:21` |

### Utility functions

| Function | Purpose | Source |
|---|---|---|
| `expand_template` | Expand `(a\|b)` / `[opt]` syntax | `nebulento/bracket_expansion.py:101` |
| `expand_slots` | Fill `{slot}` with entity sample values | `nebulento/bracket_expansion.py:148` |
| `normalize_utterance` | Apostrophe + whitespace normalisation for queries | `nebulento/bracket_expansion.py:86` |
| `normalize_example` | Normalisation + padatious translation for templates | `nebulento/bracket_expansion.py:69` |
| `translate_padatious` | Convert `:0` word-slot tokens to `{wordN}` | `nebulento/bracket_expansion.py:45` |
| `clean_braces` | Normalise `{{entity}}` → `{entity}` | `nebulento/bracket_expansion.py:33` |
| `fuzzy_match` | Score two strings with a chosen strategy | `nebulento/fuzz.py:51` |
| `match_one` | Best-match from a list of candidates | `nebulento/fuzz.py:82` |
| `match_all` | All candidates sorted by score | `nebulento/fuzz.py:101` |

---

## Contents

| Page | Description |
|---|---|
| [Installation](installation.md) | pip install, from source, optional deps |
| [Quick Start](quickstart.md) | 5-minute guide: add intent, entity, calc_intent |
| [Intent API](intent-api.md) | Full `IntentContainer` and `DomainIntentContainer` reference |
| [Match Strategies](strategies.md) | All nine `MatchStrategy` values: what they measure, when to use |
| [Template Syntax](template-syntax.md) | Expansion rules, entity slots, padatious compat |
| [Entity Extraction](entity-extraction.md) | How entity registration and slot filling works |
| [Normalisation](normalisation.md) | Apostrophe handling, case folding, whitespace |
| [OVOS Plugin](ovos-plugin.md) | `NebulentoPipeline`: events, config, confidence tiers |
| [Domain Matching](domain-matching.md) | `DomainIntentContainer` guide |
| [Configuration](configuration.md) | All OVOS plugin config keys |
| [Benchmark](benchmark.md) | Accuracy table, how to reproduce |
| [Troubleshooting](troubleshooting.md) | Common issues and solutions |
