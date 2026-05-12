# Installation

## Requirements

- Python 3.8 or later
- [rapidfuzz](https://github.com/maxbachmann/rapidfuzz) (installed automatically)
- [quebra_frases](https://github.com/OpenJarbas/quebra_frases) (installed automatically)

## From PyPI

```bash
pip install nebulento
```

This installs the core library and the OVOS pipeline plugin entry point. `rapidfuzz` and `quebra_frases` are pulled in as hard dependencies.

## From Source

```bash
git clone https://github.com/OpenJarbas/nebulento
cd nebulento
pip install -e .
```

Using [uv](https://github.com/astral-sh/uv):

```bash
uv pip install -e .
```

## Optional: Benchmark Dependencies

To run `benchmark/compare.py` against padaos, padacioso and padatious you need the benchmark extras:

```bash
pip install nebulento[benchmark]
# installs: padaos, padacioso, padatious
```

## OVOS Plugin Registration

When nebulento is installed it registers an `opm.pipeline` entry point:

```
ovos-nebulento-pipeline-plugin = nebulento.opm:NebulentoPipeline
```

`ovos-plugin-manager` discovers this automatically. No extra registration steps are required.

## Verifying the Install

```python
from nebulento import IntentContainer, MatchStrategy

c = IntentContainer()
c.add_intent("hello", ["hello", "hi there"])
result = c.calc_intent("hello")
assert result["name"] == "hello"
print("OK")
```
