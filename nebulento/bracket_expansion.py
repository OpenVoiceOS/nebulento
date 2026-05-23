"""Template expansion and text normalisation utilities.

Expansion delegates to :mod:`ovos_spec_tools`. The symbols here are kept as
thin deprecation shims so downstream code keeps working.
"""

import itertools
import re
import warnings
from typing import Dict, List

from ovos_utils.log import deprecated
from ovos_spec_tools import expand as _spec_expand

from nebulento.version import VERSION_MAJOR

_REMOVAL = f"{VERSION_MAJOR + 1}.0.0"

# Apostrophe variants replaced with a space to preserve word boundaries.
_APOSTROPHES = (
    "'",   # U+0027 ASCII apostrophe
    "’",  # RIGHT SINGLE QUOTATION MARK
    "‘",  # LEFT SINGLE QUOTATION MARK
    "ʼ",  # MODIFIER LETTER APOSTROPHE
    "ʹ",  # MODIFIER LETTER PRIME
    "`",        # U+0060 GRAVE ACCENT
    "´",  # ACUTE ACCENT
    "＇",  # FULLWIDTH APOSTROPHE
)
_APOS_RE = re.compile("|".join(re.escape(a) for a in _APOSTROPHES))
_WS_RE = re.compile(r"\s+")


def _drop_apostrophes(text: str) -> str:
    """Replace all apostrophe variants with a single space."""
    return _APOS_RE.sub(" ", text)


def _normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace to a single space and strip ends."""
    return _WS_RE.sub(" ", text).strip()


def clean_braces(example: str) -> str:
    """Normalise accidental double-braces: ``{{entity}}`` -> ``{entity}``."""
    return example.replace("{{", "{").replace("}}", "}")


def translate_padatious(example: str) -> str:
    """Translate Padatious ``:0`` word-slot tokens to ``{wordN}`` entity syntax."""
    if ":0" not in example:
        return example
    tokens = example.split()
    i = 0
    for idx, token in enumerate(tokens):
        if token == ":0":
            tokens[idx] = "{" + f"word{i}" + "}"
            i += 1
    return " ".join(tokens)


def _lowercase_slots(text: str) -> str:
    """Lowercase ``{Slot}`` placeholder names so they pass strict validators."""
    return re.sub(r"\{([^\{\}]+)\}", lambda m: "{" + m.group(1).lower() + "}", text)


def normalize_example(example: str) -> str:
    """Normalise a training template for storage."""
    text = clean_braces(translate_padatious(example))
    text = _drop_apostrophes(text)
    text = _lowercase_slots(text)
    return _normalize_whitespace(text)


def normalize_utterance(text: str) -> str:
    """Normalise a plain query utterance for matching."""
    return _normalize_whitespace(_drop_apostrophes(text))


@deprecated("use ovos_spec_tools.expand", _REMOVAL)
def expand_template(template: str) -> List[str]:
    """Expand a template into all concrete string variants.

    .. deprecated::
        Use :func:`ovos_spec_tools.expand` instead. This shim delegates to it.
    """
    warnings.warn(
        "nebulento.bracket_expansion.expand_template is deprecated; "
        "use ovos_spec_tools.expand instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return list(_spec_expand(template))


@deprecated("use ovos_spec_tools.expand", _REMOVAL)
def expand_slots(template: str, slots: Dict[str, List[str]]) -> List[str]:
    """Expand a template and substitute ``{slot}`` placeholders.

    .. deprecated::
        Use :func:`ovos_spec_tools.expand` and substitute slots in caller code.
    """
    warnings.warn(
        "nebulento.bracket_expansion.expand_slots is deprecated; "
        "use ovos_spec_tools.expand instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    all_sentences: List[str] = []
    for sentence in _spec_expand(template):
        matches = re.findall(r"\{([^\{\}]+)\}", sentence)
        if matches:
            slot_options = [slots.get(m, [f"{{{m}}}"]) for m in matches]
            for combination in itertools.product(*slot_options):
                filled = sentence
                for slot, replacement in zip(matches, combination):
                    filled = filled.replace(f"{{{slot}}}", replacement)
                all_sentences.append(filled)
        else:
            all_sentences.append(sentence)
    return all_sentences
