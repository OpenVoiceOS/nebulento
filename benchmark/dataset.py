"""Benchmark dataset — English subset of ``OpenVoiceOS/intents-for-eval``.

Loaded from the Hugging Face Hub at import time. Two of the dataset's configs
are used:

- ``en-US-templates`` — training templates, one row per template, grouped here
  by ``intent_id``. This is the corpus every template / sample engine trains on
  (nebulento, padatious, padacioso, padaos).
- ``en-US-test`` — labelled evaluation utterances across six splits:
  ``template``, ``paraphrase``, ``near_ood``, ``asr_noise``, ``typos`` carry a
  real intent label; ``far_ood`` utterances should match nothing.

The third config, ``en-US-keywords``, holds required/optional keyword vocab for
keyword engines (Adapt, palavreado) and is intentionally not used here —
nebulento is a fuzzy template matcher, so it is benchmarked on the templates.

Slots are turned into entities: every ``{slot}`` placeholder in a template
carries example values in the dataset, and those are collected into ``ENTITIES``
so engines can register them (the equivalent of a padatious ``.entity`` file)
and fill the slot at match time.

Exported symbols (kept stable for ``compare.py`` / ``accuracy.py``):

- ``INTENTS``     — ``{intent_id: {"train": [...], "test_match": [...],
  "entities": [slot_name, ...], "domain": str}}``
- ``ENTITIES``    — ``{slot_name: [example_value, ...]}`` (union across templates)
- ``DOMAINS``     — ``{domain: [intent_id, ...]}``
- ``NO_MATCH_UTTERANCES`` — far-OOD utterances that should not match any intent
- ``TEST_SPLITS`` — ``{split_name: [(utterance, expected_intent_or_None), ...]}``
"""
from collections import defaultdict

from datasets import load_dataset

_REPO = "OpenVoiceOS/intents-for-eval"
_LANG = "en-US"

#: test splits whose utterances carry a real intent label
_LABELLED_SPLITS = ("template", "paraphrase", "near_ood", "asr_noise", "typos")
#: test splits whose utterances should match nothing
_NOMATCH_SPLITS = ("far_ood",)


def _load():
    templates = load_dataset(_REPO, f"{_LANG}-templates")["train"]
    test = load_dataset(_REPO, f"{_LANG}-test")["test"]

    intents = {}
    domains = defaultdict(list)
    entities = defaultdict(list)
    for row in templates:
        iid = row["intent_id"]
        if iid not in intents:
            intents[iid] = {"train": [], "test_match": [],
                            "entities": [], "domain": row["domain"]}
            domains[row["domain"]].append(iid)
        if row["template"] not in intents[iid]["train"]:
            intents[iid]["train"].append(row["template"])
        for slot in row["slots"] or []:
            name = slot["name"]
            if name not in intents[iid]["entities"]:
                intents[iid]["entities"].append(name)
            for example in slot["examples"] or []:
                if example not in entities[name]:
                    entities[name].append(example)

    no_match = []
    splits = defaultdict(list)
    for row in test:
        utt = row["utterance"]
        expected = row["expected_intent"] or None
        if row["split"] in _NOMATCH_SPLITS or expected is None:
            no_match.append(utt)
            splits[row["split"]].append((utt, None))
        else:
            if expected in intents:
                intents[expected]["test_match"].append(utt)
            splits[row["split"]].append((utt, expected))

    return intents, dict(entities), dict(domains), no_match, dict(splits)


INTENTS, ENTITIES, DOMAINS, NO_MATCH_UTTERANCES, TEST_SPLITS = _load()
