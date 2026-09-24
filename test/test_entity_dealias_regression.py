"""Regression test for the entity-name munging bug (mirrors
ovos-padatious-pipeline-plugin#95).

ovos-workshop's register_entity_file() emits entity registration messages
with name = ``<skill_id>:<entity>_<md5hex32>``. Nebulento's slot-tagging
logic (IntentContainer.match_fuzzy) only tags a matched entity value into
the result when the literal token ``{<registered_entity_name>}`` appears in
one of the intent's own templates. Since skill authors write bare templates
like ``"drive to {place}"``, a munged registration name can never satisfy
that check, so the entity slot is silently dropped from match_data.
"""
from ovos_bus_client.message import Message
from ovos_utils.fakebus import FakeBus

from nebulento.opm import NebulentoPipeline


def _make_pipeline():
    return NebulentoPipeline(bus=FakeBus(), config={})


def test_entity_registered_via_real_workshop_munged_name_is_matched():
    p = _make_pipeline()
    lang = p.lang
    skill_id = "skill-driver.tigregotico"
    entity_bare = "place"
    # exact shape emitted by ovos_workshop.skills.base.register_entity_file
    munged_name = f"{skill_id}:{entity_bare}_1234567890abcdef1234567890abcdef"

    p.bus.emit(Message("padatious:register_intent", {
        "name": f"{skill_id}:drive.intent",
        "samples": ["drive to {place}", "navigate to {place}"],
        "lang": lang,
    }))
    p.bus.emit(Message("padatious:register_entity", {
        "name": munged_name,
        "samples": ["kitchen", "office", "garden"],
        "lang": lang,
    }))

    match = p.match_low(["drive to kitchen"], lang, Message("recognizer_loop:utterance"))
    assert match is not None, "intent failed to match at all"
    assert "kitchen" in (match.match_data.get("place") or []), \
        f"expected bare 'place' slot to contain 'kitchen', got {match.match_data!r}"
    p.shutdown()


def test_detach_entity_via_munged_name_actually_removes_it():
    p = _make_pipeline()
    lang = p.lang
    skill_id = "skill-driver.tigregotico"
    munged_name = f"{skill_id}:place_1234567890abcdef1234567890abcdef"

    p.bus.emit(Message("padatious:register_intent", {
        "name": f"{skill_id}:drive.intent",
        "samples": ["drive to {place}"],
        "lang": lang,
    }))
    p.bus.emit(Message("padatious:register_entity", {
        "name": munged_name,
        "samples": ["kitchen"],
        "lang": lang,
    }))
    container = p.containers[lang]
    assert "place" in container.registered_entities

    p.bus.emit(Message("detach_entity", {"name": munged_name, "lang": lang}))
    assert "place" not in container.registered_entities, \
        "remove path did not dealias the munged name, entity was never removed"
    p.shutdown()
