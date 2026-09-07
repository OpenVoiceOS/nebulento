"""End-to-end tests for NebulentoPipeline using ovoscope.

Built on top of ovoscope's reusable :class:`E2EPipelineHarness` so this file
only contains nebulento-specific concerns (config, intent samples,
assertions).  The harness handles MiniCroft startup, Configuration
save/restore, bus capture, and per-test skill detach.
"""
import threading
import time
import unittest
from typing import Optional

import pytest

ovoscope = pytest.importorskip("ovoscope", reason="ovoscope not installed; skipping E2E tests")

from ovos_bus_client.message import Message  # noqa: E402
from ovoscope import (  # noqa: E402
    E2EPipelineHarness,
    detach_intent,
    detach_skill,
    make_session,
    register_padatious_entity,
    register_padatious_intent,
)

from nebulento.opm import NebulentoPipeline  # noqa: E402

PIPELINE_ID = "ovos-nebulento-pipeline-plugin"
CONFIG_KEY = "nebulento"

_HELLO_SAMPLES = ["hello", "hi", "hey", "greetings", "good morning"]
_BYE_SAMPLES = ["goodbye", "bye", "see you later", "farewell", "take care"]
_LIGHTS_ON_SAMPLES = ["turn on the lights", "switch on lights", "lights on please"]
_LIGHTS_OFF_SAMPLES = ["turn off the lights", "switch off lights", "lights off"]


class _NebulentoHarness(E2EPipelineHarness):
    """Project-specific harness binding for NebulentoPipeline."""

    PIPELINE_ID = PIPELINE_ID
    CONFIG_KEY = CONFIG_KEY
    PLUGIN_CONFIG = {"strategy": "TOKEN_SET_RATIO"}
    SKILL_ID = "test_skill_nebulento"

    pipeline: NebulentoPipeline  # type: ignore[assignment]

    def _register_intent(self, name, samples):
        register_padatious_intent(self.bus, name, samples,
                                   skill_id=name.split(":", 1)[0])

    def _register_entity(self, name, samples):
        register_padatious_entity(self.bus, name, samples,
                                   skill_id=self.SKILL_ID)


class TestRegisteredIntentMatch(_NebulentoHarness):
    def test_exact_utterance_dispatches_intent(self):
        self._register_intent(f"{self.SKILL_ID}:hello", _HELLO_SAMPLES)
        msg = self.send_and_capture("hello", expected_types=[f"{self.SKILL_ID}:hello"])
        self.assertIsNotNone(msg, "expected intent match on bus")
        self.assertEqual(msg.msg_type, f"{self.SKILL_ID}:hello")
        self.assertEqual(msg.data.get("utterance"), "hello")

    def test_no_match_when_no_intents_registered(self):
        self.expect_no_match("hello")

    def test_no_match_unrelated_utterance(self):
        self._register_intent(f"{self.SKILL_ID}:hello", _HELLO_SAMPLES)
        self.expect_no_match("set a timer for five minutes")

    def test_best_intent_selected_among_multiple(self):
        self._register_intent(f"{self.SKILL_ID}:hello", _HELLO_SAMPLES)
        self._register_intent(f"{self.SKILL_ID}:bye", _BYE_SAMPLES)
        msg = self.send_and_capture("goodbye", expected_types=[f"{self.SKILL_ID}:bye"])
        self.assertIsNotNone(msg)
        self.assertEqual(msg.msg_type, f"{self.SKILL_ID}:bye")

    def test_utterance_field_preserved(self):
        self._register_intent(f"{self.SKILL_ID}:hello", _HELLO_SAMPLES)
        msg = self.send_and_capture("hello", expected_types=[f"{self.SKILL_ID}:hello"])
        self.assertIsNotNone(msg)
        self.assertEqual(msg.data.get("utterance"), "hello")


class TestDetach(_NebulentoHarness):
    def test_detach_intent_prevents_match(self):
        self._register_intent(f"{self.SKILL_ID}:hello", _HELLO_SAMPLES)
        msg = self.send_and_capture("hello", expected_types=[f"{self.SKILL_ID}:hello"])
        self.assertIsNotNone(msg)

        detach_intent(self.bus, f"{self.SKILL_ID}:hello", skill_id=self.SKILL_ID)
        self.expect_no_match("hello")

    def test_detach_skill_removes_all_its_intents(self):
        self._register_intent(f"{self.SKILL_ID}:hello", _HELLO_SAMPLES)
        self._register_intent(f"{self.SKILL_ID}:bye", _BYE_SAMPLES)
        self._register_intent("skill_b_nebulento:lights_on", _LIGHTS_ON_SAMPLES)

        detach_skill(self.bus, self.SKILL_ID)

        self.expect_no_match("hello")
        self.expect_no_match("goodbye")
        msg = self.send_and_capture(
            "turn on the lights", expected_types=["skill_b_nebulento:lights_on"]
        )
        self.assertIsNotNone(msg, "skill_b intent should still be active after skill_a detach")
        detach_skill(self.bus, "skill_b_nebulento")


class TestEntityExtraction(_NebulentoHarness):
    def test_entity_slot_captured_in_match(self):
        self._register_entity("item", ["milk", "bread", "eggs", "cheese"])
        self._register_intent(
            f"{self.SKILL_ID}:buy",
            ["buy {item}", "get {item}", "purchase {item}"],
        )
        msg = self.send_and_capture("buy milk", expected_types=[f"{self.SKILL_ID}:buy"])
        self.assertIsNotNone(msg)
        self.assertEqual(msg.msg_type, f"{self.SKILL_ID}:buy")
        self.assertIn("milk", msg.data.get("item", []))


class TestSessionBlacklist(_NebulentoHarness):
    def test_blacklisted_intent_is_skipped(self):
        self._register_intent(f"{self.SKILL_ID}:hello", _HELLO_SAMPLES)
        sess = make_session(
            "bl-intent-test",
            blacklisted_intents=[f"{self.SKILL_ID}:hello"],
        )
        self.expect_no_match("hello", session=sess, timeout=3.0)

    def test_blacklisted_skill_is_skipped(self):
        self._register_intent(f"{self.SKILL_ID}:hello", _HELLO_SAMPLES)
        sess = make_session(
            "bl-skill-test",
            blacklisted_skills=[self.SKILL_ID],
        )
        self.expect_no_match("hello", session=sess, timeout=3.0)


class _HierarchicalNebulentoHarness(E2EPipelineHarness):
    """Project-specific harness binding for HierarchicalNebulentoPipeline."""

    PIPELINE_ID = "ovos-nebulento-hierarchical-pipeline-plugin"
    CONFIG_KEY = "nebulento_hierarchical"
    PLUGIN_CONFIG = {"strategy": "TOKEN_SET_RATIO"}
    SKILL_ID = "media_skill_nebulento"

    def _register_intent(self, name, samples):
        register_padatious_intent(self.bus, name, samples,
                                   skill_id=name.split(":", 1)[0])


class TestHierarchicalRouting(_HierarchicalNebulentoHarness):
    def test_routes_to_correct_domain(self):
        self._register_intent(f"{self.SKILL_ID}:play", _HELLO_SAMPLES)
        self._register_intent("home_skill_nebulento:lights_on", _LIGHTS_ON_SAMPLES)

        msg = self.send_and_capture(
            "turn on the lights",
            expected_types=["home_skill_nebulento:lights_on"],
        )
        self.assertIsNotNone(msg)
        self.assertEqual(msg.msg_type, "home_skill_nebulento:lights_on")
        detach_skill(self.bus, "home_skill_nebulento")

    def test_detach_skill_removes_domain(self):
        self._register_intent(f"{self.SKILL_ID}:hello", _HELLO_SAMPLES)
        msg = self.send_and_capture("hello", expected_types=[f"{self.SKILL_ID}:hello"])
        self.assertIsNotNone(msg)

        detach_skill(self.bus, self.SKILL_ID)
        self.expect_no_match("hello")


if __name__ == "__main__":
    unittest.main()
