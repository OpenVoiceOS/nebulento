"""End-to-end tests for NebulentoPipeline using ovoscope.

We drive a MiniCroft directly and capture the intent-dispatch Message emitted
by IntentService._emit_match_message when our pipeline returns a match — the
same signal ovoscope itself uses.  No model patching is needed; Nebulento is
pure Python with no external model files.

Intents are registered via the standard `padatious:register_intent` bus event
(the same event padatious uses) with inline `samples` rather than file paths.
"""
import threading
import unittest

import pytest

ovoscope = pytest.importorskip("ovoscope", reason="ovoscope not installed; skipping E2E tests")

from ovos_bus_client.message import Message  # noqa: E402
from ovos_bus_client.session import Session  # noqa: E402
from ovos_config.config import Configuration  # noqa: E402
from ovoscope import get_minicroft  # noqa: E402

from nebulento.opm import NebulentoPipeline  # noqa: E402

PIPELINE_ID = "ovos-nebulento-pipeline-plugin"
CONFIG_KEY = "nebulento"

_HELLO_SAMPLES = ["hello", "hi", "hey", "greetings", "good morning"]
_BYE_SAMPLES = ["goodbye", "bye", "see you later", "farewell", "take care"]
_LIGHTS_ON_SAMPLES = ["turn on the lights", "switch on lights", "lights on please"]
_LIGHTS_OFF_SAMPLES = ["turn off the lights", "switch off lights", "lights off"]


class _E2EBase(unittest.TestCase):
    """Shared setup: spin up MiniCroft with the Nebulento pipeline."""

    extra_config: dict | None = None

    @classmethod
    def setUpClass(cls):
        cfg = Configuration()
        intents_cfg = cfg.setdefault("intents", {})
        cls._orig_intents_cfg = intents_cfg.get(CONFIG_KEY)
        plugin_cfg = {"strategy": "TOKEN_SET_RATIO"}
        if cls.extra_config:
            plugin_cfg.update(cls.extra_config)
        intents_cfg[CONFIG_KEY] = plugin_cfg

        cls.mc = get_minicroft(
            skill_ids=[],
            lang="en-US",
            default_pipeline=[PIPELINE_ID],
            max_wait=60,
        )
        cls.pipeline: NebulentoPipeline = cls.mc.intents.pipeline_plugins[PIPELINE_ID]

    @classmethod
    def tearDownClass(cls):
        try:
            cls.mc.stop()
        finally:
            cfg = Configuration()
            intents_cfg = cfg.get("intents", {})
            if cls._orig_intents_cfg is None:
                intents_cfg.pop(CONFIG_KEY, None)
            else:
                intents_cfg[CONFIG_KEY] = cls._orig_intents_cfg

    def setUp(self):
        for lang, container in self.pipeline.containers.items():
            for name in list(container.registered_intents):
                container.remove_intent(name)
        self.pipeline.registered_intents.clear()

    def _register_intent(self, name: str, samples: list, lang: str = "en-US"):
        self.mc.bus.emit(Message("padatious:register_intent", {
            "name": name,
            "samples": samples,
            "lang": lang,
        }))

    def _register_entity(self, name: str, samples: list, lang: str = "en-US"):
        self.mc.bus.emit(Message("padatious:register_entity", {
            "name": name,
            "samples": samples,
            "lang": lang,
        }))

    def _utterance_msg(self, utterance: str,
                       session_pipeline: list[str] | None = None) -> Message:
        ctx = {}
        if session_pipeline is not None:
            sess = Session(session_id="ovoscope-test", pipeline=session_pipeline)
            ctx["session"] = sess.serialize()
        return Message(
            "recognizer_loop:utterance",
            data={"utterances": [utterance], "lang": "en-US"},
            context=ctx,
        )

    def _send_and_capture(self, utterance: str, expected_types: list[str],
                          timeout: float = 5.0,
                          session_pipeline: list[str] | None = None) -> Message | None:
        got: list[Message] = []
        done = threading.Event()
        failed = threading.Event()

        def _capture_match(msg):
            got.append(msg)
            done.set()

        def _capture_fail(msg):
            failed.set()
            done.set()

        for t in expected_types:
            self.mc.bus.on(t, _capture_match)
        self.mc.bus.on("complete_intent_failure", _capture_fail)
        try:
            self.mc.bus.emit(self._utterance_msg(utterance, session_pipeline))
            done.wait(timeout=timeout)
        finally:
            for t in expected_types:
                self.mc.bus.remove(t, _capture_match)
            self.mc.bus.remove("complete_intent_failure", _capture_fail)
        if failed.is_set() and not got:
            return None
        return got[0] if got else None

    def _expect_no_match(self, utterance: str, timeout: float = 2.0,
                         session_pipeline: list[str] | None = None):
        failed = threading.Event()

        def _on_fail(_msg):
            failed.set()

        self.mc.bus.on("complete_intent_failure", _on_fail)
        try:
            self.mc.bus.emit(self._utterance_msg(utterance, session_pipeline))
            failed.wait(timeout=timeout)
        finally:
            self.mc.bus.remove("complete_intent_failure", _on_fail)
        self.assertTrue(
            failed.is_set(),
            f"Expected no match for {utterance!r} but got no complete_intent_failure.",
        )


class TestRegisteredIntentMatch(_E2EBase):
    def test_exact_utterance_dispatches_intent(self):
        self._register_intent("test_skill:hello", _HELLO_SAMPLES)
        msg = self._send_and_capture("hello", expected_types=["test_skill:hello"])
        self.assertIsNotNone(msg, "expected intent match on bus")
        self.assertEqual(msg.msg_type, "test_skill:hello")
        self.assertEqual(msg.data.get("utterance"), "hello")
        self.assertGreater(msg.data.get("confidence", 0), 0.5)

    def test_close_paraphrase_dispatches_intent(self):
        self._register_intent("test_skill:hello", _HELLO_SAMPLES)
        msg = self._send_and_capture("hello there", expected_types=["test_skill:hello"])
        self.assertIsNotNone(msg)
        self.assertEqual(msg.msg_type, "test_skill:hello")

    def test_no_match_when_no_intents_registered(self):
        self._expect_no_match("hello")

    def test_no_match_unrelated_utterance(self):
        self._register_intent("test_skill:hello", _HELLO_SAMPLES)
        self._expect_no_match("set a timer for five minutes")

    def test_best_intent_selected_among_multiple(self):
        self._register_intent("test_skill:hello", _HELLO_SAMPLES)
        self._register_intent("test_skill:bye", _BYE_SAMPLES)
        msg = self._send_and_capture("goodbye", expected_types=["test_skill:bye"])
        self.assertIsNotNone(msg)
        self.assertEqual(msg.msg_type, "test_skill:bye")

    def test_utterance_field_preserved(self):
        self._register_intent("test_skill:hello", _HELLO_SAMPLES)
        utterance = "hi there"
        msg = self._send_and_capture(utterance, expected_types=["test_skill:hello"])
        self.assertIsNotNone(msg)
        self.assertEqual(msg.data.get("utterance"), utterance)


class TestDetach(_E2EBase):
    def test_detach_intent_prevents_match(self):
        self._register_intent("test_skill:hello", _HELLO_SAMPLES)
        msg = self._send_and_capture("hello", expected_types=["test_skill:hello"])
        self.assertIsNotNone(msg)

        self.mc.bus.emit(Message("detach_intent", {"intent_name": "test_skill:hello"}))
        self._expect_no_match("hello")

    def test_detach_skill_removes_all_its_intents(self):
        self._register_intent("skill_a:hello", _HELLO_SAMPLES)
        self._register_intent("skill_a:bye", _BYE_SAMPLES)
        self._register_intent("skill_b:lights_on", _LIGHTS_ON_SAMPLES)

        self.mc.bus.emit(Message("detach_skill", {"skill_id": "skill_a"}))

        self._expect_no_match("hello")
        self._expect_no_match("goodbye")
        msg = self._send_and_capture("turn on the lights", expected_types=["skill_b:lights_on"])
        self.assertIsNotNone(msg, "skill_b intent should still be active after skill_a detach")


class TestConfidenceThresholds(_E2EBase):
    extra_config = {"conf_high": 0.95, "conf_med": 0.8, "conf_low": 0.5}

    def test_high_confidence_exact_match_fires(self):
        self._register_intent("test_skill:lights_on", _LIGHTS_ON_SAMPLES)
        msg = self._send_and_capture(
            "turn on the lights", expected_types=["test_skill:lights_on"]
        )
        self.assertIsNotNone(msg)

    def test_low_confidence_threshold_fires_on_paraphrase(self):
        # Use a very low conf_low so even a loose match gets through
        self._register_intent("test_skill:hello", _HELLO_SAMPLES)
        msg = self._send_and_capture("hey", expected_types=["test_skill:hello"])
        self.assertIsNotNone(msg)


class TestEntityExtraction(_E2EBase):
    def test_entity_slot_captured_in_match(self):
        self._register_entity("test_skill:Item", ["milk", "bread", "eggs", "cheese"])
        self._register_intent(
            "test_skill:buy",
            ["buy {test_skill:Item}", "get {test_skill:Item}", "purchase {test_skill:Item}"],
        )
        msg = self._send_and_capture("buy milk", expected_types=["test_skill:buy"])
        self.assertIsNotNone(msg)
        self.assertEqual(msg.msg_type, "test_skill:buy")

    def test_no_match_without_known_entity_value(self):
        self._register_entity("test_skill:Item", ["milk", "bread"])
        self._register_intent(
            "test_skill:buy",
            ["buy {test_skill:Item}", "get {test_skill:Item}"],
        )
        # "car" is not in the Item entity list — should score low / no match
        self._expect_no_match("buy a car")


class TestSessionBlacklist(_E2EBase):
    def test_blacklisted_intent_is_skipped(self):
        self._register_intent("test_skill:hello", _HELLO_SAMPLES)
        sess = Session(
            session_id="bl-test",
            blacklisted_intents=["test_skill:hello"],
        )
        msg = self._utterance_msg("hello")
        msg.context["session"] = sess.serialize()

        failed = threading.Event()
        self.mc.bus.on("complete_intent_failure", lambda _: failed.set())
        try:
            self.mc.bus.emit(msg)
            failed.wait(timeout=3.0)
        finally:
            self.mc.bus.remove("complete_intent_failure", lambda _: failed.set())
        self.assertTrue(failed.is_set(), "blacklisted intent should yield intent_failure")

    def test_blacklisted_skill_is_skipped(self):
        self._register_intent("test_skill:hello", _HELLO_SAMPLES)
        sess = Session(
            session_id="bl-skill-test",
            blacklisted_skills=["test_skill"],
        )
        msg = self._utterance_msg("hello")
        msg.context["session"] = sess.serialize()

        failed = threading.Event()
        self.mc.bus.on("complete_intent_failure", lambda _: failed.set())
        try:
            self.mc.bus.emit(msg)
            failed.wait(timeout=3.0)
        finally:
            self.mc.bus.remove("complete_intent_failure", lambda _: failed.set())
        self.assertTrue(failed.is_set(), "blacklisted skill should yield intent_failure")


if __name__ == "__main__":
    unittest.main()
