"""Pipeline plugin tests for NebulentoPipeline using a mock bus."""
import unittest
from unittest import mock

from ovos_bus_client.message import Message

from nebulento.opm import NebulentoPipeline


def _register_intent_msg(name, samples, lang="en-US"):
    return Message("padatious:register_intent", {
        "name": name,
        "samples": samples,
        "lang": lang,
    })


def _register_entity_msg(name, samples, lang="en-US"):
    return Message("padatious:register_entity", {
        "name": name,
        "samples": samples,
        "lang": lang,
    })


def _get_last_reply(bus):
    return bus.emit.call_args[0][0]


class TestNebulentoPipeline(unittest.TestCase):
    def setUp(self):
        self.bus = mock.Mock()
        self.pipeline = NebulentoPipeline(bus=self.bus, config={
            "conf_high": 0.95, "conf_med": 0.8, "conf_low": 0.5,
        })
        self.pipeline.register_intent(_register_intent_msg(
            "test_skill:HelloIntent",
            ["hello", "hi", "how are you", "hey there"],
        ))
        self.pipeline.register_intent(_register_intent_msg(
            "test_skill:GoodbyeIntent",
            ["goodbye", "see you later", "bye", "farewell"],
        ))

    def test_match_high_exact_utterance(self):
        msg = Message("recognizer_loop:utterance", {"utterances": ["hello"], "lang": "en-US"})
        result = self.pipeline.match_high(["hello"], "en-US", msg)
        self.assertIsNotNone(result)
        self.assertEqual(result.match_type, "test_skill:HelloIntent")

    def test_match_high_none_on_unrelated(self):
        msg = Message("recognizer_loop:utterance",
                      {"utterances": ["set timer for five minutes"], "lang": "en-US"})
        result = self.pipeline.match_high(["set timer for five minutes"], "en-US", msg)
        self.assertIsNone(result)

    def test_match_medium_fires(self):
        msg = Message("recognizer_loop:utterance", {"utterances": ["hello"], "lang": "en-US"})
        result = self.pipeline.match_medium(["hello"], "en-US", msg)
        self.assertIsNotNone(result)

    def test_match_low_fires(self):
        msg = Message("recognizer_loop:utterance", {"utterances": ["hello"], "lang": "en-US"})
        result = self.pipeline.match_low(["hello"], "en-US", msg)
        self.assertIsNotNone(result)

    def test_best_intent_wins_over_weaker(self):
        msg = Message("recognizer_loop:utterance", {"utterances": ["goodbye"], "lang": "en-US"})
        result = self.pipeline.match_high(["goodbye"], "en-US", msg)
        self.assertIsNotNone(result)
        self.assertEqual(result.match_type, "test_skill:GoodbyeIntent")

    def test_detach_intent_removes_match(self):
        self.pipeline.handle_detach_intent(
            Message("detach_intent", {"intent_name": "test_skill:HelloIntent"})
        )
        msg = Message("recognizer_loop:utterance", {"utterances": ["hello"], "lang": "en-US"})
        result = self.pipeline.match_high(["hello"], "en-US", msg)
        self.assertIsNone(result)

    def test_detach_skill_removes_all_intents(self):
        self.pipeline.handle_detach_skill(
            Message("detach_skill", {"skill_id": "test_skill"})
        )
        msg = Message("recognizer_loop:utterance", {"utterances": ["hello"], "lang": "en-US"})
        result = self.pipeline.match_high(["hello"], "en-US", msg)
        self.assertIsNone(result)

    def test_registered_intent_names(self):
        self.assertIn("test_skill:HelloIntent", self.pipeline.registered_intents)
        self.assertIn("test_skill:GoodbyeIntent", self.pipeline.registered_intents)

    def test_train_emits_trained_signal(self):
        self.pipeline.train()
        self.bus.emit.assert_called()
        last_msg = _get_last_reply(self.bus)
        self.assertEqual(last_msg.msg_type, "mycroft.skills.trained")

    def test_shutdown_removes_bus_handlers(self):
        self.pipeline.shutdown()
        self.bus.remove.assert_called()

    def test_match_result_utterance_field(self):
        msg = Message("recognizer_loop:utterance", {"utterances": ["hello"], "lang": "en-US"})
        result = self.pipeline.match_high(["hello"], "en-US", msg)
        self.assertIsNotNone(result)
        self.assertEqual(result.utterance, "hello")

    def test_multiple_hypotheses_best_wins(self):
        msg = Message("recognizer_loop:utterance",
                      {"utterances": ["zzz gibberish", "hello"], "lang": "en-US"})
        result = self.pipeline.match_high(["zzz gibberish", "hello"], "en-US", msg)
        self.assertIsNotNone(result)
        self.assertEqual(result.match_type, "test_skill:HelloIntent")

    def test_blacklisted_intent_not_returned(self):
        from ovos_bus_client.session import Session, SessionManager
        sess = Session("test-session")
        sess.blacklisted_intents = ["test_skill:HelloIntent"]
        with mock.patch.object(SessionManager, "get", return_value=sess):
            msg = Message("recognizer_loop:utterance",
                          {"utterances": ["hello"], "lang": "en-US"},
                          context={"session": sess.serialize()})
            result = self.pipeline.match_high(["hello"], "en-US", msg)
        self.assertIsNone(result)

    def test_blacklisted_skill_not_returned(self):
        from ovos_bus_client.session import Session, SessionManager
        sess = Session("test-session-2")
        sess.blacklisted_skills = ["test_skill"]
        with mock.patch.object(SessionManager, "get", return_value=sess):
            msg = Message("recognizer_loop:utterance",
                          {"utterances": ["hello"], "lang": "en-US"},
                          context={"session": sess.serialize()})
            result = self.pipeline.match_high(["hello"], "en-US", msg)
        self.assertIsNone(result)

    def test_duplicate_registration_skipped_silently(self):
        # Second registration of the same intent should not raise
        self.pipeline.register_intent(_register_intent_msg(
            "test_skill:HelloIntent",
            ["hello again"],
        ))

    def test_calc_intent_returns_none_on_empty_container(self):
        pipeline = NebulentoPipeline(bus=mock.Mock(), config={})
        result = pipeline.calc_intent(["hello"])
        self.assertIsNone(result)


class TestNebulentoPipelineWithEntities(unittest.TestCase):
    def setUp(self):
        self.bus = mock.Mock()
        self.pipeline = NebulentoPipeline(bus=self.bus, config={
            "conf_high": 0.9, "conf_med": 0.5, "conf_low": 0.3,
            "strategy": "TOKEN_SET_RATIO",
        })
        self.pipeline.register_intent(_register_intent_msg(
            "skill:BuyIntent",
            ["buy {item}", "purchase {item}", "get {item} for me"],
        ))
        self.pipeline.register_entity(_register_entity_msg(
            "skill:item",
            ["milk", "cheese", "bread"],
        ))

    def test_intent_fires_on_buy(self):
        msg = Message("recognizer_loop:utterance", {"utterances": ["buy milk"], "lang": "en-US"})
        result = self.pipeline.match_medium(["buy milk"], "en-US", msg)
        self.assertIsNotNone(result)
        self.assertEqual(result.match_type, "skill:BuyIntent")

    def test_remove_entity(self):
        self.pipeline._detach_entity("skill:item", "en-US")
        self.assertNotIn("skill:item", self.pipeline.containers["en-US"].registered_entities)


if __name__ == "__main__":
    unittest.main()
