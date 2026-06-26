"""Pipeline plugin tests for NebulentoPipeline using a mock bus."""
import unittest
from unittest import mock

from ovos_bus_client.message import Message

from nebulento.opm import HierarchicalNebulentoPipeline, NebulentoPipeline


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

    def test_handle_detach_entity_via_bus(self):
        """detach_entity bus event removes the entity from the container."""
        self.pipeline.handle_detach_entity(Message("detach_entity", {
            "entity_name": "skill:item", "lang": "en-US",
        }))
        self.assertNotIn(
            "skill:item",
            self.pipeline.containers["en-US"].registered_entities,
        )

    def test_handle_detach_entity_missing_name_is_noop(self):
        """detach_entity without a name must not raise or clear the container."""
        self.pipeline.handle_detach_entity(Message("detach_entity", {}))
        self.assertIn(
            "skill:item",
            self.pipeline.containers["en-US"].registered_entities,
        )


class TestHierarchicalNebulentoPipeline(unittest.TestCase):
    def setUp(self):
        self.bus = mock.Mock()
        self.pipeline = HierarchicalNebulentoPipeline(bus=self.bus, config={
            "conf_high": 0.95, "conf_med": 0.8, "conf_low": 0.5,
        })
        self.pipeline.register_intent(_register_intent_msg(
            "media_skill:PlayIntent",
            ["play music", "play some music", "put on a song", "start the music"],
        ))
        self.pipeline.register_intent(_register_intent_msg(
            "home_skill:LightsIntent",
            ["turn on the lights", "lights on", "switch the lights on"],
        ))

    def test_container_is_hierarchical(self):
        from nebulento import HierarchicalIntentContainer
        self.assertIsInstance(self.pipeline.containers["en-US"],
                              HierarchicalIntentContainer)

    def test_intent_filed_under_domain(self):
        container = self.pipeline.containers["en-US"]
        self.assertIn("media_skill", container.domains)
        self.assertIn("home_skill", container.domains)

    def test_match_routes_to_correct_domain(self):
        msg = Message("recognizer_loop:utterance",
                      {"utterances": ["play music"], "lang": "en-US"})
        result = self.pipeline.match_high(["play music"], "en-US", msg)
        self.assertIsNotNone(result)
        self.assertEqual(result.match_type, "media_skill:PlayIntent")

    def test_match_none_on_unrelated(self):
        msg = Message("recognizer_loop:utterance",
                      {"utterances": ["what is the capital of france"], "lang": "en-US"})
        result = self.pipeline.match_high(
            ["what is the capital of france"], "en-US", msg)
        self.assertIsNone(result)

    def test_detach_skill_removes_domain(self):
        self.pipeline.handle_detach_skill(
            Message("detach_skill", {"skill_id": "media_skill"})
        )
        self.assertNotIn("media_skill",
                         self.pipeline.containers["en-US"].domains)

    def test_detach_intent_removes_match(self):
        self.pipeline.handle_detach_intent(
            Message("detach_intent", {"intent_name": "media_skill:PlayIntent"})
        )
        msg = Message("recognizer_loop:utterance",
                      {"utterances": ["play music"], "lang": "en-US"})
        result = self.pipeline.match_high(["play music"], "en-US", msg)
        self.assertIsNone(result)


def _spec_register_template_msg(skill_id, intent_name, samples, lang="en-US",
                                blacklist=None):
    return Message("ovos.intent.register.template", {
        "skill_id": skill_id,
        "intent_name": intent_name,
        "lang": lang,
        "samples": samples,
        "blacklist": blacklist or [],
    })


def _spec_register_entity_msg(skill_id, entity_name, samples, lang="en-US"):
    return Message("ovos.entity.register", {
        "skill_id": skill_id,
        "entity_name": entity_name,
        "lang": lang,
        "samples": samples,
    })


class TestNebulentoPipelineIntent4(unittest.TestCase):
    """OVOS-INTENT-4 template registration surface (alongside legacy)."""

    def setUp(self):
        self.bus = mock.Mock()
        self.pipeline = NebulentoPipeline(bus=self.bus, config={
            "conf_high": 0.95, "conf_med": 0.8, "conf_low": 0.5,
        })

    def test_register_template_intent_then_fuzzy_match(self):
        self.pipeline.handle_register_template(_spec_register_template_msg(
            "test_skill", "HelloIntent",
            ["hello", "hi", "how are you", "hey there"],
        ))
        # intent stored under the combined skill_id:intent_name label
        self.assertIn("test_skill:HelloIntent",
                      self.pipeline.registered_intents)
        msg = Message("recognizer_loop:utterance",
                      {"utterances": ["hello"], "lang": "en-US"})
        result = self.pipeline.match_high(["hello"], "en-US", msg)
        self.assertIsNotNone(result)
        self.assertEqual(result.match_type, "test_skill:HelloIntent")
        self.assertEqual(result.skill_id, "test_skill")

    def test_register_template_fuzzy_inexact(self):
        """A fuzzy (non-exact) utterance still matches via medium confidence."""
        self.pipeline.handle_register_template(_spec_register_template_msg(
            "test_skill", "GreetIntent",
            ["good morning", "good evening", "good afternoon"],
        ))
        msg = Message("recognizer_loop:utterance",
                      {"utterances": ["goodd morning"], "lang": "en-US"})
        result = self.pipeline.match_medium(["goodd morning"], "en-US", msg)
        self.assertIsNotNone(result)
        self.assertEqual(result.match_type, "test_skill:GreetIntent")

    def test_register_template_with_entity(self):
        self.pipeline = NebulentoPipeline(bus=mock.Mock(), config={
            "conf_high": 0.9, "conf_med": 0.5, "conf_low": 0.3,
            "strategy": "TOKEN_SET_RATIO",
        })
        self.pipeline.handle_register_template(_spec_register_template_msg(
            "shop_skill", "BuyIntent",
            ["buy {item}", "purchase {item}", "get {item} for me"],
        ))
        self.pipeline.handle_register_entity_spec(_spec_register_entity_msg(
            "shop_skill", "item", ["milk", "cheese", "bread"],
        ))
        msg = Message("recognizer_loop:utterance",
                      {"utterances": ["buy milk"], "lang": "en-US"})
        result = self.pipeline.match_medium(["buy milk"], "en-US", msg)
        self.assertIsNotNone(result)
        self.assertEqual(result.match_type, "shop_skill:BuyIntent")

    def test_legacy_and_spec_handlers_coexist(self):
        """Legacy padatious registration still works after INTENT-4 wiring."""
        self.pipeline.register_intent(_register_intent_msg(
            "legacy_skill:ByeIntent", ["goodbye", "bye", "see you"],
        ))
        self.pipeline.handle_register_template(_spec_register_template_msg(
            "spec_skill", "HiIntent", ["hello", "hi there"],
        ))
        self.assertIn("legacy_skill:ByeIntent",
                      self.pipeline.registered_intents)
        self.assertIn("spec_skill:HiIntent",
                      self.pipeline.registered_intents)

    def test_deregister_intent_spec_removes_match(self):
        self.pipeline.handle_register_template(_spec_register_template_msg(
            "test_skill", "HelloIntent", ["hello", "hi", "hey there"],
        ))
        self.pipeline.handle_deregister_intent_spec(Message(
            "ovos.intent.deregister",
            {"skill_id": "test_skill", "intent_name": "HelloIntent",
             "lang": "en-US"},
        ))
        msg = Message("recognizer_loop:utterance",
                      {"utterances": ["hello"], "lang": "en-US"})
        result = self.pipeline.match_high(["hello"], "en-US", msg)
        self.assertIsNone(result)

    def test_deregister_skill_spec_removes_all(self):
        self.pipeline.handle_register_template(_spec_register_template_msg(
            "test_skill", "HelloIntent", ["hello", "hi", "hey there"],
        ))
        self.pipeline.handle_deregister_skill_spec(Message(
            "ovos.skill.deregister", {"skill_id": "test_skill"},
        ))
        msg = Message("recognizer_loop:utterance",
                      {"utterances": ["hello"], "lang": "en-US"})
        result = self.pipeline.match_high(["hello"], "en-US", msg)
        self.assertIsNone(result)

    def test_disable_then_enable_intent(self):
        self.pipeline.handle_register_template(_spec_register_template_msg(
            "test_skill", "HelloIntent", ["hello", "hi", "hey there"],
        ))
        disable = Message("ovos.intent.disable",
                          {"skill_id": "test_skill",
                           "intent_name": "HelloIntent", "lang": "en-US"})
        self.pipeline.handle_disable_intent_spec(disable)
        msg = Message("recognizer_loop:utterance",
                      {"utterances": ["hello"], "lang": "en-US"})
        self.assertIsNone(self.pipeline.match_high(["hello"], "en-US", msg))
        # re-enable restores match candidacy
        self.pipeline.handle_enable_intent_spec(Message(
            "ovos.intent.enable",
            {"skill_id": "test_skill", "intent_name": "HelloIntent",
             "lang": "en-US"}))
        result = self.pipeline.match_high(["hello"], "en-US", msg)
        self.assertIsNotNone(result)
        self.assertEqual(result.match_type, "test_skill:HelloIntent")

    def test_malformed_template_missing_skill_id_ignored(self):
        self.pipeline.handle_register_template(Message(
            "ovos.intent.register.template",
            {"intent_name": "Orphan", "lang": "en-US", "samples": ["hi"]}))
        self.assertEqual(self.pipeline.registered_intents, [])

    def test_empty_samples_template_ignored(self):
        self.pipeline.handle_register_template(_spec_register_template_msg(
            "test_skill", "EmptyIntent", [],
        ))
        self.assertNotIn("test_skill:EmptyIntent",
                         self.pipeline.registered_intents)

    def test_shutdown_removes_spec_handlers(self):
        bus = mock.Mock()
        pipeline = NebulentoPipeline(bus=bus, config={})
        pipeline.shutdown()
        removed = {call.args[0] for call in bus.remove.call_args_list}
        self.assertIn("ovos.intent.register.template", removed)
        self.assertIn("ovos.entity.register", removed)
        self.assertIn("ovos.intent.disable", removed)


if __name__ == "__main__":
    unittest.main()
