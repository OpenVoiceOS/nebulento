"""``detach_skill`` removes only the intents the named skill owns. The owning
skill of ``a.b:c`` is exactly ``a.b`` (colon namespace separator, OVOS-MSG-1
§2.1.1), so a skill whose id is a prefix of another skill's id must not take
the other skill's intents down with it."""
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


class TestDetachSkillPrefixMatch(unittest.TestCase):
    def setUp(self):
        self.bus = mock.Mock()
        self.pipeline = NebulentoPipeline(bus=self.bus, config={
            "conf_high": 0.95, "conf_med": 0.8, "conf_low": 0.5,
        })

    def _register(self, name):
        self.pipeline.register_intent(_register_intent_msg(
            name, ["hello " + name.split(":")[1]]))

    def test_prefix_skill_id_keeps_other_skills(self):
        for name in ("cal:add", "calendar:reminder"):
            self._register(name)
        self.pipeline.handle_detach_skill(Message("detach_skill", {"skill_id": "cal"}))
        self.assertEqual(sorted(self.pipeline.registered_intents),
                          ["calendar:reminder"])

    def test_longer_skill_id_removes_only_its_own_intents(self):
        for name in ("cal:add", "calendar:reminder"):
            self._register(name)
        self.pipeline.handle_detach_skill(Message("detach_skill", {"skill_id": "calendar"}))
        self.assertEqual(sorted(self.pipeline.registered_intents),
                          ["cal:add"])


if __name__ == "__main__":
    unittest.main()
