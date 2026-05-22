"""Intent service wrapping Nebulento."""

from functools import lru_cache
from os.path import isfile
from typing import Optional, Dict, List, Union

from langcodes import closest_match
from ovos_bus_client.client import MessageBusClient
from ovos_bus_client.message import Message
from ovos_bus_client.session import SessionManager, Session
from ovos_config.config import Configuration
from ovos_plugin_manager.templates.pipeline import ConfidenceMatcherPipeline, IntentHandlerMatch
from ovos_utils import flatten_list
from ovos_utils.fakebus import FakeBus
from ovos_utils.lang import standardize_lang_tag
from ovos_utils.log import LOG

from nebulento import HierarchicalIntentContainer, IntentContainer, MatchStrategy


class NebulentoIntent:
    """
    A set of data describing how a query fits into an intent.

    Attributes:
        name (str): Name of matched intent
        sent (str): The input utterance associated with the intent
        conf (float): Confidence in [0.0, 1.0]
        matches (dict): Entity name → extracted value
    """

    def __init__(self, name, sent, matches=None, conf=0.0):
        self.name = name
        self.sent = sent
        self.matches = matches or {}
        self.conf = conf

    def __getitem__(self, item):
        return self.matches[item]

    def __contains__(self, item):
        return item in self.matches

    def get(self, key, default=None):
        return self.matches.get(key, default)

    def __repr__(self):
        return repr(self.__dict__)


class NebulentoPipeline(ConfidenceMatcherPipeline):
    """OVOS pipeline plugin for Nebulento fuzzy intent matching."""

    def __init__(self, bus: Optional[Union[MessageBusClient, FakeBus]] = None,
                 config: Optional[Dict] = None):
        if config is None:
            config = Configuration().get("intents", {}).get("nebulento", {})
        super().__init__(config=config, bus=bus)

        core_config = Configuration()
        self.lang = standardize_lang_tag(core_config.get("lang", "en-US"))
        langs = core_config.get("secondary_langs") or []
        if self.lang not in langs:
            langs.append(self.lang)
        langs = [standardize_lang_tag(l) for l in langs]

        self.conf_high = self.config.get("conf_high") or 0.95
        self.conf_med = self.config.get("conf_med") or 0.8
        self.conf_low = self.config.get("conf_low") or 0.5
        self.max_words = self.config.get("max_words") or 50

        # Allow strategy to be configured via mycroft.conf:
        #   "nebulento": {"strategy": "TOKEN_SET_RATIO"}
        strategy_name = self.config.get("strategy", "DAMERAU_LEVENSHTEIN_SIMILARITY")
        try:
            self.strategy = MatchStrategy[strategy_name]
        except KeyError:
            LOG.warning(f"Unknown nebulento strategy {strategy_name!r}, falling back to DAMERAU_LEVENSHTEIN_SIMILARITY")
            self.strategy = MatchStrategy.DAMERAU_LEVENSHTEIN_SIMILARITY

        self.containers = {lang: self._build_container() for lang in langs}

        self.bus.on("padatious:register_intent", self.register_intent)
        self.bus.on("padatious:register_entity", self.register_entity)
        self.bus.on("detach_intent", self.handle_detach_intent)
        self.bus.on("detach_skill", self.handle_detach_skill)
        self.bus.on("mycroft.skills.train", self.train)

        self.registered_intents: List[str] = []
        self.registered_entities: List[dict] = []
        LOG.debug("Loaded Nebulento pipeline")

    def train(self, message=None):
        """No training required — emit trained signal immediately."""
        self.bus.emit(Message("mycroft.skills.trained"))

    # ── container-shape hooks — overridden by HierarchicalNebulentoPipeline ──

    def _build_container(self) -> IntentContainer:
        """Create the per-language container instance."""
        return IntentContainer(fuzzy_strategy=self.strategy)

    def _add_intent(self, container: IntentContainer, name: str,
                    samples: List[str]) -> None:
        """Register intent *name* with *samples* on *container*."""
        container.add_intent(name, samples)

    def _add_entity(self, container: IntentContainer, name: str,
                    samples: List[str]) -> None:
        """Register entity *name* with *samples* on *container*."""
        container.add_entity(name, samples)

    def _remove_intent(self, container: IntentContainer, name: str) -> None:
        """Remove intent *name* from *container*."""
        container.remove_intent(name)

    def _remove_entity(self, container: IntentContainer, name: str) -> None:
        """Remove entity *name* from *container*."""
        container.remove_entity(name)

    def _remove_skill(self, container: IntentContainer, skill_id: str) -> None:
        """Remove anything left for *skill_id* after per-intent detach.

        No-op for the flat container — intents and entities are already
        removed individually by :meth:`handle_detach_skill`.
        """

    # ── confidence-level matchers ──────────────────────────────────────────

    def _match_level(self, utterances, limit, lang=None,
                     message: Optional[Message] = None) -> Optional[IntentHandlerMatch]:
        LOG.debug(f"Nebulento matching confidence > {limit}")
        utterances = flatten_list(utterances)
        lang = standardize_lang_tag(lang or self.lang)
        intent = self.calc_intent(utterances, lang, message)
        if intent is not None and intent.conf > limit:
            skill_id = intent.name.split(":")[0]
            return IntentHandlerMatch(match_type=intent.name,
                                     match_data=intent.matches,
                                     skill_id=skill_id,
                                     utterance=intent.sent)

    def match_high(self, utterances: List[str], lang: str, message: Message) -> Optional[IntentHandlerMatch]:
        return self._match_level(utterances, self.conf_high, lang, message)

    def match_medium(self, utterances: List[str], lang: str, message: Message) -> Optional[IntentHandlerMatch]:
        return self._match_level(utterances, self.conf_med, lang, message)

    def match_low(self, utterances: List[str], lang: str, message: Message) -> Optional[IntentHandlerMatch]:
        return self._match_level(utterances, self.conf_low, lang, message)

    # ── registration ───────────────────────────────────────────────────────

    def _register_object(self, message, object_name, register_func):
        file_name = message.data.get("file_name")
        samples = message.data.get("samples")
        name = message.data["name"]
        LOG.debug(f"Registering Nebulento {object_name}: {name}")

        if (not file_name or not isfile(file_name)) and not samples:
            LOG.error(f"Could not find file {file_name}")
            return

        if not samples and isfile(file_name):
            with open(file_name) as f:
                samples = [line.strip() for line in f if line.strip()]

        register_func(name, samples)

    def register_intent(self, message):
        lang = standardize_lang_tag(message.data.get("lang", self.lang))
        if lang not in self.containers:
            return
        name = message.data["name"]
        self.registered_intents.append(name)
        container = self.containers[lang]
        try:
            self._register_object(
                message, "intent",
                lambda n, s: self._add_intent(container, n, s))
        except RuntimeError:
            # skill reload — intent already registered, skip silently
            pass

    def register_entity(self, message):
        lang = standardize_lang_tag(message.data.get("lang", self.lang))
        if lang not in self.containers:
            return
        self.registered_entities.append(message.data)
        container = self.containers[lang]
        try:
            self._register_object(
                message, "entity",
                lambda n, s: self._add_entity(container, n, s))
        except RuntimeError:
            # skill reload — entity already registered, skip silently
            pass

    # ── detach ─────────────────────────────────────────────────────────────

    def _detach_intent(self, intent_name: str):
        if intent_name in self.registered_intents:
            self.registered_intents.remove(intent_name)
            for container in self.containers.values():
                self._remove_intent(container, intent_name)

    def _detach_entity(self, name: str, lang: str):
        if lang in self.containers:
            self._remove_entity(self.containers[lang], name)

    def handle_detach_intent(self, message):
        self._detach_intent(message.data.get("intent_name"))

    def handle_detach_skill(self, message):
        skill_id = message.data["skill_id"]
        for intent_name in [i for i in self.registered_intents if i.startswith(skill_id)]:
            self._detach_intent(intent_name)
        skill_id_colon = skill_id + ":"
        for en in self.registered_entities:
            if en["name"].startswith(skill_id_colon):
                self._detach_entity(en["name"], en.get("lang", self.lang))
        for container in self.containers.values():
            self._remove_skill(container, skill_id)

    # ── intent calculation ─────────────────────────────────────────────────

    def calc_intent(self, utterances: List[str], lang: str = None,
                    message: Optional[Message] = None) -> Optional[NebulentoIntent]:
        if isinstance(utterances, str):
            utterances = [utterances]
        utterances = [u for u in utterances if len(u.split()) < self.max_words]
        if not utterances:
            LOG.error(f"All utterances exceed max_words={self.max_words}, skipping")
            return None

        lang = self._get_closest_lang(lang or self.lang)
        if lang is None:
            return None

        sess = SessionManager.get(message)
        container = self.containers[lang]

        results = [_calc_nebulento_intent(utt, container, sess) for utt in utterances]
        results = [r for r in results if r is not None]
        return max(results, key=lambda r: r.conf) if results else None

    def _get_closest_lang(self, lang: str) -> Optional[str]:
        if not self.containers:
            return None
        lang = standardize_lang_tag(lang)
        closest, score = closest_match(lang, list(self.containers.keys()))
        return closest if score < 10 else None

    def shutdown(self):
        self.bus.remove("padatious:register_intent", self.register_intent)
        self.bus.remove("padatious:register_entity", self.register_entity)
        self.bus.remove("detach_intent", self.handle_detach_intent)
        self.bus.remove("detach_skill", self.handle_detach_skill)
        self.bus.remove("mycroft.skills.train", self.train)


class HierarchicalNebulentoPipeline(NebulentoPipeline):
    """Hierarchical Nebulento pipeline using two-stage domain routing.

    Same behaviour and bus surface as :class:`NebulentoPipeline` except the
    per-language container is a :class:`HierarchicalIntentContainer`. Each
    registered intent is filed under a domain == ``skill_id`` (taken from the
    intent name's ``<skill_id>:<intent>`` prefix); at match time a top-level
    classifier picks the domain and only that domain's sub-container resolves
    the intent. Utterances that match no domain are rejected before any
    sub-container runs.

    Configuration is read from ``intents.nebulento_hierarchical`` so this plugin
    can coexist with the flat plugin in the same OVOS instance. Accepts every
    key the flat plugin does, plus ``domain_threshold`` — the minimum top-level
    classifier confidence required to route a query (``0.0`` disables the gate).
    """

    def __init__(self, bus: Optional[Union[MessageBusClient, FakeBus]] = None,
                 config: Optional[Dict] = None):
        if config is None:
            config = Configuration().get("intents", {}).get("nebulento_hierarchical", {})
        # set before super().__init__ — _build_container() runs inside it
        self.domain_threshold = (config or {}).get("domain_threshold", 0.0)
        super().__init__(bus=bus, config=config)

    # ── container-shape hooks ───────────────────────────────────────────────

    @staticmethod
    def _domain_of(name: str) -> str:
        """Extract the domain (skill_id) from a ``skill_id:name`` label."""
        return name.split(":", 1)[0] if ":" in name else name

    def _build_container(self) -> HierarchicalIntentContainer:
        return HierarchicalIntentContainer(fuzzy_strategy=self.strategy,
                                           domain_threshold=self.domain_threshold)

    def _add_intent(self, container: HierarchicalIntentContainer, name: str,
                    samples: List[str]) -> None:
        container.register_domain_intent(self._domain_of(name), name, samples)

    def _add_entity(self, container: HierarchicalIntentContainer, name: str,
                    samples: List[str]) -> None:
        container.register_domain_entity(self._domain_of(name), name, samples)

    def _remove_intent(self, container: HierarchicalIntentContainer,
                       name: str) -> None:
        container.remove_domain_intent(self._domain_of(name), name)

    def _remove_entity(self, container: HierarchicalIntentContainer,
                       name: str) -> None:
        container.remove_domain_entity(self._domain_of(name), name)

    def _remove_skill(self, container: HierarchicalIntentContainer,
                      skill_id: str) -> None:
        # in hierarchical mode the skill_id IS the domain
        container.remove_domain(skill_id)


@lru_cache(maxsize=128)  # covers burst of multiple ASR hypotheses without thrashing
def _calc_nebulento_intent(utt: str,
                           container: IntentContainer,
                           sess: Session) -> Optional[NebulentoIntent]:
    """Match one utterance against the container, respecting session blacklists."""
    try:
        result = container.calc_intent(utt)
        if result is None or not result.get("name"):
            return None
        if result["name"] in sess.blacklisted_intents:
            return None
        if result["name"].split(":")[0] in sess.blacklisted_skills:
            return None
        return NebulentoIntent(
            name=result["name"],
            sent=utt,
            conf=result["conf"],
            matches=result.get("entities", {}),
        )
    except Exception as e:
        LOG.error(e)
        return None
