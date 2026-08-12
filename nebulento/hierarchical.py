"""Hierarchical intent container — two-stage domain routing."""

from collections import defaultdict
from typing import Dict, List, Optional

from nebulento.container import IntentContainer, MatchResult
from nebulento.fuzz import MatchStrategy


class HierarchicalIntentContainer:
    """Two-stage intent engine: domain classification followed by intent matching.

    Intents are grouped into *domains*.  At query time the engine first selects
    the most likely domain, then runs that domain's intent container to find the
    best intent within it.

    The top-level domain classifier is trained automatically: every sample
    passed to :meth:`register_domain_intent` is also fed to :attr:`domain_engine`
    under its domain name, so the container works standalone with no manual
    classifier setup.

    Domains can also be selected explicitly, bypassing the top-level classifier.

    Example::

        from nebulento import HierarchicalIntentContainer

        d = HierarchicalIntentContainer()
        d.register_domain_intent("media", "play", ["play {song}", "put on {song}"])
        d.register_domain_intent("home",  "lights_on", ["lights on", "turn on the lights"])

        result = d.calc_intent("play some jazz")
        # result["name"] == "play"

    Args:
        fuzzy_strategy: Similarity algorithm forwarded to every
            :class:`~nebulento.container.IntentContainer` created internally.
        ignore_case: Case-folding flag forwarded to child containers.
        domain_threshold: Minimum confidence the top-level classifier must
            reach for a query to be routed at all.  When the best domain
            scores below this, :meth:`calc_intent` returns a no-match instead
            of resolving an intent — this is the off-topic rejection gate.
            ``0.0`` (default) disables the gate; every query is routed to its
            best domain.
    """

    def __init__(self, fuzzy_strategy: MatchStrategy = MatchStrategy.DAMERAU_LEVENSHTEIN_SIMILARITY,
                 ignore_case: bool = True, domain_threshold: float = 0.0) -> None:
        self.fuzzy_strategy = fuzzy_strategy
        self.ignore_case = ignore_case
        self.domain_threshold = domain_threshold
        #: Top-level classifier that maps free-text queries to a domain name.
        self.domain_engine: IntentContainer = IntentContainer(
            fuzzy_strategy=fuzzy_strategy, ignore_case=ignore_case
        )
        #: Per-domain intent containers, keyed by domain name.
        self.domains: Dict[str, IntentContainer] = {}
        #: Raw training samples accumulated per domain (for inspection / re-training).
        self.training_data: Dict[str, List[str]] = defaultdict(list)
        #: Domains whose classifier entry is stale and must be rebuilt before a query.
        self._dirty_domains: set = set()

    # ── internal ───────────────────────────────────────────────────────────

    def _sync_domain_classifier(self) -> None:
        """Rebuild stale classifier entries.

        Registration only marks a domain dirty; the top-level classifier is
        rebuilt here, lazily, the first time a query needs it. This keeps bulk
        registration linear instead of re-expanding the whole corpus per call.
        """
        for domain_name in self._dirty_domains:
            if domain_name in self.domain_engine.intent_names:
                self.domain_engine.remove_intent(domain_name)
            samples = self.training_data.get(domain_name)
            if samples:
                self.domain_engine.add_intent(domain_name, samples)
        self._dirty_domains.clear()

    # ── domain management ──────────────────────────────────────────────────

    def remove_domain(self, domain_name: str) -> None:
        """Remove a domain and all its intents, entities, and training data.

        Args:
            domain_name: Domain to remove.
        """
        self.training_data.pop(domain_name, None)
        self.domains.pop(domain_name, None)
        self._dirty_domains.discard(domain_name)
        if domain_name in self.domain_engine.intent_names:
            self.domain_engine.remove_intent(domain_name)

    # ── intent management ──────────────────────────────────────────────────

    def register_domain_intent(self, domain_name: str, intent_name: str,
                                intent_samples: List[str]) -> None:
        """Register an intent inside a domain.

        Creates the domain's :class:`~nebulento.container.IntentContainer`
        on first use. The top-level domain classifier is marked stale and
        rebuilt lazily on the next query.

        Args:
            domain_name: Target domain (created if it does not exist).
            intent_name: Unique intent name within the domain.
            intent_samples: Training templates for the intent.
        """
        if domain_name not in self.domains:
            self.domains[domain_name] = IntentContainer(
                fuzzy_strategy=self.fuzzy_strategy, ignore_case=self.ignore_case
            )
        self.domains[domain_name].add_intent(intent_name, intent_samples)
        self.training_data[domain_name] += intent_samples
        self._dirty_domains.add(domain_name)

    def remove_domain_intent(self, domain_name: str, intent_name: str) -> None:
        """Remove a specific intent from a domain.

        Args:
            domain_name: Domain that owns the intent.
            intent_name: Intent to remove.
        """
        if domain_name in self.domains:
            self.domains[domain_name].remove_intent(intent_name)

    # ── entity management ──────────────────────────────────────────────────

    def register_domain_entity(self, domain_name: str, entity_name: str,
                                entity_samples: List[str]) -> None:
        """Register an entity inside a domain.

        Creates the domain's container on first use.

        Args:
            domain_name: Target domain.
            entity_name: Entity name.
            entity_samples: Sample values for the entity.
        """
        if domain_name not in self.domains:
            self.domains[domain_name] = IntentContainer(
                fuzzy_strategy=self.fuzzy_strategy, ignore_case=self.ignore_case
            )
        self.domains[domain_name].add_entity(entity_name, entity_samples)

    def remove_domain_entity(self, domain_name: str, entity_name: str) -> None:
        """Remove a specific entity from a domain.

        Args:
            domain_name: Domain that owns the entity.
            entity_name: Entity to remove.
        """
        if domain_name in self.domains:
            self.domains[domain_name].remove_entity(entity_name)

    def slot_names(self, intent_name: str) -> List[str]:
        """Return the declared ``{slot}`` names for *intent_name*.

        Searches every domain's sub-container, since an intent lives under a
        single domain but callers address it by its flat name.  Drives
        OVOS-CONTEXT-1 §7 context fill.  Empty list when unknown or slotless.
        """
        for domain in self.domains.values():
            slots = domain.slot_names(intent_name)
            if slots:
                return slots
        return []

    # ── query API ──────────────────────────────────────────────────────────

    def calc_domain(self, query: str) -> MatchResult:
        """Classify *query* into the best-matching domain.

        Args:
            query: Raw utterance to classify.

        Returns:
            :class:`~nebulento.container.MatchResult` dict whose ``name`` key
            is the predicted domain name (or ``None`` if no domain matched).
        """
        self._sync_domain_classifier()
        return self.domain_engine.calc_intent(query)

    def calc_intent(self, query: str,
                    domain: Optional[str] = None) -> MatchResult:
        """Return the best-matching intent for *query*, optionally within *domain*.

        If *domain* is ``None``, the domain is inferred by :meth:`calc_domain`.
        When the inferred domain scores below :attr:`domain_threshold`, or the
        inferred/supplied domain has no registered intents, a no-match result
        is returned.  Passing *domain* explicitly bypasses the classifier and
        the threshold gate.

        Args:
            query: Raw utterance to evaluate.
            domain: Domain to restrict matching to.  ``None`` triggers automatic
                domain classification.

        Returns:
            :class:`~nebulento.container.MatchResult` dict.  ``name`` is
            ``None`` when no domain or intent could be matched.
        """
        no_match: MatchResult = {
            "best_match": None,
            "conf": 0.0,
            "entities": {},
            "match_strategy": self.fuzzy_strategy.name,
            "utterance": query,
            "utterance_consumed": "",
            "utterance_remainder": "",
            "name": None,
        }

        resolved_domain: Optional[str] = domain
        if resolved_domain is None:
            self._sync_domain_classifier()
            dom_result = self.domain_engine.calc_intent(query)
            if float(dom_result.get("conf", 0.0)) < self.domain_threshold:  # type: ignore[arg-type]
                return no_match
            resolved_domain = dom_result.get("name")  # type: ignore[assignment]

        if resolved_domain in self.domains:
            return self.domains[resolved_domain].calc_intent(query)
        return no_match
