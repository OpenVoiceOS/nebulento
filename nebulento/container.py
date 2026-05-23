"""Core intent-matching container."""

import re
import logging
from typing import Dict, Iterator, List, Optional

from ovos_spec_tools import expand as expand_template

from nebulento.fuzz import MatchStrategy, match_one
from nebulento.bracket_expansion import normalize_example, normalize_utterance
import quebra_frases

LOG = logging.getLogger("nebulento")

#: Type alias for a single intent-match result dict.
MatchResult = Dict[str, object]


class IntentContainer:
    """Fuzzy intent matcher backed by rapidfuzz similarity strategies.

    Registers intent templates and entity samples, then scores an utterance
    against every registered intent using the chosen :class:`~nebulento.fuzz.MatchStrategy`.

    Example::

        from nebulento import IntentContainer, MatchStrategy

        c = IntentContainer(fuzzy_strategy=MatchStrategy.TOKEN_SET_RATIO)
        c.add_intent("play", ["play {song}", "put on {song}"])
        c.add_entity("song", ["jazz", "rock"])
        result = c.calc_intent("play some jazz")
        # result["name"] == "play", result["entities"] == {"song": ["jazz"]}

    Args:
        fuzzy_strategy: Similarity algorithm used for all matches.
            Defaults to :attr:`~MatchStrategy.DAMERAU_LEVENSHTEIN_SIMILARITY`
            (lowest false-positive rate of the fuzzy strategies).
        ignore_case: When ``True`` (default) utterances and templates are
            lowercased before comparison.
    """

    def __init__(self, fuzzy_strategy: MatchStrategy = MatchStrategy.DAMERAU_LEVENSHTEIN_SIMILARITY,
                 ignore_case: bool = True) -> None:
        self.fuzzy_strategy = fuzzy_strategy
        self.ignore_case = ignore_case
        self.registered_intents: Dict[str, List[str]] = {}
        self.registered_entities: Dict[str, List[str]] = {}
        self.available_contexts: Dict[str, Dict[str, object]] = {}
        self.required_contexts: Dict[str, List[str]] = {}
        self.excluded_contexts: Dict[str, List[str]] = {}
        self.excluded_keywords: Dict[str, List[str]] = {}

    # ── properties ─────────────────────────────────────────────────────────

    @property
    def intent_names(self) -> List[str]:
        """Names of all currently registered intents."""
        return list(self.registered_intents)

    # ── internal helpers ────────────────────────────────────────────────────

    def _norm(self, text: str) -> str:
        """Normalise *text* for storage or comparison.

        Strips apostrophes (replacing with space), collapses whitespace, and
        optionally lowercases based on :attr:`ignore_case`.
        """
        text = normalize_utterance(text)
        if self.ignore_case:
            text = text.lower()
        return text

    # ── registration ───────────────────────────────────────────────────────

    def add_intent(self, name: str, lines: List[str]) -> None:
        """Register an intent with one or more training templates.

        Templates support ``(a|b)`` alternation, ``[optional]`` syntax, and
        ``{entity}`` capture placeholders.  All variants are expanded at
        registration time and duplicates are discarded.

        Args:
            name: Unique intent identifier, e.g. ``"skill_id:intent_name"``.
            lines: List of training templates.

        Raises:
            RuntimeError: If *name* is already registered.  Call
                :meth:`remove_intent` first.
        """
        if name in self.registered_intents:
            raise RuntimeError(f"Intent already registered: {name!r}. Call remove_intent first.")
        expanded = {
            self._norm(e)
            for line in lines
            for e in expand_template(normalize_example(line))
        }
        self.registered_intents[name] = list(expanded)

    def remove_intent(self, name: str) -> None:
        """Unregister an intent.  Silently does nothing if *name* is not registered.

        Args:
            name: Intent identifier to remove.
        """
        self.registered_intents.pop(name, None)

    def add_entity(self, name: str, lines: List[str]) -> None:
        """Register an entity with sample values used to boost match confidence.

        When the matched template contains ``{name}`` and the extracted value
        appears in the registered samples, confidence is boosted.

        Args:
            name: Entity name (case-insensitive, stored lowercase).
            lines: List of sample values.  Alternation syntax is expanded.

        Raises:
            RuntimeError: If *name* is already registered.
        """
        name = name.lower()
        if name in self.registered_entities:
            raise RuntimeError(f"Entity already registered: {name!r}. Call remove_entity first.")
        expanded = {
            self._norm(e)
            for line in lines
            for e in expand_template(line)
        }
        self.registered_entities[name] = list(expanded)

    def remove_entity(self, name: str) -> None:
        """Unregister an entity.  Silently does nothing if *name* is not registered.

        Args:
            name: Entity name to remove (case-insensitive).
        """
        self.registered_entities.pop(name.lower(), None)

    # ── context / keyword gating ───────────────────────────────────────────

    def set_context(self, intent_name: str, context_name: str,
                    context_val: object = None) -> None:
        """Mark a context as active for *intent_name*.

        Args:
            intent_name: Intent whose context availability is updated.
            context_name: Context key.
            context_val: Optional associated value (unused by the matcher itself).
        """
        self.available_contexts.setdefault(intent_name, {})[context_name] = context_val

    def unset_context(self, intent_name: str, context_name: str) -> None:
        """Remove an active context for *intent_name*.

        Args:
            intent_name: Target intent.
            context_name: Context key to remove.
        """
        if intent_name in self.available_contexts:
            self.available_contexts[intent_name].pop(context_name, None)

    def require_context(self, intent_name: str, context_name: str) -> None:
        """Gate *intent_name* so it only matches when *context_name* is active.

        Multiple calls accumulate requirements — all must be satisfied.

        Args:
            intent_name: Intent to gate.
            context_name: Required context key.
        """
        self.required_contexts.setdefault(intent_name, []).append(context_name)

    def unrequire_context(self, intent_name: str, context_name: str) -> None:
        """Remove a context requirement from *intent_name*.

        Args:
            intent_name: Target intent.
            context_name: Requirement to lift.
        """
        if intent_name in self.required_contexts:
            self.required_contexts[intent_name] = [
                c for c in self.required_contexts[intent_name] if c != context_name
            ]

    def exclude_context(self, intent_name: str, context_name: str) -> None:
        """Suppress *intent_name* whenever *context_name* is active.

        Args:
            intent_name: Intent to suppress.
            context_name: Context that triggers suppression.
        """
        self.excluded_contexts.setdefault(intent_name, []).append(context_name)

    def unexclude_context(self, intent_name: str, context_name: str) -> None:
        """Lift a context-based suppression from *intent_name*.

        Args:
            intent_name: Target intent.
            context_name: Suppression to remove.
        """
        if intent_name in self.excluded_contexts:
            self.excluded_contexts[intent_name] = [
                c for c in self.excluded_contexts[intent_name] if c != context_name
            ]

    def exclude_keywords(self, intent_name: str, samples: List[str]) -> None:
        """Suppress *intent_name* when any keyword in *samples* appears in the query.

        Single-word keywords use whole-word matching (not substring).  Multi-word
        keywords use a word-boundary regex so ``"play"`` does not trigger on
        ``"display"``.

        Args:
            intent_name: Intent to suppress.
            samples: Keywords or phrases whose presence blocks the intent.
        """
        self.excluded_keywords.setdefault(intent_name, [])
        self.excluded_keywords[intent_name] += samples

    # ── internal filtering ─────────────────────────────────────────────────

    def _filter(self, query: str) -> List[str]:
        """Return the list of intent names that should be skipped for *query*.

        Checks keyword exclusions, required contexts, and excluded contexts.
        """
        excluded: List[str] = []
        q_lower = query.lower()
        query_words = set(q_lower.split())

        for intent_name, keywords in self.excluded_keywords.items():
            def _hit(kw: str, _qw: set = query_words, _ql: str = q_lower) -> bool:
                if " " not in kw:
                    return kw.lower() in _qw
                return bool(re.search(r"\b" + re.escape(kw.lower()) + r"\b", _ql))
            if any(_hit(kw) for kw in keywords):
                excluded.append(intent_name)

        for intent_name, contexts in self.required_contexts.items():
            avail = self.available_contexts.get(intent_name, {})
            if any(c not in avail for c in contexts):
                excluded.append(intent_name)

        for intent_name, contexts in self.excluded_contexts.items():
            avail = self.available_contexts.get(intent_name, {})
            if any(c in avail for c in contexts):
                excluded.append(intent_name)

        return excluded

    # ── matching ───────────────────────────────────────────────────────────

    def match_entities(self, sentence: str) -> Dict[str, List[str]]:
        """Find registered entity values present in *sentence*.

        Uses ``quebra_frases.chunk`` for token-aware substring detection.

        Args:
            sentence: Normalised utterance to search.

        Returns:
            Dict mapping entity name → list of matched sample values found in
            *sentence*.  Empty list means the entity was not found.
        """
        sentence = self._norm(sentence)
        return {
            entity: [s for s in samples if s in quebra_frases.chunk(sentence, samples)]
            for entity, samples in self.registered_entities.items()
        }

    def match_fuzzy(self, sentence: str) -> Iterator[MatchResult]:
        """Yield one result dict per registered intent, scored against *sentence*.

        Each result contains:

        - ``name`` – intent name
        - ``conf`` – similarity score in ``[0.0, 1.0]``, boosted when registered
          entity values are found
        - ``entities`` – dict of entity name → matched values (lowercase keys)
        - ``best_match`` – the training template that scored highest
        - ``utterance`` – normalised query
        - ``utterance_consumed`` – space-joined words accounted for by the match
        - ``utterance_remainder`` – space-joined words not accounted for
        - ``match_strategy`` – name of the :class:`~MatchStrategy` used

        Intents excluded by context gates or keyword rules are skipped entirely.

        Args:
            sentence: Raw or pre-normalised utterance.

        Yields:
            One :class:`MatchResult` dict per non-excluded intent.
        """
        sentence = self._norm(sentence)
        entities = self.match_entities(sentence)
        excluded = self._filter(sentence)

        for intent, samples in self.registered_intents.items():
            if intent in excluded or not samples:
                continue

            sent, score = match_one(sentence, samples, strategy=self.fuzzy_strategy)
            sentence_tokens = quebra_frases.word_tokenize(sentence)
            sent_tokens = quebra_frases.word_tokenize(sent)
            remainder = [w for w in sentence_tokens if w not in sent_tokens]
            consumed = [w for w in sentence_tokens if w in sent_tokens]

            tagged: Dict[str, List[str]] = {}
            for ent, v in entities.items():
                if v and any("{" + ent + "}" in s for s in samples):
                    score = 0.25 + score * 0.75
                    tagged[ent] = v
                    consumed += [w for w in v if w not in consumed]
                    remainder = [w for w in remainder if w not in v]

            yield {
                "best_match": sent,
                "conf": min(score, 1.0),
                "entities": {k.lower(): v for k, v in tagged.items()},
                "match_strategy": self.fuzzy_strategy.name,
                "utterance": sentence,
                "utterance_remainder": " ".join(remainder),
                "utterance_consumed": " ".join(consumed),
                "name": intent,
            }

    # ── public API ─────────────────────────────────────────────────────────

    def calc_intents(self, query: str) -> Iterator[MatchResult]:
        """Yield a scored result for every registered intent.

        Results include all intents not suppressed by context/keyword gates,
        in registration order.

        Args:
            query: Raw utterance to evaluate.

        Yields:
            :class:`MatchResult` dicts, one per intent.
        """
        yield from self.match_fuzzy(query)

    def calc_intent(self, query: str) -> MatchResult:
        """Return the single best-matching intent for *query*.

        Tie-breaking order (most-to-least preferred):

        1. Higher ``conf`` score.
        2. More words consumed (more specific template).
        3. Intent name alphabetically (deterministic).

        When no intents are registered or all are suppressed, returns a result
        dict with ``name=None`` and all other keys present with empty/zero values.

        Args:
            query: Raw utterance to evaluate.

        Returns:
            Best-matching :class:`MatchResult`.  Always a dict with keys:
            ``name``, ``conf``, ``entities``, ``best_match``, ``utterance``,
            ``utterance_consumed``, ``utterance_remainder``, ``match_strategy``.
        """
        _empty: MatchResult = {
            "best_match": None,
            "conf": 0.0,
            "entities": {},
            "match_strategy": self.fuzzy_strategy.name,
            "utterance": self._norm(query),
            "utterance_consumed": "",
            "utterance_remainder": "",
            "name": None,
        }
        candidates = list(self.calc_intents(query))
        if not candidates:
            return _empty

        best_conf = max(r["conf"] for r in candidates)  # type: ignore[type-var]
        ties = [r for r in candidates if r["conf"] == best_conf]

        if len(ties) > 1:
            LOG.debug("tied intents: %s", [t["name"] for t in ties])
            ties.sort(key=lambda t: (
                -len(str(t["utterance_consumed"]).split()),
                t["name"],
            ))

        return ties[0]
