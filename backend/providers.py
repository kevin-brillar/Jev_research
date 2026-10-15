"""Decision providers: where the numbers come from.

Three implementations, one contract (`answer` -> `Answers`):

* `LiveProvider`  - TypeSafe System One over the official SDK, typed accessors,
                    model id pinned from config, request id kept for support.
* `MockProvider`  - ordered rules from config/mock_rules.yaml. Data, not logic:
                    the same rule grammar covers text, attachments, auth and any
                    state path, so a scenario is added by editing YAML.
* `FixtureProvider` - recorded answers supplied by the caller. This is what makes
                    shadow evaluation possible: the evaluate node replays a
                    question set against labelled cases with no network.

Because all three return a normalized `Answers`, the policy engine, every node and
every test behave identically in all three modes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol

from backend import config
from backend.answers import Answers, answers_from_questions
from backend.questions import QuestionSet

MOCK_MODEL = "mock/jev"


class Provider(Protocol):
    async def answer(self, question_set: QuestionSet, state: dict[str, Any]) -> Answers: ...


# --------------------------------------------------------------------------- mock


@dataclass
class MockProvider:
    """Deterministic provider driven by config/mock_rules.yaml."""

    rules_document: dict[str, Any] | None = None
    model: str = MOCK_MODEL

    def __post_init__(self) -> None:
        self._doc = self.rules_document if self.rules_document is not None else config.mock_rules()

    # ------------------------------------------------------------ rule matching
    @staticmethod
    def _lookup(state: dict[str, Any], path: str) -> Any:
        value: Any = state
        for part in path.split("."):
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return None
        return value

    def _matches(self, when: dict[str, Any], state: dict[str, Any]) -> bool:
        if when.get("always"):
            return True
        text = str(self._lookup(state, "request.text") or "")
        haystack = text.lower()

        if "text_any" in when and not any(str(word).lower() in haystack for word in when["text_any"]):
            return False
        if "text_all" in when and not all(str(word).lower() in haystack for word in when["text_all"]):
            return False
        if "regex" in when and not re.search(str(when["regex"]), text, re.IGNORECASE):
            return False
        if "attachments" in when:
            attached = bool(self._lookup(state, "request.attachment_names"))
            if bool(when["attachments"]) != attached:
                return False
        if "authenticated" in when and bool(when["authenticated"]) != bool(
            self._lookup(state, "request.authenticated")
        ):
            return False
        for path, expected in (when.get("state") or {}).items():
            if self._lookup(state, path) != expected:
                return False
        return True

    def _answer_for(self, set_name: str, state: dict[str, Any]) -> dict[str, Any]:
        set_doc = (self._doc.get("sets") or {}).get(set_name)
        if set_doc is None:
            raise KeyError(f"mock_rules.yaml has no set '{set_name}'")
        for rule in set_doc.get("rules") or []:
            if self._matches(rule.get("when") or {}, state):
                return rule["answer"]
        default = set_doc.get("default")
        if default is None:
            raise KeyError(f"mock_rules.yaml set '{set_name}' has no default and no rule matched")
        return default

    # ------------------------------------------------------------------- public
    async def answer(self, question_set: QuestionSet, state: dict[str, Any]) -> Answers:
        answer = self._answer_for(question_set.name, state)
        kinds = {question.kind: name for name, question in question_set.questions.items()}
        choice = str(answer.get("choice", ""))
        raw = {
            kinds.get("choice", "choice"): {
                "choice": choice,
                "confidence": float(answer.get("confidence", 0.0)),
                "probabilities": {choice: float(answer.get("confidence", 0.0))},
            },
            kinds.get("score", "score"): {"score": float(answer.get("score", 0.0))},
            kinds.get("noul", "noul"): {"noul": float(answer.get("noul", 0.0))},
        }
        return answers_from_questions(kinds, raw, model=self.model)


# -------------------------------------------------------------------------- fixture


@dataclass
class FixtureProvider:
    """Replay provider: answers supplied up front, keyed by question set.

    Used by the shadow-evaluation node and by tests that assert exact policy
    behaviour without depending on rule ordering.
    """

    answers: dict[str, dict[str, Any]]
    model: str = "fixture/jev"

    async def answer(self, question_set: QuestionSet, state: dict[str, Any]) -> Answers:
        payload = self.answers.get(question_set.name)
        if payload is None:
            raise KeyError(f"FixtureProvider has no answers for set '{question_set.name}'")
        kinds = {question.kind: name for name, question in question_set.questions.items()}
        choice = str(payload.get("choice", ""))
        raw = {
            kinds.get("choice", "choice"): {
                "choice": choice,
                "confidence": float(payload.get("confidence", 0.0)),
                "probabilities": {choice: float(payload.get("confidence", 0.0))},
            },
            kinds.get("score", "score"): {"score": float(payload.get("score", 0.0))},
            kinds.get("noul", "noul"): {"noul": float(payload.get("noul", 0.0))},
        }
        return answers_from_questions(kinds, raw, model=self.model)


# ----------------------------------------------------------------------------- live


class LiveProvider:
    """TypeSafe System One via the official SDK.

    The SDK is imported lazily so every offline path (mock, fixture, tests) works
    without credentials or the package installed. Model ids are pinned from
    config, because a threshold tuned against `${model}-latest` silently moves.
    """

    def __init__(self, model: str | None = None, base_url: str | None = None) -> None:
        import os

        self.model = model or os.getenv("TYPESAFE_DEFAULT_MODEL", "jev-1.13.0")
        self.base_url = base_url or os.getenv("TYPESAFE_BASE_URL")

    @staticmethod
    def _build_questions(question_set: QuestionSet) -> dict[str, Any]:
        from typesafe_sdk import Choice, Noul, Score

        builders = {"choice": Choice, "score": Score, "noul": Noul}
        questions: dict[str, Any] = {}
        for name, question in question_set.questions.items():
            builder = builders[question.kind]
            payload = {"instructions": question.instructions}
            if question.criteria is not None:
                payload["criteria"] = question.criteria
            questions[name] = builder(**payload)
        return questions

    async def answer(self, question_set: QuestionSet, state: dict[str, Any]) -> Answers:
        from typesafe_sdk import AsyncTypeSafeClient

        client_kwargs: dict[str, Any] = {}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        async with AsyncTypeSafeClient(**client_kwargs) as client:
            response = await client.system_one(
                state=state,
                questions=self._build_questions(question_set),
                model=self.model,
            )

        # Typed accessors on SystemOneResponse (SDK >= 0.7.1); the generic
        # `answers` map stays the fallback so a shape change cannot break both.
        def pick(kind: str, name: str) -> Any:
            for attribute in (f"{kind}s", "answers"):
                container = getattr(response, attribute, None)
                if isinstance(container, dict) and name in container:
                    return container[name]
            return None

        kinds = {question.kind: name for name, question in question_set.questions.items()}
        raw: dict[str, dict[str, Any]] = {}
        for kind, name in kinds.items():
            answer = pick(kind, name)
            if answer is None:
                continue
            dumped = answer.model_dump(mode="json") if hasattr(answer, "model_dump") else dict(answer)
            raw[name] = dumped

        return answers_from_questions(
            kinds,
            raw,
            model=getattr(response, "model", self.model),
            request_id=getattr(response, "request_id", None),
        )


def build_provider(mode: str | None = None, **kwargs: Any) -> Provider:
    """Pick a provider from JEV_MODE: "live" -> TypeSafe, anything else -> mock."""
    import os

    resolved = (mode or os.getenv("JEV_MODE", "mock")).lower()
    if resolved == "live":
        return LiveProvider(**kwargs)
    if resolved == "fixture":
        return FixtureProvider(**kwargs)
    return MockProvider(**kwargs)
