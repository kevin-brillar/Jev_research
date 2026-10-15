"""Policy engine: turns typed answers into an execution mode.

The model supplies numbers; this module decides what they mean. Every threshold,
route rule and diagnostic string comes from config/policy.yaml, so adding a route
or tightening a limit is a config change and the same engine gates every question
set (triage, authorization, verification, draft review).

Rule grammar, resolved from the most specific to the least:

    policy.sets.<set>.routes.<route>  ->  policy.sets.<set>.defaults
                                      ->  policy.defaults

Keys:
    min_confidence          review when the choice confidence is below it
    max_risk                review when the risk score is above it
    max_human_probability   review when the noul probability is above it
    always_human            review regardless of the numbers
    requires_gate           documentation: this route must pass that gate node
    fail_closed             overrides policy.on_error for this route
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from backend.answers import Answers
from backend import config

AUTOMATIC = "automatic"
REVIEW = "human_review"

DEFAULT_REASON_TEMPLATES: dict[str, str] = {
    "below_confidence": "route confidence {confidence} is below the {min_confidence} floor for {route}",
    "above_risk": "risk score {risk} is above the {max_risk} ceiling for {route}",
    "above_human": "human-review probability {human_probability} is above the {max_human_probability} ceiling for {route}",
    "always_human": "route {route} is declared always_human",
    "unknown_route": "route {route} is not declared in policy set {policy_set}",
    "requires_gate": "route {route} must pass the {requires_gate} gate",
    "fail_closed": "Jev could not be consulted ({error}); failing closed",
    "ok": "all thresholds passed for {route}",
}


@dataclass(frozen=True)
class RouteRule:
    min_confidence: float | None = None
    max_risk: float | None = None
    max_human_probability: float | None = None
    always_human: bool = False
    requires_gate: str | None = None
    fail_closed: bool | None = None
    unknown_route: str | None = None


@dataclass
class Verdict:
    question_set: str
    route: str
    execution_mode: str
    reasons: list[str] = field(default_factory=list)
    numbers: dict[str, float] = field(default_factory=dict)
    rule_source: str = ""

    @property
    def automatic(self) -> bool:
        return self.execution_mode == AUTOMATIC

    def as_dict(self) -> dict[str, Any]:
        return {
            "question_set": self.question_set,
            "recommended_route": self.route,
            "execution_mode": self.execution_mode,
            "reasons": list(self.reasons),
            **self.numbers,
        }


def _rule_from(mapping: dict[str, Any]) -> RouteRule:
    return RouteRule(
        min_confidence=mapping.get("min_confidence"),
        max_risk=mapping.get("max_risk"),
        max_human_probability=mapping.get("max_human_probability"),
        always_human=bool(mapping.get("always_human", False)),
        requires_gate=mapping.get("requires_gate"),
        fail_closed=mapping.get("fail_closed"),
        unknown_route=mapping.get("unknown_route"),
    )


class PolicyEngine:
    """Threshold evaluation over any question set, driven entirely by config."""

    def __init__(self, document: dict[str, Any] | None = None) -> None:
        self._doc = document if document is not None else config.policy()
        self._global_defaults = _rule_from(self._doc.get("defaults") or {})
        self._set_documents: dict[str, dict[str, Any]] = self._doc.get("sets") or {}
        self._templates = {**DEFAULT_REASON_TEMPLATES, **(self._doc.get("templates") or {})}

    # ------------------------------------------------------------------ public
    @property
    def on_error(self) -> str:
        return str(self._doc.get("on_error", REVIEW))

    def sets(self) -> list[str]:
        return sorted(self._set_documents)

    def evaluate(self, question_set: str, answers: Answers) -> Verdict:
        """Apply the thresholds for one question set to one set of answers."""
        set_doc = self._set_documents.get(question_set)
        if set_doc is None:
            return self._verdict(
                question_set,
                answers.choice,
                REVIEW,
                self._format(
                    "unknown_route",
                    route=answers.choice,
                    policy_set=question_set,
                    **answers.numbers(),
                ),
                answers,
                "policy.set.missing",
            )

        declared = set_doc.get("routes") or {}
        set_defaults = {**(set_doc.get("defaults") or {})}
        if declared:
            set_defaults.setdefault("unknown_route", self._global_defaults.unknown_route)

        if answers.choice in declared:
            rule = _rule_from({**self._mapping(self._global_defaults), **set_defaults, **declared[answers.choice]})
            source = f"policy.sets.{question_set}.routes.{answers.choice}"
        else:
            rule = _rule_from({**self._mapping(self._global_defaults), **set_defaults})
            source = f"policy.sets.{question_set}.defaults"

        if rule.always_human:
            return self._verdict(
                question_set,
                answers.choice,
                REVIEW,
                self._format("always_human", route=answers.choice, **answers.numbers()),
                answers,
                source,
            )

        # Routes that are not declared at all become a review, which is how a new
        # model option cannot silently gain automatic rights.
        if declared and answers.choice not in declared:
            return self._verdict(
                question_set,
                answers.choice,
                REVIEW,
                self._format(
                    "unknown_route",
                    route=answers.choice,
                    policy_set=question_set,
                    **answers.numbers(),
                ),
                answers,
                source,
            )

        reasons: list[str] = []
        numbers = answers.numbers()
        if rule.min_confidence is not None and numbers["confidence"] < float(rule.min_confidence):
            reasons.append(
                self._format("below_confidence", route=answers.choice, min_confidence=rule.min_confidence, **numbers)
            )
        if rule.max_risk is not None and numbers["risk"] > float(rule.max_risk):
            reasons.append(
                self._format("above_risk", route=answers.choice, max_risk=rule.max_risk, **numbers)
            )
        if rule.max_human_probability is not None and numbers["human_probability"] > float(rule.max_human_probability):
            reasons.append(
                self._format(
                    "above_human",
                    route=answers.choice,
                    max_human_probability=rule.max_human_probability,
                    **numbers,
                )
            )
        if rule.requires_gate:
            reasons.append(
                self._format("requires_gate", route=answers.choice, requires_gate=rule.requires_gate, **numbers)
            )

        if reasons:
            return self._verdict(question_set, answers.choice, REVIEW, "; ".join(reasons), answers, source)
        return self._verdict(
            question_set, answers.choice, AUTOMATIC, self._format("ok", route=answers.choice, **numbers), answers, source
        )

    def fail_closed(self, question_set: str, error: str) -> Verdict:
        """Verdict used when the provider could not answer at all."""
        mode = self.on_error
        set_doc = self._set_documents.get(question_set) or {}
        override = (set_doc.get("defaults") or {}).get("fail_closed")
        if override is not None:
            mode = AUTOMATIC if override is False else REVIEW
        return Verdict(
            question_set=question_set,
            route=REVIEW,
            execution_mode=mode,
            reasons=[self._format("fail_closed", error=error)],
            numbers={},
            rule_source="policy.on_error",
        )

    # --------------------------------------------------------------- internals
    @staticmethod
    def _mapping(rule: RouteRule) -> dict[str, Any]:
        return {
            key: value
            for key, value in {
                "min_confidence": rule.min_confidence,
                "max_risk": rule.max_risk,
                "max_human_probability": rule.max_human_probability,
                "always_human": rule.always_human,
            }.items()
            if value is not None
        }

    def _format(self, key: str, **values: Any) -> str:
        template = self._templates.get(key, DEFAULT_REASON_TEMPLATES.get(key, key))
        try:
            return template.format(**values)
        except (KeyError, IndexError):
            return template

    @staticmethod
    def _verdict(
        question_set: str,
        route: str,
        mode: str,
        reason: str,
        answers: Answers,
        source: str,
    ) -> Verdict:
        return Verdict(
            question_set=question_set,
            route=route,
            execution_mode=mode,
            reasons=[reason],
            numbers=answers.numbers(),
            rule_source=source,
        )
