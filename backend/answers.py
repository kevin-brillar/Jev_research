"""Normalized answers: the one shape every provider must produce.

Live Jev, the deterministic mock and a recorded fixture all collapse into this
dataclass, so the policy engine, the nodes and the tests never branch on which
provider answered. A provider that cannot fill a required field raises
`IncompleteAnswers`, which the decision core turns into a fail-closed verdict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class IncompleteAnswers(Exception):
    """The provider did not return the judgments the question set asked for."""


@dataclass
class Answers:
    """Typed answers for one decision.

    Attributes:
        choice: selected option for the set's choice question (route key).
        confidence: distribution concentration for that choice, 0-1.
        risk: fractional score for the set's score question, on the level scale.
        human: probability the noul question is yes, 0-1.
        probabilities: the choice's full distribution, kept for audit and tuning.
        by_question: raw per-question payloads keyed by question name.
        model: model id that answered (or "mock"), for the ledger.
        request_id: provider request id when available.
    """

    choice: str
    confidence: float
    risk: float
    human: float
    probabilities: dict[str, float] = field(default_factory=dict)
    by_question: dict[str, dict[str, Any]] = field(default_factory=dict)
    model: str = "unknown"
    request_id: str | None = None

    def numbers(self) -> dict[str, float]:
        return {
            "confidence": round(self.confidence, 4),
            "risk": round(self.risk, 4),
            "human_probability": round(self.human, 4),
        }


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def answers_from_questions(
    question_names: dict[str, str],
    raw: dict[str, dict[str, Any]],
    *,
    model: str,
    request_id: str | None = None,
) -> Answers:
    """Build `Answers` from per-question payloads keyed by question name.

    Args:
        question_names: mapping of primitive kind -> question name for the set
            ({"choice": "route", "score": "risk", "noul": "needs_human"}).
        raw: provider payloads keyed by question name, each carrying its own
            primitive field ("choice" / "score" / "noul") plus "confidence".
        model: model id to record.
        request_id: provider request id when available.

    Raises:
        IncompleteAnswers: a question the set asked for is missing or malformed.
    """

    def payload(kind: str) -> dict[str, Any]:
        name = question_names.get(kind)
        if not name:
            raise IncompleteAnswers(f"question set declares no {kind} question")
        body = raw.get(name)
        if not isinstance(body, dict):
            raise IncompleteAnswers(f"no answer for {kind} question '{name}'")
        return body

    choice_body = payload("choice")
    score_body = payload("score")
    noul_body = payload("noul")

    selected = choice_body.get("choice")
    if not isinstance(selected, str) or not selected:
        raise IncompleteAnswers(f"choice answer has no selection: {choice_body!r}")

    return Answers(
        choice=selected,
        confidence=_as_float(choice_body.get("confidence")),
        risk=_as_float(score_body.get("score")),
        human=_as_float(noul_body.get("noul")),
        probabilities={
            str(key): _as_float(value)
            for key, value in (choice_body.get("probabilities") or {}).items()
        },
        by_question=raw,
        model=model,
        request_id=request_id,
    )
