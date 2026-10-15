"""Question sets and their deterministic linter.

A question set is the artifact the build-Jev workflow produces, so it must be
checkable by code rather than by opinion. `lint_question_set` enforces the parts
of the TypeSafe guidance that are mechanical:

* one judgment per question (a choice/score/noul, never a compound ask)
* instructions long enough to stand alone and free of code identifiers
* criteria written as concrete situations, not adjectives
* a choice offers an outcome for requests that fit nothing
* a score's levels are ordered and describe a scale
* backticked state paths resolve against the set's declared state fields
* noul instructions are yes/no questions

Everything it checks is a rule from config (see `lint:` in each question file), so
tightening the bar is a config change. Findings are returned, never raised: the
caller decides whether an error blocks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import jsonschema

from backend import config

PRIMITIVES = ("choice", "score", "noul")

# Words a yes/no judgment must open with; keeps nouls from drifting into scores.
NOUL_OPENERS = (
    "is", "are", "was", "were", "does", "do", "did", "has", "have", "had",
    "can", "could", "should", "would", "will", "must",
)

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["sets"],
    "properties": {
        "version": {"type": "string"},
        "lint": {"type": "object"},
        "sets": {
            "type": "object",
            "minProperties": 1,
            "additionalProperties": {
                "type": "object",
                "required": ["questions"],
                "properties": {
                    "description": {"type": "string"},
                    "state_fields": {"type": "array", "items": {"type": "string"}},
                    "questions": {
                        "type": "object",
                        "minProperties": 1,
                        "additionalProperties": {
                            "type": "object",
                            "required": ["type", "instructions"],
                            "properties": {
                                "type": {"enum": list(PRIMITIVES)},
                                "instructions": {"type": "string", "minLength": 3},
                                "criteria": {
                                    "type": ["object", "array", "string", "null"]
                                },
                            },
                        },
                    },
                },
            },
        },
    },
}


@dataclass(frozen=True)
class Question:
    name: str
    kind: str
    instructions: str
    criteria: Any = None
    lint: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Provider-shaped dict: the same shape the original scaffold used."""
        payload: dict[str, Any] = {"type": self.kind, "instructions": self.instructions}
        if self.criteria is not None:
            payload["criteria"] = self.criteria
        return payload


@dataclass
class QuestionSet:
    name: str
    questions: dict[str, Question]
    description: str = ""
    state_fields: list[str] = field(default_factory=list)
    lint_rules: dict[str, Any] = field(default_factory=dict)
    source: str = ""

    def as_dict(self) -> dict[str, dict[str, Any]]:
        return {name: question.as_dict() for name, question in self.questions.items()}

    def criteria_for(self, question_name: str) -> list[str]:
        """Choice keys or score levels, in declared order (nouls have none)."""
        criteria = self.questions[question_name].criteria
        if isinstance(criteria, dict):
            return list(criteria)
        if isinstance(criteria, list):
            return [str(item) for item in criteria]
        return []


@dataclass(frozen=True)
class Finding:
    severity: str  # "error" | "warning"
    rule: str
    question: str
    message: str


@dataclass
class LintReport:
    set_name: str
    findings: list[Finding] = field(default_factory=list)

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "error"]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, Any]:
        return {
            "set": self.set_name,
            "ok": self.ok,
            "errors": [f.__dict__ for f in self.errors],
            "warnings": [f.__dict__ for f in self.warnings],
        }


DEFAULT_LINT_RULES: dict[str, Any] = {
    "min_instruction_words": 5,
    "min_criteria_words": 4,
    "no_match_required": False,
    "no_match_options": [],
    "require_ordered_levels": True,
    "min_score_levels": 3,
}


def _words(text: str) -> list[str]:
    return [w for w in re.split(r"\s+", text.strip()) if w]


def parse_question_set(name: str, doc: dict[str, Any], lint_rules: dict[str, Any], source: str) -> QuestionSet:
    questions = {
        q_name: Question(
            name=q_name,
            kind=spec["type"],
            instructions=spec["instructions"],
            criteria=spec.get("criteria"),
            lint=dict(spec.get("lint") or {}),
        )
        for q_name, spec in (doc.get("questions") or {}).items()
    }
    return QuestionSet(
        name=name,
        questions=questions,
        description=doc.get("description", ""),
        state_fields=list(doc.get("state_fields") or []),
        lint_rules={**lint_rules, **(doc.get("lint") or {})},
        source=source,
    )


def load_library() -> dict[str, QuestionSet]:
    """Every question set from config/questions/*.yaml, validated against the schema."""
    library: dict[str, QuestionSet] = {}
    for path in config.question_files():
        doc = config.read_yaml(path)
        try:
            jsonschema.validate(doc, _SCHEMA)
        except jsonschema.ValidationError as exc:
            raise ValueError(f"{path.name}: invalid question-set file: {exc.message}") from exc
        rules = {**DEFAULT_LINT_RULES, **(doc.get("lint") or {})}
        for set_name, set_doc in (doc.get("sets") or {}).items():
            if set_name in library:
                raise ValueError(f"Duplicate question set '{set_name}' in {path.name}")
            library[set_name] = parse_question_set(set_name, set_doc, rules, path.name)
    return library


def get_question_set(name: str) -> QuestionSet:
    library = load_library()
    if name not in library:
        raise KeyError(f"Unknown question set '{name}'. Known: {sorted(library)}")
    return library[name]


_TEMPLATE_RE = re.compile(r"\{(?P<path>[A-Za-z0-9_.\[\]]+)\}")


def lint_question_set(question_set: QuestionSet) -> LintReport:
    """Mechanical checks over one question set. No model, no network, no opinion."""
    rules = {**DEFAULT_LINT_RULES, **question_set.lint_rules}
    report = LintReport(set_name=question_set.name)

    def add(severity: str, rule: str, question: str, message: str) -> None:
        report.findings.append(Finding(severity, rule, question, message))

    if not question_set.questions:
        add("error", "set.empty", "-", "question set has no questions")
        return report
    if not question_set.state_fields:
        add(
            "warning",
            "set.no_state",
            "-",
            "no state_fields declared; every judgment needs the facts it judges",
        )

    declared = [field_path.rstrip(".") for field_path in question_set.state_fields]

    for name, question in question_set.questions.items():
        # question-level rules win over set-level rules, which win over the file defaults
        question_rules = {**rules, **question.lint}
        if not name.isidentifier():
            add("error", "question.identifier", name, "question id must be a valid Python identifier")

        instruction = question.instructions or ""
        if len(_words(instruction)) < int(rules["min_instruction_words"]):
            add("error", "instruction.too_short", name, "instruction is too short to stand alone")
        if name in instruction.split():
            add(
                "warning",
                "instruction.uses_id",
                name,
                f"instruction contains the code identifier '{name}'; question ids are not sent to the model",
            )
        for match in _TEMPLATE_RE.finditer(instruction):
            path = match.group("path")
            if not any(path == field_path or path.startswith(f"{field_path}.") for field_path in declared):
                add(
                    "error",
                    "instruction.undeclared_state",
                    name,
                    f"instruction references '{path}' which is not in state_fields",
                )

        if question.kind == "choice":
            if not isinstance(question.criteria, dict) or len(question.criteria) < 2:
                add("error", "choice.criteria", name, "a choice needs at least two options")
                continue
            for option, text in question.criteria.items():
                if len(_words(str(text))) < int(rules["min_criteria_words"]):
                    add("error", "choice.criteria_text", name, f"option '{option}' is not a concrete situation")
            options = set(question.criteria)
            if question_rules["no_match_required"]:
                allowed = set(question_rules["no_match_options"])
                if not (options & allowed) and not any(
                    "nothing" in str(v).lower() or "no option" in str(v).lower()
                    for v in question.criteria.values()
                ):
                    add(
                        "error",
                        "choice.no_match",
                        name,
                        f"no outcome for requests that fit nothing (expected one of {sorted(allowed)})",
                    )

        elif question.kind == "score":
            if not isinstance(question.criteria, list):
                add("error", "score.criteria", name, "score criteria must be an ordered list of levels")
                continue
            if rules["require_ordered_levels"] and len(question.criteria) < int(rules["min_score_levels"]):
                add(
                    "error",
                    "score.levels",
                    name,
                    f"a graded scale needs at least {rules['min_score_levels']} levels",
                )
            for level in question.criteria:
                if len(_words(str(level))) < int(rules["min_criteria_words"]):
                    add("error", "score.level_text", name, f"level '{str(level)[:40]}' is not a concrete situation")

        elif question.kind == "noul":
            first = (_words(instruction) or [""])[0].lower()
            if first not in NOUL_OPENERS:
                add(
                    "error",
                    "noul.question_form",
                    name,
                    f"a noul must be a yes/no question (found '{first}', expected one of {list(NOUL_OPENERS[:6])}...)",
                )
        else:
            add("error", "question.type", name, f"unknown primitive '{question.kind}'")

    return report


def lint_library() -> dict[str, LintReport]:
    return {name: lint_question_set(qs) for name, qs in load_library().items()}
