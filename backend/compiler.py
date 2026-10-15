"""Compiler: one spec, N runtimes.

The compiler owns everything that is not runtime-specific:

* node-type dispatch through a registry, so a new node type is one entry here plus
  one method on the runtime adapter (never an if-chain in a builder)
* the shared semantics of the node types that must behave identically everywhere:
  a `jev` node consults Jev once, writes the decision into state for prompts, and
  returns the route; a `router` node evaluates state conditions; state updates are
  applied the same way
* graph assembly from the spec's structural edges plus each node's route map, with
  the default route always declared (an unmatched route must not silently end a run)

Runtimes only translate node objects into their own ADK types.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, MutableMapping, Protocol

from backend import config, spec as spec_module, tools
from backend.instructions import assemble_instruction
from backend.jev import Decision, DecisionLedger, decide, scope_state
from backend.policy import PolicyEngine
from backend.questions import load_library

#: node type -> adapter method name. Adding a node type means one row here.
BUILDERS: dict[str, str] = {
    "llm": "build_llm",
    "function": "build_function",
    "jev": "build_jev",
    "router": "build_router",
    "human": "build_human",
    "termination": "build_termination",
}


@dataclass
class BuildContext:
    """Everything a runtime needs to translate one spec, all of it from config."""

    spec: spec_module.WorkflowSpec
    runtime: str
    policy: PolicyEngine
    questions: dict[str, Any]
    ledger: DecisionLedger | None = None
    model_override: str | None = None

    def model_for(self, node: spec_module.NodeSpec) -> str:
        if node.model:
            return node.model
        if self.model_override:
            return self.model_override
        if self.spec.default_model:
            return self.spec.default_model
        raise ValueError(f"node '{node.id}' has no model and the spec declares no default_model")

    def instruction_for(self, node: spec_module.NodeSpec) -> str:
        return assemble_instruction(
            node_id=node.id,
            instruction=node.instruction,
            workflow_id=self.spec.id,
            model=self.model_for(node),
            guidance=node.guidance,
            tools_offered=node.tools,
        )


@dataclass
class Graph:
    """Runtime-neutral view of a compiled workflow."""

    spec: spec_module.WorkflowSpec
    nodes: dict[str, Any] = field(default_factory=dict)
    route_map: dict[str, dict[str, str]] = field(default_factory=dict)
    default_routes: dict[str, str] = field(default_factory=dict)
    structural_edges: list[tuple[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class CompileResult:
    root: Any
    spec: spec_module.WorkflowSpec
    context: BuildContext
    graph: Graph

    @property
    def warnings(self) -> list[str]:
        return self.graph.warnings


class RuntimeAdapter(Protocol):
    name: str

    def build_llm(self, ctx: BuildContext, node: spec_module.NodeSpec) -> Any: ...
    def build_function(self, ctx: BuildContext, node: spec_module.NodeSpec) -> Any: ...
    def build_jev(self, ctx: BuildContext, node: spec_module.NodeSpec) -> Any: ...
    def build_router(self, ctx: BuildContext, node: spec_module.NodeSpec) -> Any: ...
    def build_human(self, ctx: BuildContext, node: spec_module.NodeSpec) -> Any: ...
    def build_termination(self, ctx: BuildContext, node: spec_module.NodeSpec) -> Any: ...
    def finalize(self, ctx: BuildContext, graph: Graph) -> Any: ...


# --------------------------------------------------------------- shared node logic


def apply_state(target: MutableMapping[str, Any], updates: dict[str, Any]) -> None:
    """Write dotted keys into session state verbatim.

    State is keyed flat ("case.pending_action", "jev_route") on purpose: ADK's own
    `{key}` instruction templating then works, scoping is a key lookup, and the
    ledger can record which keys a decision saw.
    """
    for key, value in updates.items():
        target[key] = value


LITERALS: dict[str, Any] = {"true": True, "false": False, "null": None, "none": None}

_CONDITION_RE = re.compile(
    r"^\s*(?P<left>[A-Za-z_][\w.]*)\s*(?P<op>==|!=|>=|<=|>|<|not in|in)\s*(?P<right>.+?)\s*$"
)


def _resolve(state: MutableMapping[str, Any], path: str) -> Any:
    if path in state:
        return state[path]
    value: Any = state
    for part in path.split("."):
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            return None
    return value


def _literal(text: str) -> Any:
    stripped = text.strip()
    if stripped.lower() in LITERALS:
        return LITERALS[stripped.lower()]
    if stripped[:1] in ("'", '"') and stripped[-1:] == stripped[:1]:
        return stripped[1:-1]
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return stripped


def evaluate_condition(condition: str, state: MutableMapping[str, Any]) -> bool | None:
    """Evaluate `path op literal` without `eval`.

    The Atenxion backend runs conditions through a character-allowlisted `eval`
    (see its IfElseAgent). Here the grammar is small and parsed, because a learned
    decision should not be re-derived by evaluating templated state as code.
    """
    match = _CONDITION_RE.match(condition)
    if not match:
        return None
    left = _resolve(state, match.group("left"))
    right = _literal(match.group("right"))
    op = match.group("op")
    if op == "==":
        return left == right
    if op == "!=":
        return left != right
    if op in (">", ">=", "<", "<="):
        try:
            left_num, right_num = float(left), float(right)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        return {
            ">": left_num > right_num,
            ">=": left_num >= right_num,
            "<": left_num < right_num,
            "<=": left_num <= right_num,
        }[op]
    if op == "in":
        return isinstance(right, (list, tuple, set, str)) and left in right
    if op == "not in":
        return isinstance(right, (list, tuple, str)) and left not in right
    return None


@dataclass
class NodeOutcome:
    """What a deterministic or decision node produced, runtime-neutral."""

    route: str | None = None
    output: Any = None
    state: dict[str, Any] = field(default_factory=dict)
    log: list[str] = field(default_factory=list)
    review_required: bool = False


async def run_jev_node(node: spec_module.NodeSpec, state: MutableMapping[str, Any], ctx: BuildContext) -> NodeOutcome:
    """Consult Jev once for a `jev` node and translate it into a route.

    Shared by both runtimes so the node's meaning cannot drift between them. The
    decision is written into state as flat `jev_*` keys, which is how a downstream
    model sees it: guidance blocks in config/guidance.yaml reference
    `{jev_route}`, `{jev_execution_mode}` and the numbers, with no code in the path.
    """
    scoped = scope_state(dict(state), node.state_scope)
    decision: Decision = await decide(node.question_set or "", scoped, policy=ctx.policy, ledger=ctx.ledger)

    declared = node.routes or {}
    route = decision.route if decision.route in declared else (node.default_route or decision.route)
    log = [
        f"{node.id}: jev[{decision.question_set}] route={decision.route} "
        f"mode={decision.execution_mode} conf={decision.confidence} risk={decision.risk_score} "
        f"human={decision.human_probability} -> {route}"
    ]
    if decision.route not in declared:
        log.append(
            f"{node.id}: route '{decision.route}' has no edge; falling back to default_route '{node.default_route}'"
        )
    if decision.reasons:
        log.append(f"{node.id}: {'; '.join(decision.reasons)}")

    history = list(_resolve(state, "jev_history") or [])
    history.append(decision.as_dict())
    updates: dict[str, Any] = {
        **decision.prompt_vars(),
        f"jev_{node.id}": decision.as_dict(),
        "jev_history": history,
    }
    return NodeOutcome(route=route, output=decision.as_dict(), state=updates, log=log, review_required=not decision.automatic)


async def run_router_node(node: spec_module.NodeSpec, state: MutableMapping[str, Any]) -> NodeOutcome:
    """First matching condition wins; otherwise the declared default route."""
    for entry in node.conditions:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            continue
        condition, target = str(entry[0]), str(entry[1])
        verdict = evaluate_condition(condition, state)
        if verdict:
            return NodeOutcome(route=target, output=verdict, log=[f"{node.id}: '{condition}' -> {target}"])
        if verdict is None:
            return NodeOutcome(
                route=node.default_route,
                output=None,
                log=[f"{node.id}: condition '{condition}' could not be evaluated; using default_route"],
            )
    return NodeOutcome(route=node.default_route, output=None, log=[f"{node.id}: no condition matched; default_route"])


# ------------------------------------------------------------------- compilation


def compile_workflow(
    name: str,
    adapter: RuntimeAdapter,
    *,
    ledger: DecisionLedger | None = None,
    model_override: str | None = None,
    spec: spec_module.WorkflowSpec | None = None,
) -> CompileResult:
    """Validate a spec and translate it into a runnable agent for one runtime.

    Args:
        name: workflow name under config/workflows (ignored when `spec` is given).
        adapter: the runtime that knows ADK 1.x or ADK 2.x node types.
        ledger: optional decision ledger; every Jev node then records its verdict.
        model_override: replace the spec's model everywhere (used to run offline).
        spec: pre-loaded spec (tests, or a spec built in memory).
    """
    resolved = spec if spec is not None else spec_module.load_workflow(name)
    problems = resolved.validate()
    if problems:
        raise ValueError("spec is not buildable:\n  - " + "\n  - ".join(problems))

    context = BuildContext(
        spec=resolved,
        runtime=adapter.name,
        policy=PolicyEngine(),
        questions=load_library(),
        ledger=ledger,
        model_override=model_override,
    )

    graph = Graph(spec=resolved)
    for node_id, node in resolved.nodes.items():
        builder = getattr(adapter, BUILDERS[node.type])
        graph.nodes[node_id] = builder(context, node)

    # Route-bookkeeping: every routed node keeps its map and its default, so the
    # runtime can emit a route-tagged edge plus a fallback edge.
    graph.route_map = resolved.route_targets()
    graph.default_routes = {node_id: target for node_id, target in resolved.default_routes().items() if target}
    graph.structural_edges = [
        (edge.source, edge.target) for edge in resolved.declared_edges
    ]

    for node_id, node in resolved.nodes.items():
        if node.type in spec_module.ROUTED_TYPES and node.default_route not in node.routes.values():
            graph.warnings.append(
                f"{node_id}: default_route '{node.default_route}' is not one of the route targets "
                f"({sorted(set(node.routes.values()))}); it is used only for unmatched routes"
            )

    root = adapter.finalize(context, graph)
    return CompileResult(root=root, spec=resolved, context=context, graph=graph)


def validate_spec_file(name: str) -> list[str]:
    """Validate a spec without building it (used by tests and CLI tooling)."""
    path = config.workflow_path(name)
    parsed = spec_module.parse_workflow(config.read_yaml(path), source_path=str(path))
    return parsed.validate()


def describe(name: str) -> dict[str, Any]:
    """Human/agent-readable summary of a spec: nodes, prompt sizes, guidance, tools."""
    loaded = spec_module.load_workflow(name)
    context = BuildContext(
        spec=loaded,
        runtime="describe",
        policy=PolicyEngine(),
        questions=load_library(),
    )
    return {
        "id": loaded.id,
        "name": loaded.name,
        "model": loaded.default_model,
        "nodes": [
            {
                "id": node.id,
                "type": node.type,
                "guidance": node.guidance,
                "tools": node.tools,
                "question_set": node.question_set,
                "state_scope": node.state_scope,
                "routes": node.routes,
                "default_route": node.default_route,
                "instruction_chars": len(context.instruction_for(node)) if node.type == "llm" else 0,
            }
            for node in loaded.nodes.values()
        ],
        "edges": [
            {"from": edge.source, "to": edge.target, "route": edge.route} for edge in loaded.edges()
        ],
        "config_sources": {
            "policy_sets": context.policy.sets(),
            "question_sets": sorted(context.questions),
            "guidance_blocks": sorted(config.guidance().get("blocks", {})),
            "tools": sorted(tools.TOOLS),
        },
    }
