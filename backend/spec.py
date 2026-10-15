"""Workflow specs: the artifact an agent (or a person) authors.

The format follows the canvas specs the Atenxion dashboard compiles into ADK
(`version` / `id` / `nodes` / `edges`, node `type`s, router conditions), extended
with the node types this experiment needs. One spec is compiled to both runtimes,
so a scenario is written once and tested on ADK 1.31 and ADK 2.8.

Validation is deterministic and runs before anything is built: unknown node types,
unresolvable function names, dangling route targets, tools or guidance blocks that
do not exist, unreachable nodes, and route keys that no edge consumes. Every check
reports a message a person or an agent can act on, because a spec is an input to a
build, not a program.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from backend import config, tools
from backend.functions import NODE_FUNCTIONS
from backend.instructions import guidance_blocks

NODE_TYPES: tuple[str, ...] = ("function", "llm", "jev", "router", "human", "termination")

#: Node types that must name the route targets they can jump to.
ROUTED_TYPES: tuple[str, ...] = ("jev", "router", "human")


@dataclass
class NodeSpec:
    id: str
    type: str
    name: str = ""
    description: str = ""
    instruction: str = ""
    guidance: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    model: str | None = None
    function: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    question_set: str | None = None
    policy: str | None = None
    state_scope: list[str] | None = None
    routes: dict[str, str] = field(default_factory=dict)
    default_route: str | None = None
    conditions: list[list[Any]] = field(default_factory=list)
    message: str = ""
    timeout_seconds: int | None = None
    output_key: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return self.name or self.id


@dataclass(frozen=True)
class EdgeSpec:
    source: str
    target: str
    route: str | None = None


@dataclass
class WorkflowSpec:
    id: str
    name: str = ""
    description: str = ""
    version: str = "1"
    default_model: str = ""
    state: dict[str, Any] = field(default_factory=dict)
    nodes: dict[str, NodeSpec] = field(default_factory=dict)
    declared_edges: list[EdgeSpec] = field(default_factory=list)
    source_path: str = ""

    # ------------------------------------------------------------------ graph
    @property
    def entry(self) -> str:
        """First node: a `start`-less spec enters at the first declared node."""
        return next(iter(self.nodes))

    def route_edges(self) -> list[EdgeSpec]:
        return [
            EdgeSpec(node_id, target, route)
            for node_id, node in self.nodes.items()
            for route, target in node.routes.items()
        ]

    def edges(self) -> list[EdgeSpec]:
        return [*self.declared_edges, *self.route_edges()]

    def successors(self, node_id: str) -> list[EdgeSpec]:
        return [edge for edge in self.edges() if edge.source == node_id]

    def default_routes(self) -> dict[str, str]:
        return {node_id: node.default_route for node_id, node in self.nodes.items() if node.default_route}

    def route_targets(self) -> dict[str, dict[str, str]]:
        return {node_id: dict(node.routes) for node_id, node in self.nodes.items() if node.routes}

    def reachable(self) -> set[str]:
        seen: set[str] = set()
        stack = [self.entry]
        while stack:
            current = stack.pop()
            if current in seen or current not in self.nodes:
                continue
            seen.add(current)
            stack.extend(edge.target for edge in self.successors(current))
        return seen

    # ------------------------------------------------------------- validation
    def validate(self) -> list[str]:
        """Return human-readable problems; an empty list means the spec is buildable."""
        problems: list[str] = []

        if not self.id:
            problems.append("spec has no id")
        if not self.nodes:
            problems.append("spec has no nodes")
            return problems

        for node_id, node in self.nodes.items():
            if not node_id.isidentifier():
                problems.append(f"node id '{node_id}' is not a valid Python identifier (ADK requires it)")

            if node.type not in NODE_TYPES:
                problems.append(f"node '{node_id}': unknown type '{node.type}' (known: {list(NODE_TYPES)})")
                continue

            if node.type == "llm":
                if not node.instruction.strip():
                    problems.append(f"node '{node_id}': llm node has no instruction")
                problems.extend(self._check_guidance(node))
                problems.extend(self._check_tools(node))

            elif node.type == "function":
                if not node.function:
                    problems.append(f"node '{node_id}': function node has no function name")
                elif node.function not in NODE_FUNCTIONS:
                    problems.append(
                        f"node '{node_id}': unknown function '{node.function}' (known: {sorted(NODE_FUNCTIONS)})"
                    )

            elif node.type == "jev":
                if not node.question_set:
                    problems.append(f"node '{node_id}': jev node has no question_set")
                if not node.state_scope:
                    problems.append(
                        f"node '{node_id}': jev node declares no state_scope, so the judgment would see all of state"
                    )

            elif node.type == "human":
                if not node.message.strip():
                    problems.append(f"node '{node_id}': human node has no message")

            if node.type in ROUTED_TYPES:
                if not node.routes:
                    problems.append(f"node '{node_id}': {node.type} node has no routes")
                if not node.default_route:
                    problems.append(
                        f"node '{node_id}': {node.type} node has no default_route; an unmatched route would end the branch"
                    )
                for route, target in node.routes.items():
                    if target not in self.nodes:
                        problems.append(f"node '{node_id}': route '{route}' targets unknown node '{target}'")
                if node.default_route and node.default_route not in self.nodes:
                    problems.append(
                        f"node '{node_id}': default_route targets unknown node '{node.default_route}'"
                    )

        for edge in self.declared_edges:
            if edge.source not in self.nodes:
                problems.append(f"edge from unknown node '{edge.source}'")
            if edge.target not in self.nodes:
                problems.append(f"edge to unknown node '{edge.target}'")

        unreachable = set(self.nodes) - self.reachable()
        if unreachable:
            problems.append(f"unreachable nodes: {sorted(unreachable)}")

        for node_id, node in self.nodes.items():
            if node.type != "termination" and not self.successors(node_id):
                problems.append(f"node '{node_id}' has no outgoing edge and is not a termination node")

        if not any(node.type == "termination" for node in self.nodes.values()):
            problems.append("spec has no termination node")

        targets = {edge.target for edge in self.edges()}
        for node_id, node in self.nodes.items():
            if node.type == "termination":
                continue
            if node_id not in targets and node_id != self.entry:
                problems.append(f"node '{node_id}' is never used as a target and is not the entry node")

        return problems

    @staticmethod
    def _check_guidance(node: NodeSpec) -> list[str]:
        available = set(guidance_blocks())
        return [
            f"node '{node.id}': unknown guidance block '{name}' (known: {sorted(available)})"
            for name in node.guidance
            if name not in available
        ]

    @staticmethod
    def _check_tools(node: NodeSpec) -> list[str]:
        return [
            f"node '{node.id}': unknown tool '{name}' (known: {sorted(tools.TOOLS)})"
            for name in node.tools
            if name not in tools.TOOLS
        ]


def parse_node(raw: dict[str, Any]) -> NodeSpec:
    return NodeSpec(
        id=str(raw.get("id") or ""),
        type=str(raw.get("type") or ""),
        name=str(raw.get("name") or ""),
        description=str(raw.get("description") or ""),
        instruction=str(raw.get("instruction") or ""),
        guidance=list(raw.get("guidance") or []),
        tools=list(raw.get("tools") or []),
        model=raw.get("model"),
        function=raw.get("function"),
        params=dict(raw.get("params") or {}),
        question_set=raw.get("question_set"),
        policy=raw.get("policy"),
        state_scope=list(raw["state_scope"]) if raw.get("state_scope") is not None else None,
        routes={str(k): str(v) for k, v in (raw.get("routes") or {}).items()},
        default_route=raw.get("default_route"),
        conditions=list(raw.get("conditions") or []),
        message=str(raw.get("message") or ""),
        timeout_seconds=raw.get("timeout_seconds"),
        output_key=raw.get("output_key"),
        raw=raw,
    )


def parse_workflow(doc: dict[str, Any], source_path: str = "") -> WorkflowSpec:
    nodes: dict[str, NodeSpec] = {}
    for raw in doc.get("nodes") or []:
        node = parse_node(raw)
        if node.id in nodes:
            raise ValueError(f"{source_path}: duplicate node id '{node.id}'")
        nodes[node.id] = node
    return WorkflowSpec(
        id=str(doc.get("id") or ""),
        name=str(doc.get("name") or ""),
        description=str(doc.get("description") or ""),
        version=str(doc.get("version") or "1"),
        default_model=str(doc.get("default_model") or ""),
        state=dict(doc.get("state") or {}),
        nodes=nodes,
        declared_edges=[
            EdgeSpec(str(edge.get("from")), str(edge.get("to")), edge.get("route"))
            for edge in (doc.get("edges") or [])
        ],
        source_path=source_path,
    )


def load_workflow(name: str) -> WorkflowSpec:
    """Load and validate a spec by workflow name (config/workflows/<name>.yaml)."""
    path = config.workflow_path(name)
    spec = parse_workflow(config.read_yaml(path), source_path=str(path))
    problems = spec.validate()
    if problems:
        joined = "\n  - ".join(problems)
        raise ValueError(f"workflow '{name}' is not buildable:\n  - {joined}")
    return spec
