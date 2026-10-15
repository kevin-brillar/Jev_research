"""ADK 1.x: one chat agent with Jev as a function tool."""

from __future__ import annotations

import os

from google.adk.agents import LlmAgent

from backend.jev import route_with_jev


root_agent = LlmAgent(
    name="atenxion_jev_adk1",
    model=os.getenv("ATENXION_WRITER_MODEL", "gemini-2.5-flash"),
    description="Atenxion chat demo using Jev for routing and safety decisions.",
    tools=[route_with_jev],
    instruction=(
        "You are an Atenxion research assistant. For every user request, call "
        "route_with_jev exactly once before answering. Briefly explain the selected "
        "route, confidence, risk, and whether human review is required. Never claim "
        "that an external action was completed."
    ),
)

