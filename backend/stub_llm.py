"""Offline model stand-in, selected by the spec (`model: stub`).

A test double, never a prompt: answers come from config/stub_responses.yaml keyed
by node id, and the node id travels on the model name (`stub:<node_id>`) so the
double cannot be confused with a real model. This is what lets both runtimes be
exercised end to end with no network and no credentials.
"""

from __future__ import annotations

from typing import AsyncGenerator

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from pydantic import Field

from backend import config


class StubLlm(BaseLlm):
    """Returns the canned answer for the node that owns it."""

    responses: dict = Field(default_factory=dict)

    @property
    def node_id(self) -> str:
        return self.model.split(":", 1)[1] if ":" in self.model else ""

    def _answer(self) -> str:
        document = self.responses or config.stub_responses()
        nodes = document.get("nodes") or {}
        entry = nodes.get(self.node_id) or {}
        return str(entry.get("text") or (document.get("default") or {}).get("text", "Understood."))

    async def generate_content_async(
        self, llm_request, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        from google.genai import types

        yield LlmResponse(
            content=types.Content(role="model", parts=[types.Part(text=self._answer())])
        )
