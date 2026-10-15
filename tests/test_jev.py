from __future__ import annotations

import pytest

from backend.jev import _apply_atenxion_policy, route_with_jev


@pytest.mark.asyncio
async def test_mock_routes_policy_question_to_rag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JEV_MODE", "mock")

    result = await route_with_jev("What is our annual leave policy?")

    assert result["recommended_route"] == "rag"
    assert result["execution_mode"] == "automatic"


@pytest.mark.asyncio
async def test_mock_escalates_external_action(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JEV_MODE", "mock")

    result = await route_with_jev("Refund the customer's last payment")

    assert result["recommended_route"] == "human_review"
    assert result["execution_mode"] == "human_review"


def test_incomplete_answer_fails_closed() -> None:
    result = _apply_atenxion_policy({}, "jev-test")

    assert result["recommended_route"] == "human_review"

