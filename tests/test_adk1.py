def test_adk1_agent_is_small_and_has_one_jev_tool() -> None:
    from backend.adk1 import root_agent

    assert root_agent.name == "atenxion_jev_adk1"
    assert len(root_agent.tools) == 1

