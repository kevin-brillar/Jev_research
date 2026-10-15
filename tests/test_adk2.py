def test_adk2_workflow_is_one_node_graph() -> None:
    from backend.adk2 import root_agent

    assert root_agent.name == "atenxion_jev_adk2"
    assert len(root_agent.edges) == 1

