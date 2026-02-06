from typing import Dict, Any
from langgraph.graph import StateGraph, END

from app.agents.multi.state import MultiAgentState
from app.agents.multi.planner import node_plan
from app.agents.multi.worker import node_work
from app.agents.multi.critic import node_critic


MAX_ATTEMPTS = 2  # initial + 1 retry


def _route_after_critic(state: Dict[str, Any]) -> str:
    if state.get("done") is True:
        return "end"
    # stop after max attempts
    if int(state.get("attempts", 0)) >= MAX_ATTEMPTS:
        return "end"
    return "work"


def build_multi_agent_graph():
    g = StateGraph(MultiAgentState)

    g.add_node("plan", node_plan)
    g.add_node("work", node_work)
    g.add_node("critic", node_critic)

    g.set_entry_point("plan")
    g.add_edge("plan", "work")
    g.add_edge("work", "critic")

    g.add_conditional_edges(
        "critic",
        _route_after_critic,
        {"work": "work", "end": END},
    )

    return g.compile()