import logging
from langgraph.graph import StateGraph, END

from app.agents.state import AgentState
from app.agents.nodes import (
    node_retrieve,
    node_assess,
    node_rewrite,
    node_retrieve_rewritten,
    node_answer,
)

logger = logging.getLogger(__name__)

def build_graph():
    g = StateGraph(AgentState)

    g.add_node("retrieve", node_retrieve)
    g.add_node("assess", node_assess)
    g.add_node("rewrite", node_rewrite)
    g.add_node("retrieve2", node_retrieve_rewritten)
    g.add_node("answer", node_answer)

    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "assess")

    def route(state):
        ok = state.get("_retrieval_ok", False)
        return "answer" if ok else "rewrite"

    g.add_conditional_edges("assess", route, {"answer": "answer", "rewrite": "rewrite"})
    g.add_edge("rewrite", "retrieve2")
    g.add_edge("retrieve2", "answer")
    g.add_edge("answer", END)

    return g.compile()