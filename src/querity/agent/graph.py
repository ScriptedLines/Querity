from __future__ import annotations

from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from querity.agent.nodes import AgentNodes
from querity.agent.settings import AgentSettings
from querity.agent.state import AgentState


def _status_failed(state: AgentState) -> bool:
    return state.get("status") == "failed"


def route_after_analyze(state: AgentState, settings: AgentSettings) -> str:
    if _status_failed(state):
        return "finalize"
    plan = state.get("plan")
    if plan is None:
        return "retrieve"
    can_clarify = (
        settings.enable_clarification
        and int(state.get("clarification_count", 0)) < 1
        and not state.get("clarification")
    )
    if plan.needs_clarification and can_clarify:
        return "clarify"
    if not plan.answerable and can_clarify:
        return "clarify"
    if not plan.answerable:
        return "finalize"
    return "retrieve"


def route_after_retrieve(state: AgentState) -> str:
    return "finalize" if _status_failed(state) else "generate"


def route_after_generate(state: AgentState) -> str:
    return "finalize" if _status_failed(state) else "guard"


def route_after_guard(state: AgentState, settings: AgentSettings) -> str:
    if _status_failed(state):
        return "finalize"
    guard = state.get("guard")
    if guard is not None and guard.ok:
        return "review"
    if state.get("human_edited"):
        return "human_review"
    if int(state.get("revision_count", 0)) < settings.max_revisions:
        return "revise"
    return "human_review"


def route_after_review(state: AgentState, settings: AgentSettings) -> str:
    if _status_failed(state):
        return "finalize"
    review = state.get("review")
    if review is not None and (review.approved or not review.blocking):
        return "execute" if state.get("pre_approved") else "human_review"
    if int(state.get("revision_count", 0)) < settings.max_revisions:
        return "revise"
    return "human_review"


def route_after_human(state: AgentState) -> str:
    if state.get("status") in {"failed", "rejected"}:
        return "finalize"
    decision = state.get("human_decision")
    if decision is None:
        return "finalize"
    if decision.kind == "edit":
        return "guard"
    if decision.kind == "revise":
        return "generate"
    if decision.kind == "approve":
        guard = state.get("guard")
        return "execute" if guard is not None and guard.ok else "finalize"
    return "finalize"


def route_after_execute(state: AgentState, settings: AgentSettings) -> str:
    execution = state.get("execution")
    if execution is not None and execution.ok:
        return "persist"
    if state.get("human_edited"):
        return "human_review"
    if int(state.get("revision_count", 0)) < settings.max_revisions:
        return "revise"
    return "finalize"


def route_after_persist(state: AgentState) -> str:
    return "finalize"


def build_graph(
    nodes: AgentNodes,
    settings: AgentSettings,
    checkpointer: BaseCheckpointSaver | None = None,
) -> Any:
    graph = StateGraph(AgentState)
    graph.add_node("ingest", nodes.ingest_turn)
    graph.add_node("analyze", nodes.analyze)
    graph.add_node("clarify", nodes.clarify)
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("generate", nodes.generate)
    graph.add_node("guard", nodes.guard)
    graph.add_node("review", nodes.review)
    graph.add_node("revise", nodes.revise)
    graph.add_node("human_review", nodes.human_review)
    graph.add_node("execute", nodes.execute)
    graph.add_node("persist", nodes.persist)
    graph.add_node("finalize", nodes.finalize)

    graph.add_edge(START, "ingest")
    graph.add_edge("ingest", "analyze")
    graph.add_conditional_edges(
        "analyze",
        lambda state: route_after_analyze(state, settings),
        {"clarify": "clarify", "retrieve": "retrieve", "finalize": "finalize"},
    )
    graph.add_conditional_edges(
        "clarify",
        lambda state: "finalize" if _status_failed(state) else "analyze",
        {"analyze": "analyze", "finalize": "finalize"},
    )
    graph.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {"generate": "generate", "finalize": "finalize"},
    )
    graph.add_conditional_edges(
        "generate",
        route_after_generate,
        {"guard": "guard", "finalize": "finalize"},
    )
    graph.add_conditional_edges(
        "guard",
        lambda state: route_after_guard(state, settings),
        {
            "review": "review",
            "revise": "revise",
            "human_review": "human_review",
            "finalize": "finalize",
        },
    )
    graph.add_conditional_edges(
        "review",
        lambda state: route_after_review(state, settings),
        {
            "execute": "execute",
            "human_review": "human_review",
            "revise": "revise",
            "finalize": "finalize",
        },
    )
    graph.add_edge("revise", "generate")
    graph.add_conditional_edges(
        "human_review",
        route_after_human,
        {
            "guard": "guard",
            "generate": "generate",
            "execute": "execute",
            "finalize": "finalize",
        },
    )
    graph.add_conditional_edges(
        "execute",
        lambda state: route_after_execute(state, settings),
        {
            "persist": "persist",
            "revise": "revise",
            "human_review": "human_review",
            "finalize": "finalize",
        },
    )
    graph.add_conditional_edges(
        "persist",
        route_after_persist,
        {"finalize": "finalize"},
    )
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer)
