"""
Day 4: Smarter Intent Detection + Robustness
---------------------------------------------
Improvements over chatbot.py:
  1. Intent detection uses the LLM itself (structured output) instead of
     keyword matching -> catches phrasing keyword matching would miss.
  2. A third branch: "sales" (pricing/product questions), plus a
     "clarify" branch for when the model isn't confident about intent.
  3. try/except around every LLM call so a network/API failure doesn't
     crash the whole app -> the bot degrades gracefully instead.

Flow:
    START -> detect_intent -> (support_node | sales_node | standard_node | clarify_node) -> END
"""

import os
from typing import TypedDict, Annotated, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3,
    api_key=os.getenv("GROQ_API_KEY"),
)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class State(TypedDict):
    messages: Annotated[list, add_messages]
    intent: str


# ---------------------------------------------------------------------------
# Structured output schema for intent classification
# ---------------------------------------------------------------------------
class IntentClassification(BaseModel):
    intent: Literal["support", "sales", "general"] = Field(
        description="The user's underlying intent."
    )
    confidence: Literal["high", "low"] = Field(
        description="How confident you are in this classification."
    )


classifier_llm = llm.with_structured_output(IntentClassification)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def detect_intent(state: State) -> dict:
    last_user_msg = state["messages"][-1].content

    try:
        result = classifier_llm.invoke([
            SystemMessage(content=(
                "Classify the user's message into one of: support "
                "(technical problems, errors, complaints), sales "
                "(pricing, plans, product questions), or general "
                "(everything else, small talk, factual questions). "
                "If the message is too vague or ambiguous to classify "
                "confidently, set confidence to 'low'."
            )),
            HumanMessage(content=last_user_msg),
        ])
        intent = result.intent if result.confidence == "high" else "clarify"
    except Exception as e:
        print(f"[warning] intent classification failed: {e}")
        intent = "general"  # safe fallback so the bot still responds

    return {"intent": intent}


def route_by_intent(state: State) -> Literal["support_node", "sales_node", "standard_node", "clarify_node"]:
    return {
        "support": "support_node",
        "sales": "sales_node",
        "clarify": "clarify_node",
    }.get(state["intent"], "standard_node")


def _safe_llm_call(system_prompt: str, messages: list) -> AIMessage:
    """Wraps the LLM call so one failure doesn't crash the graph."""
    try:
        return llm.invoke([SystemMessage(content=system_prompt)] + messages)
    except Exception as e:
        return AIMessage(content=(
            "Sorry, I'm having trouble reaching the model right now "
            f"({e}). Please try again in a moment."
        ))


def standard_node(state: State) -> dict:
    response = _safe_llm_call(
        "You are a friendly, general-purpose assistant. Keep answers concise.",
        state["messages"],
    )
    return {"messages": [response]}


def support_node(state: State) -> dict:
    response = _safe_llm_call(
        "You are a specialized technical support agent. Be empathetic, "
        "ask a clarifying question if needed, and give step-by-step "
        "troubleshooting.",
        state["messages"],
    )
    return {"messages": [response]}


def sales_node(state: State) -> dict:
    response = _safe_llm_call(
        "You are a helpful sales assistant. Answer pricing and product "
        "questions clearly and honestly, without being pushy.",
        state["messages"],
    )
    return {"messages": [response]}


def clarify_node(state: State) -> dict:
    response = _safe_llm_call(
        "The user's intent is unclear. Ask ONE short, friendly "
        "clarifying question to understand what they need "
        "(support, pricing/sales, or general info).",
        state["messages"],
    )
    return {"messages": [response]}


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------
graph = StateGraph(State)

graph.add_node("detect_intent", detect_intent)
graph.add_node("standard_node", standard_node)
graph.add_node("support_node", support_node)
graph.add_node("sales_node", sales_node)
graph.add_node("clarify_node", clarify_node)

graph.set_entry_point("detect_intent")

graph.add_conditional_edges(
    "detect_intent",
    route_by_intent,
    {
        "support_node": "support_node",
        "sales_node": "sales_node",
        "standard_node": "standard_node",
        "clarify_node": "clarify_node",
    },
)

for node in ["standard_node", "support_node", "sales_node", "clarify_node"]:
    graph.add_edge(node, END)

app = graph.compile()


def main():
    print("Day 4 Chatbot (LLM intent classification) — type 'quit' to exit\n")
    state: State = {"messages": [], "intent": ""}

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            print("Bye!")
            break
        if not user_input:
            continue

        state["messages"].append(HumanMessage(content=user_input))
        state = app.invoke(state)

        last_reply = state["messages"][-1]
        print(f"Bot [{state['intent']}]: {last_reply.content}\n")


if __name__ == "__main__":
    main()