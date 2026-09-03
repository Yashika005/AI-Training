"""
Week 1 Project: Intent-Routing Chatbot with LangGraph
------------------------------------------------------
Covers:
  - StateGraph (the graph object)
  - State (centralized shared data object)
  - Nodes (functions that read/update state)
  - Edges (fixed routing) and Conditional Edges (dynamic routing)

Flow:
    START -> detect_intent -> (support_node OR standard_node) -> END

Run:
    python chatbot.py
"""

import os
from typing import TypedDict, Annotated, Literal

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

load_dotenv()

# ---------------------------------------------------------------------------
# 1. LLM setup
# ---------------------------------------------------------------------------
# Any LangChain chat model works here (ChatOpenAI, ChatGoogleGenerativeAI,
# ChatAnthropic, etc). Groq is used by default because it's free with no
# credit card required. Get a key at https://console.groq.com
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3,
    api_key=os.getenv("GROQ_API_KEY"),
)

# ---------------------------------------------------------------------------
# 2. State definition
# ---------------------------------------------------------------------------
# This is the "single source of truth" that flows through every node.
# `add_messages` is a reducer: instead of overwriting the messages list,
# LangGraph appends new messages to it automatically.
class State(TypedDict):
    messages: Annotated[list, add_messages]
    intent: str


# ---------------------------------------------------------------------------
# 3. Nodes
# ---------------------------------------------------------------------------
def detect_intent(state: State) -> dict:
    """Reads the latest user message and classifies intent.
    Returns only the fields it wants to update (LangGraph merges the rest)."""
    last_user_msg = state["messages"][-1].content.lower()

    support_keywords = [
        "help", "support", "issue", "problem", "error",
        "bug", "not working", "broken", "refund", "cancel", "trouble",
    ]
    intent = "support" if any(k in last_user_msg for k in support_keywords) else "general"
    return {"intent": intent}


def route_by_intent(state: State) -> Literal["support_node", "standard_node"]:
    """This is the conditional edge function. It doesn't update state,
    it only returns the NAME of the next node to visit."""
    return "support_node" if state["intent"] == "support" else "standard_node"


def standard_node(state: State) -> dict:
    system = SystemMessage(
        content="You are a friendly, general-purpose assistant. Keep answers concise and helpful."
    )
    response = llm.invoke([system] + state["messages"])
    return {"messages": [response]}


def support_node(state: State) -> dict:
    system = SystemMessage(
        content=(
            "You are a specialized technical support agent. Be empathetic, "
            "ask a clarifying question if the issue is vague, and give clear "
            "step-by-step troubleshooting."
        )
    )
    response = llm.invoke([system] + state["messages"])
    return {"messages": [response]}


# ---------------------------------------------------------------------------
# 4. Build the graph
# ---------------------------------------------------------------------------
graph = StateGraph(State)

graph.add_node("detect_intent", detect_intent)
graph.add_node("standard_node", standard_node)
graph.add_node("support_node", support_node)

graph.set_entry_point("detect_intent")

# Conditional edge: after detect_intent, branch based on route_by_intent()
graph.add_conditional_edges(
    "detect_intent",
    route_by_intent,
    {
        "support_node": "support_node",
        "standard_node": "standard_node",
    },
)

# Fixed edges: both branches end the graph run
graph.add_edge("standard_node", END)
graph.add_edge("support_node", END)

app = graph.compile()


# ---------------------------------------------------------------------------
# 5. Simple CLI loop (this is what gives the bot "memory" across turns —
#    we keep re-feeding the accumulated state back into the graph)
# ---------------------------------------------------------------------------
def main():
    print("LangGraph Intent-Routing Chatbot — type 'quit' to exit\n")
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