"""
Day 5, Part A: Persistent Memory via Checkpointing
----------------------------------------------------
Improvement over chatbot_day4.py:
  Instead of manually passing the growing `state` dict around in a
  Python loop (which forgets everything the moment the script exits),
  LangGraph's checkpointer saves state automatically, keyed by a
  `thread_id`. This means:
    - Each user/session can have its own independent conversation
    - You could restart the script and resume a conversation by
      reusing the same thread_id (with a persistent checkpointer
      like SqliteSaver, instead of the in-memory one used here)

This file reuses the exact same graph as chatbot_day4.py, only the
compile step and invocation pattern change.
"""

import os
from typing import TypedDict, Annotated, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3,
    api_key=os.getenv("GROQ_API_KEY"),
)


class State(TypedDict):
    messages: Annotated[list, add_messages]
    intent: str


class IntentClassification(BaseModel):
    intent: Literal["support", "sales", "general"] = Field(description="The user's underlying intent.")
    confidence: Literal["high", "low"] = Field(description="Confidence in this classification.")


classifier_llm = llm.with_structured_output(IntentClassification)


def detect_intent(state: State) -> dict:
    last_user_msg = state["messages"][-1].content
    try:
        result = classifier_llm.invoke([
            SystemMessage(content=(
                "Classify into: support, sales, or general. "
                "Set confidence to 'low' if ambiguous."
            )),
            HumanMessage(content=last_user_msg),
        ])
        intent = result.intent if result.confidence == "high" else "clarify"
    except Exception as e:
        print(f"[warning] intent classification failed: {e}")
        intent = "general"
    return {"intent": intent}


def route_by_intent(state: State) -> Literal["support_node", "sales_node", "standard_node", "clarify_node"]:
    return {
        "support": "support_node",
        "sales": "sales_node",
        "clarify": "clarify_node",
    }.get(state["intent"], "standard_node")


def _safe_llm_call(system_prompt: str, messages: list) -> AIMessage:
    try:
        return llm.invoke([SystemMessage(content=system_prompt)] + messages)
    except Exception as e:
        return AIMessage(content=f"Sorry, I'm having trouble reaching the model right now ({e}).")


def standard_node(state: State) -> dict:
    return {"messages": [_safe_llm_call("You are a friendly, general-purpose assistant.", state["messages"])]}


def support_node(state: State) -> dict:
    return {"messages": [_safe_llm_call(
        "You are a specialized technical support agent. Be empathetic and give step-by-step troubleshooting.",
        state["messages"])]}


def sales_node(state: State) -> dict:
    return {"messages": [_safe_llm_call(
        "You are a helpful sales assistant. Answer pricing/product questions honestly.",
        state["messages"])]}


def clarify_node(state: State) -> dict:
    return {"messages": [_safe_llm_call(
        "The user's intent is unclear. Ask ONE short clarifying question.",
        state["messages"])]}


graph = StateGraph(State)
graph.add_node("detect_intent", detect_intent)
graph.add_node("standard_node", standard_node)
graph.add_node("support_node", support_node)
graph.add_node("sales_node", sales_node)
graph.add_node("clarify_node", clarify_node)
graph.set_entry_point("detect_intent")
graph.add_conditional_edges("detect_intent", route_by_intent, {
    "support_node": "support_node",
    "sales_node": "sales_node",
    "standard_node": "standard_node",
    "clarify_node": "clarify_node",
})
for node in ["standard_node", "support_node", "sales_node", "clarify_node"]:
    graph.add_edge(node, END)

# --- The key Day 5 change: compile WITH a checkpointer -----------------
memory = MemorySaver()
app = graph.compile(checkpointer=memory)


def main():
    print("Day 5 Chatbot (persistent memory via checkpointer) — type 'quit' to exit\n")
    thread_id = input("Enter a session/thread id (e.g. 'user1'): ").strip() or "default"
    config = {"configurable": {"thread_id": thread_id}}

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            print("Bye!")
            break
        if not user_input:
            continue

        # Notice: we only pass the NEW message, not the whole history —
        # the checkpointer remembers and merges prior state for us.
        result = app.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)

        last_reply = result["messages"][-1]
        print(f"Bot [{result['intent']}]: {last_reply.content}\n")


if __name__ == "__main__":
    main()