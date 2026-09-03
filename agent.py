"""
agent.py — Integrated Multi-Turn Agent
=========================================
Builds directly on chatbot_day5_memory.py. Adds, in one cohesive graph:

  1. SqliteSaver          -> persistent memory that survives a restart
  2. Config vs. State     -> thread_id read from `config`, never stored in state
  3. Multiple schemas     -> separate InputState / OutputState / OverallState
  4. Custom reducers      -> append-only interaction_log, overwrite-merge user_preferences
  5. Preference extraction-> a node that writes to user_preferences via its reducer
  6. Trimming             -> only the last N messages are sent to the LLM per call
  7. Summarization        -> older messages get compressed + deleted once history grows
  8. Streaming            -> app.stream(..., stream_mode="updates") shows each node live
  9. Graph visualization  -> see inspect_tools.py
 10. get_state / get_state_history -> see inspect_tools.py
 11. LangSmith            -> picked up automatically from .env, see README

Flow:
    START
      -> detect_intent                 (reads config.thread_id for logging only)
      -> [conditional] support_node | sales_node | standard_node | clarify_node
      -> extract_preferences
      -> [conditional] summarize_node | END
      -> summarize_node -> END
"""

import os
import sqlite3
from typing import TypedDict, Annotated, Literal, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.messages import (
    HumanMessage, SystemMessage, AIMessage, RemoveMessage, BaseMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()

# LangSmith tracing (Day 6): if these are set in your .env, every node and
# LLM call below is automatically traced — no code changes needed here.
#   LANGCHAIN_TRACING_V2=true
#   LANGCHAIN_API_KEY=your_key
#   LANGCHAIN_PROJECT=langgraph-week3-agent

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3,
    api_key=os.getenv("GROQ_API_KEY"),
)

SUMMARY_TRIGGER = 8   # summarize once more than this many messages exist
KEEP_RECENT = 4        # always keep this many most-recent messages verbatim
CONTEXT_WINDOW = 6      # max messages actually sent to the LLM per call


# ---------------------------------------------------------------------------
# Day 1: Custom reducers (append vs. overwrite-merge)
# ---------------------------------------------------------------------------
def append_log(existing: list, new: list) -> list:
    """Reducer: interaction_log always grows, never gets overwritten."""
    return (existing or []) + (new or [])


def merge_preferences(existing: dict, new: dict) -> dict:
    """Reducer: user_preferences updates per-key, doesn't wipe other keys."""
    merged = dict(existing or {})
    merged.update(new or {})
    return merged


# ---------------------------------------------------------------------------
# Day 2: Multiple schemas — input / output / overall (internal) state
# ---------------------------------------------------------------------------
class InputState(TypedDict):
    """What the caller must provide."""
    messages: Annotated[list, add_messages]


class OutputState(TypedDict):
    """What the caller gets back. interaction_log and user_preferences stay
    internal — they're not part of this schema, so they're hidden from the
    final result, even though nodes use and update them along the way."""
    messages: list
    intent: str
    summary: str


class OverallState(TypedDict):
    """The full internal state every node actually sees."""
    messages: Annotated[list, add_messages]
    intent: str
    interaction_log: Annotated[list, append_log]
    user_preferences: Annotated[dict, merge_preferences]
    summary: str


# ---------------------------------------------------------------------------
# Structured output schemas
# ---------------------------------------------------------------------------
class IntentClassification(BaseModel):
    intent: Literal["support", "sales", "general"] = Field(description="The user's underlying intent.")
    confidence: Literal["high", "low"] = Field(description="Confidence in this classification.")


class PreferenceExtraction(BaseModel):
    has_preference: bool = Field(description="True if the user stated a lasting preference/fact about themselves.")
    key: Optional[str] = Field(default=None, description="Short key, e.g. 'name', 'contact_method', 'plan'.")
    value: Optional[str] = Field(default=None, description="The value for that key.")


classifier_llm = llm.with_structured_output(IntentClassification)
preference_llm = llm.with_structured_output(PreferenceExtraction)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def trim_context(messages: list[BaseMessage], keep_last: int = CONTEXT_WINDOW) -> list[BaseMessage]:
    """Day 4: trimming. Only affects what gets SENT to the LLM this call —
    the full history still lives in state/checkpoint until summarize_node
    actually deletes anything."""
    return messages[-keep_last:]


def _safe_llm_call(system_prompt: str, messages: list, summary: str = "") -> AIMessage:
    context = trim_context(messages)
    if summary:
        system_prompt = f"{system_prompt}\n\nConversation summary so far: {summary}"
    try:
        return llm.invoke([SystemMessage(content=system_prompt)] + context)
    except Exception as e:
        return AIMessage(content=f"Sorry, I'm having trouble reaching the model right now ({e}).")


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def detect_intent(state: OverallState, config: RunnableConfig) -> dict:
    # Day 1: config vs. state. thread_id lives in `config`, not `state` —
    # it's metadata about HOW this graph is being run, not data flowing
    # through the graph. We read it here only to tag the log entry; it is
    # never written into user_preferences or messages.
    thread_id = config.get("configurable", {}).get("thread_id", "unknown")

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

    log_entry = {"thread_id": thread_id, "intent": intent, "preview": last_user_msg[:50]}
    return {"intent": intent, "interaction_log": [log_entry]}


def route_by_intent(state: OverallState) -> Literal["support_node", "sales_node", "standard_node", "clarify_node"]:
    return {
        "support": "support_node",
        "sales": "sales_node",
        "clarify": "clarify_node",
    }.get(state["intent"], "standard_node")


def standard_node(state: OverallState) -> dict:
    response = _safe_llm_call("You are a friendly, general-purpose assistant.", state["messages"], state.get("summary", ""))
    return {"messages": [response]}


def support_node(state: OverallState) -> dict:
    response = _safe_llm_call(
        "You are a specialized technical support agent. Be empathetic and give step-by-step troubleshooting.",
        state["messages"], state.get("summary", ""))
    return {"messages": [response]}


def sales_node(state: OverallState) -> dict:
    response = _safe_llm_call(
        "You are a helpful sales assistant. Answer pricing/product questions honestly.",
        state["messages"], state.get("summary", ""))
    return {"messages": [response]}


def clarify_node(state: OverallState) -> dict:
    response = _safe_llm_call(
        "The user's intent is unclear. Ask ONE short clarifying question.",
        state["messages"], state.get("summary", ""))
    return {"messages": [response]}


def extract_preferences(state: OverallState) -> dict:
    """Looks at the latest human message and decides whether it states a
    lasting preference worth remembering (name, contact method, plan...)."""
    human_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    if not human_msgs:
        return {}
    last_human = human_msgs[-1]

    try:
        result = preference_llm.invoke([
            SystemMessage(content=(
                "Decide if this message states a lasting personal preference "
                "or fact (e.g. name, preferred contact method, plan tier). "
                "If yes, extract a short key and value. If it's just a "
                "one-off question, set has_preference to false."
            )),
            last_human,
        ])
    except Exception:
        return {}

    if result.has_preference and result.key:
        return {"user_preferences": {result.key: result.value}}
    return {}


def should_summarize(state: OverallState) -> Literal["summarize_node", "__end__"]:
    return "summarize_node" if len(state["messages"]) > SUMMARY_TRIGGER else "__end__"


def summarize_node(state: OverallState) -> dict:
    """Day 5: compress everything except the most recent KEEP_RECENT
    messages into a running summary, then delete those older messages
    from state using RemoveMessage (the add_messages reducer understands
    RemoveMessage tokens and deletes by id)."""
    messages = state["messages"]
    to_summarize = messages[:-KEEP_RECENT]
    if not to_summarize:
        return {}

    existing_summary = state.get("summary", "")
    convo_text = "\n".join(f"{m.type}: {m.content}" for m in to_summarize)
    prompt = (
        f"Existing summary: {existing_summary or '(none yet)'}\n\n"
        "Extend the summary in 2-4 sentences using the new messages below. "
        "Preserve names, preferences, and any decisions made."
    )
    try:
        response = llm.invoke([SystemMessage(content=prompt), HumanMessage(content=convo_text)])
        new_summary = response.content
    except Exception:
        new_summary = existing_summary

    remove_ops = [RemoveMessage(id=m.id) for m in to_summarize if m.id]
    return {"summary": new_summary, "messages": remove_ops}


# ---------------------------------------------------------------------------
# Build the graph — note the input=/output= schema split
# ---------------------------------------------------------------------------
graph = StateGraph(OverallState, input=InputState, output=OutputState)

graph.add_node("detect_intent", detect_intent)
graph.add_node("standard_node", standard_node)
graph.add_node("support_node", support_node)
graph.add_node("sales_node", sales_node)
graph.add_node("clarify_node", clarify_node)
graph.add_node("extract_preferences", extract_preferences)
graph.add_node("summarize_node", summarize_node)

graph.set_entry_point("detect_intent")

graph.add_conditional_edges("detect_intent", route_by_intent, {
    "support_node": "support_node",
    "sales_node": "sales_node",
    "standard_node": "standard_node",
    "clarify_node": "clarify_node",
})

for node in ["standard_node", "support_node", "sales_node", "clarify_node"]:
    graph.add_edge(node, "extract_preferences")

graph.add_conditional_edges("extract_preferences", should_summarize, {
    "summarize_node": "summarize_node",
    "__end__": END,
})
graph.add_edge("summarize_node", END)

# ---------------------------------------------------------------------------
# Day 1: SqliteSaver — persists to disk, survives a full process restart
# ---------------------------------------------------------------------------
DB_PATH = os.path.join(os.path.dirname(__file__), "chatbot_memory.db")
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
checkpointer = SqliteSaver(conn)

app = graph.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# CLI — uses streaming (Day 3) instead of a single invoke() call
# ---------------------------------------------------------------------------
def main():
    print("Integrated LangGraph Agent — SQLite memory, streaming, summarization")
    print("Type 'quit' to exit.\n")

    thread_id = input("Enter a session/thread id (e.g. 'yashika'): ").strip() or "default"
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    existing = app.get_state(config)
    if existing.values.get("messages"):
        print(f"(Resuming thread '{thread_id}' — {len(existing.values['messages'])} prior messages found)\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            print("Bye!")
            break
        if not user_input:
            continue

        print("Bot: ", end="", flush=True)
        final_intent = None
        # stream_mode="updates" yields ONE dict per node as it finishes,
        # e.g. {"standard_node": {"messages": [...]}}. Compare this to
        # invoke(), which blocks and returns only the final merged result.
        for update in app.stream({"messages": [HumanMessage(content=user_input)]}, config=config, stream_mode="updates"):
            for node_name, node_output in update.items():
                if node_name in ("standard_node", "support_node", "sales_node", "clarify_node"):
                    print(node_output["messages"][-1].content)
                if node_name == "detect_intent":
                    final_intent = node_output.get("intent")

        print(f"  [routed as: {final_intent}]\n")


if __name__ == "__main__":
    main()