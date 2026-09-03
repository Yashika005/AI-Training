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

MAX_SUPERVISOR_TURNS = 6  # safety valve so a confused supervisor can't loop forever


# ---------------------------------------------------------------------------
# Reducer functions for merging state across nodes
# ---------------------------------------------------------------------------
def append_log(existing: list, new: list) -> list:
    return (existing or []) + (new or [])


# ---------------------------------------------------------------------------
# Parent (supervisor) state
# ---------------------------------------------------------------------------
class SupervisorState(TypedDict):
    messages: Annotated[list, add_messages]
    next_agent: str
    web_search_result: str
    written_content: str
    final_output: str
    error_log: Annotated[list, append_log]
    turns: int


# ---------------------------------------------------------------------------
# Day 4: Web Search Agent -- its own independent subgraph
# ---------------------------------------------------------------------------
class WebSearchState(TypedDict):
    query: str
    result: str
    error: str


def web_search_tool(query: str) -> str:
    tavily_key = os.getenv("TAVILY_API_KEY")
    if tavily_key:
        from langchain_tavily import TavilySearch
        search = TavilySearch(max_results=3)
        response = search.invoke(query)
        results = response.get("results", []) if isinstance(response, dict) else response
        return "\n".join(r.get("content", "") for r in results if isinstance(r, dict))
    return (
        f"[MOCK SEARCH RESULT for query: '{query}'] "
        "(Set TAVILY_API_KEY in .env for real web search results.)"
    )


def run_search(state: WebSearchState) -> dict:
    try:
        result = web_search_tool(state["query"])
        return {"result": result, "error": ""}
    except Exception as e:
        return {"result": "", "error": str(e)}


web_search_graph = StateGraph(WebSearchState)
web_search_graph.add_node("run_search", run_search)
web_search_graph.set_entry_point("run_search")
web_search_graph.add_edge("run_search", END)
web_search_app = web_search_graph.compile()  # a fully independent, reusable subgraph


# ---------------------------------------------------------------------------
# Day 4: Writer Agent -- its own independent subgraph
# ---------------------------------------------------------------------------
class WriterState(TypedDict):
    task: str
    context: str
    content: str
    error: str


def write_content(state: WriterState) -> dict:
    try:
        prompt = (
            f"Write a clear, well-structured response to this request:\n{state['task']}\n\n"
            f"Use this research context if it's relevant (ignore if not useful):\n{state['context']}"
        )
        response = llm.invoke([
            SystemMessage(content="You are a skilled writer. Be concise and well-organized."),
            HumanMessage(content=prompt),
        ])
        return {"content": response.content, "error": ""}
    except Exception as e:
        return {"content": "", "error": str(e)}


writer_graph = StateGraph(WriterState)
writer_graph.add_node("write_content", write_content)
writer_graph.set_entry_point("write_content")
writer_graph.add_edge("write_content", END)
writer_app = writer_graph.compile()


# ---------------------------------------------------------------------------
# Day 3: Supervisor decision (structured output)
# ---------------------------------------------------------------------------
class SupervisorDecision(BaseModel):
    next_agent: Literal["web_search_agent", "writer_agent", "FINISH"] = Field(
        description="Which specialized agent should act next, or FINISH if the task is complete."
    )
    reason: str = Field(description="One short sentence explaining the decision.")


supervisor_llm = llm.with_structured_output(SupervisorDecision)


def supervisor_node(state: SupervisorState) -> dict:
    last_user_msg = state["messages"][-1].content

    briefing = (
        f"User request: {last_user_msg}\n\n"
        f"Web search result so far: {state.get('web_search_result') or '(none yet)'}\n"
        f"Written content so far: {state.get('written_content') or '(none yet)'}\n\n"
        "You manage two agents:\n"
        "- web_search_agent: looks up current information on the web\n"
        "- writer_agent: writes a polished response using the request and any research available\n\n"
        "Decide the single next step. If the request needs current/factual info "
        "and none has been gathered yet, choose web_search_agent. If research is "
        "done (or unnecessary) and no content has been written yet, choose "
        "writer_agent. If written_content already exists and looks complete, choose FINISH."
    )

    try:
        decision = supervisor_llm.invoke([SystemMessage(content=briefing)])
        next_agent = decision.next_agent
    except Exception as e:
        # Day 5: error handling -- if the supervisor itself fails, fail safe to FINISH
        # rather than looping or crashing.
        return {
            "next_agent": "FINISH",
            "turns": state.get("turns", 0) + 1,
            "error_log": [{"node": "supervisor_node", "error": str(e)}],
        }

    return {"next_agent": next_agent, "turns": state.get("turns", 0) + 1}


def route_supervisor(state: SupervisorState) -> Literal["web_search_agent", "writer_agent", "aggregate_node"]:
    # Safety valve: force finish if we've looped too many times
    if state.get("turns", 0) >= MAX_SUPERVISOR_TURNS:
        return "aggregate_node"
    return {
        "web_search_agent": "web_search_agent",
        "writer_agent": "writer_agent",
    }.get(state["next_agent"], "aggregate_node")  # FINISH (or anything else) -> aggregate


# ---------------------------------------------------------------------------
# Wrapper nodes: translate parent state <-> child subgraph state
# ---------------------------------------------------------------------------
def web_search_node(state: SupervisorState) -> dict:
    query = state["messages"][-1].content
    result_state = web_search_app.invoke({"query": query, "result": "", "error": ""})

    if result_state.get("error"):
        # Day 5: fallback routing -- don't crash, record the failure and let the
        # supervisor proceed (writer_agent can still write without research).
        return {
            "web_search_result": "(web search unavailable)",
            "error_log": [{"node": "web_search_node", "error": result_state["error"]}],
        }
    return {"web_search_result": result_state["result"]}


def writer_node(state: SupervisorState) -> dict:
    task = state["messages"][-1].content
    context = state.get("web_search_result", "")
    result_state = writer_app.invoke({"task": task, "context": context, "content": "", "error": ""})

    if result_state.get("error"):
        return {
            "written_content": "(writer agent unavailable -- please try again)",
            "error_log": [{"node": "writer_node", "error": result_state["error"]}],
        }
    return {"written_content": result_state["content"]}


def aggregate_node(state: SupervisorState) -> dict:
    """Day 5: result aggregation -- combine whatever was produced into one
    final answer, and surface any errors that happened along the way."""
    parts = []
    if state.get("written_content"):
        parts.append(state["written_content"])
    elif state.get("web_search_result"):
        parts.append(f"Research found:\n{state['web_search_result']}")
    else:
        parts.append("No content was produced.")

    if state.get("error_log"):
        parts.append(f"\n(Note: {len(state['error_log'])} internal issue(s) occurred during processing.)")

    final_output = "\n".join(parts)
    return {
        "final_output": final_output,
        "messages": [AIMessage(content=final_output)],
    }


# ---------------------------------------------------------------------------
# Build the supervisor graph
# ---------------------------------------------------------------------------
def build_supervisor_graph() -> StateGraph:
    graph = StateGraph(SupervisorState)

    graph.add_node("supervisor_node", supervisor_node)
    graph.add_node("web_search_agent", web_search_node)
    graph.add_node("writer_agent", writer_node)
    graph.add_node("aggregate_node", aggregate_node)

    graph.set_entry_point("supervisor_node")
    graph.add_conditional_edges("supervisor_node", route_supervisor, {
        "web_search_agent": "web_search_agent",
        "writer_agent": "writer_agent",
        "aggregate_node": "aggregate_node",
    })
    graph.add_edge("web_search_agent", "supervisor_node")
    graph.add_edge("writer_agent", "supervisor_node")
    graph.add_edge("aggregate_node", END)

    return graph


app = build_supervisor_graph().compile()  # no checkpointer yet -- added in Part 2


# ---------------------------------------------------------------------------
# CLI for quick end-to-end testing (Day 5)
# ---------------------------------------------------------------------------
def main():
    print("Multi-Agent Supervisor (web search + writer) — type 'quit' to exit\n")
    while True:
        user_input = input("Request: ").strip()
        if user_input.lower() in ("quit", "exit"):
            break
        if not user_input:
            continue

        result = app.invoke({
            "messages": [HumanMessage(content=user_input)],
            "next_agent": "",
            "web_search_result": "",
            "written_content": "",
            "final_output": "",
            "error_log": [],
            "turns": 0,
        })

        print(f"\n--- Final Output ---\n{result['final_output']}\n")
        if result.get("error_log"):
            print(f"(errors encountered: {result['error_log']})\n")


if __name__ == "__main__":
    main()