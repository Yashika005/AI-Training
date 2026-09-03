
## 1. Core Concepts

### StateGraph
The `StateGraph` is the graph object itself — the container you attach
nodes and edges to, then `compile()` into a runnable app.

### State
A single shared object (here, a `TypedDict`) that flows through every node.
Each node reads what it needs from state and returns a **partial update**;
LangGraph merges that update back into the overall state automatically.

```python
class State(TypedDict):
    messages: Annotated[list, add_messages]
    intent: str
```

The `Annotated[list, add_messages]` part is a **reducer** — it tells
LangGraph "when a node returns new messages, append them to the existing
list instead of overwriting it." This is what gives the bot conversation
memory.

### Nodes
Plain Python functions that take `state` in and return a dict of updates.

```python
def detect_intent(state: State) -> dict:
    ...
    return {"intent": intent}
```

### Edges
Fixed connections between nodes — "after node A always go to node B."

```python
graph.add_edge("standard_node", END)
```

### Conditional Edges
Dynamic routing — a function inspects the state and returns the *name* of
the next node. This is the mechanism that makes intent-based routing work.

```python
def route_by_intent(state: State) -> Literal["support_node", "standard_node"]:
    return "support_node" if state["intent"] == "support" else "standard_node"

graph.add_conditional_edges("detect_intent", route_by_intent, {
    "support_node": "support_node",
    "standard_node": "standard_node",
})
```

---

## 2. How the Graph Flows

```
        START
          |
          v
   [detect_intent]  <-- reads last user message, sets state["intent"]
          |
   (conditional edge: route_by_intent)
          |
   -------+-------
   |             |
   v             v
[standard_node] [support_node]
   |             |
   -------+-------
          |
          v
         END
```

Every turn of the conversation runs through this entire path. The `intent`
field is recalculated fresh each turn, so the same conversation can bounce
between "general" and "support" depending on what the user types.

---

## 3. File Overview

| File | Purpose |
|---|---|
| `chatbot.py` | The full LangGraph app + CLI chat loop |
| `requirements.txt` | Python dependencies |
| `.env.example` | Template for your API key — rename to `.env` and fill in |

### Setup

```bash
pip install -r requirements.txt
cp .env.example .env    # then paste your Groq key into .env
python chatbot.py
```

Try messages like:
- `"Hey, what's the capital of France?"` → routes to **standard_node**
- `"I have an issue, my payment failed"` → routes to **support_node**

---

## 4. Three-Day Build Plan

### Day 1 — Environment Setup + State & Graph Skeleton
- Install dependencies, get your free Groq API key, set up `.env`
- Define the `State` TypedDict (`messages`, `intent`)
- Build a minimal graph with just one node that echoes input back,
  to confirm you understand how state enters and exits a node
- Goal: run `app.invoke({"messages": [...]})` and see it work end-to-end
  before adding any real logic

### Day 2 — Intent Detection + Conditional Routing
- Write `detect_intent()` and `route_by_intent()`
- Wire up `add_conditional_edges` so the graph actually branches
- Temporarily replace the two destination nodes with `print()`-only
  stubs so you can test routing in isolation, without worrying about
  the LLM yet
- Goal: confirm support-flavored messages go to one branch and general
  messages go to the other, every time

### Day 3 — Real LLM Nodes + Memory Loop + Testing
- Replace the stub nodes with real `standard_node` and `support_node`
  functions that call the LLM with different system prompts
- Add the CLI `while True` loop so state persists across turns
  (this is what gives you multi-turn memory)
- Test a full conversation that switches between general and support
  intents, and fix any edge cases (empty input, ambiguous phrasing)
- Goal: a working chatbot that matches the Week 1 outcome — data
  flowing through a graph via a centralized State object, routed by
  conditional edges

---

## 5. Things to Experiment With Once It Works

- Add a third node/intent category (e.g. "sales")
- Make `detect_intent` use the LLM itself for classification instead of
  keyword matching
- Add a summarization node that trims `messages` once the conversation
  gets long
- Visualize the graph with `app.get_graph().draw_mermaid()`