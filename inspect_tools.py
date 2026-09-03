import sys
from agent import app


def show_current_state(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = app.get_state(config)

    print(f"--- Current state for thread '{thread_id}' ---")
    if not snapshot.values:
        print("(no saved state — this thread hasn't been used yet)\n")
        return

    print(f"intent:            {snapshot.values.get('intent')}")
    print(f"summary:           {snapshot.values.get('summary')}")
    print(f"user_preferences:  {snapshot.values.get('user_preferences')}")
    print(f"message count:     {len(snapshot.values.get('messages', []))}")
    print(f"next node to run:  {snapshot.next}")
    print()


def show_state_history(thread_id: str, limit: int = 10):
    config = {"configurable": {"thread_id": thread_id}}

    print(f"--- Checkpoint history for thread '{thread_id}' (most recent first) ---")
    count = 0
    for state_snapshot in app.get_state_history(config):
        count += 1
        if count > limit:
            break
        msg_count = len(state_snapshot.values.get("messages", []))
        print(f"[{count}] step={state_snapshot.metadata.get('step')} "
              f"messages={msg_count} intent={state_snapshot.values.get('intent')}")
    if count == 0:
        print("(no history found — this thread hasn't been used yet)")
    print()


def visualize_graph():
    """Saves the graph structure to graph.mmd (Mermaid text) and, if the
    optional rendering dependency is available, graph.png as well."""
    mermaid_text = app.get_graph().draw_mermaid()
    with open("graph.mmd", "w") as f:
        f.write(mermaid_text)
    print("Saved graph structure to graph.mmd (paste into https://mermaid.live to view)")

    try:
        png_bytes = app.get_graph().draw_mermaid_png()
        with open("graph.png", "wb") as f:
            f.write(png_bytes)
        print("Saved graph.png")
    except Exception as e:
        print(f"(Skipped PNG render — needs extra deps: {e})")


if __name__ == "__main__":
    thread_id = sys.argv[1] if len(sys.argv) > 1 else "default"
    visualize_graph()
    print()
    show_current_state(thread_id)
    show_state_history(thread_id)