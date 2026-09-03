"""
hitl_time_travel.py — Days 6-10
===================================
Builds on multi_agent_supervisor.py (imports its graph builder + nodes,
does not duplicate them). Adds:

  Day 6: HITL concepts -- interrupts, breakpoints, pause/resume, state inspection
  Day 7: Interrupt BEFORE a critical node + CLI approval flow + resume
  Day 8: Time travel -- state history, checkpoint replay, rewinding execution
  Day 9: Editing state after interruption, replaying from a past checkpoint,
         testing approve/reject scenarios
  Day 10: Everything combined + logging + LangSmith tracing notes

Critical node chosen for the approval gate: aggregate_node -- i.e. the
graph PAUSES right before it finalizes/delivers output, and a human must
approve, edit, or reject before it continues.
"""

import os
import sqlite3

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from multi_agent_supervisor import build_supervisor_graph, SupervisorState

load_dotenv()

# LangSmith (Day 10): if LANGCHAIN_TRACING_V2 / LANGCHAIN_API_KEY /
# LANGCHAIN_PROJECT are set in .env, every node call below -- including the
# pause and resume -- shows up as a single traced run in the LangSmith UI.
# No extra code needed here beyond load_dotenv().

DB_PATH = os.path.join(os.path.dirname(__file__), "supervisor_memory.db")


def get_checkpointer():
    # Prefer SqliteSaver (persists across restarts); fall back to MemorySaver
    # if the sqlite checkpoint package isn't installed, so this file still runs.
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        return SqliteSaver(conn)
    except ImportError:
        from langgraph.checkpoint.memory import MemorySaver
        print("[note] langgraph-checkpoint-sqlite not installed, using in-memory checkpointer instead.")
        return MemorySaver()


checkpointer = get_checkpointer()

# ---------------------------------------------------------------------------
# Day 7: compile WITH interrupt_before -- this is the actual breakpoint.
# The graph will run supervisor_node -> ... -> right up to aggregate_node,
# then STOP and hand control back to us before aggregate_node executes.
# ---------------------------------------------------------------------------
app = build_supervisor_graph().compile(
    checkpointer=checkpointer,
    interrupt_before=["aggregate_node"],
)


def _fresh_input(user_text: str) -> dict:
    return {
        "messages": [HumanMessage(content=user_text)],
        "next_agent": "",
        "web_search_result": "",
        "written_content": "",
        "final_output": "",
        "error_log": [],
        "turns": 0,
    }


# ---------------------------------------------------------------------------
# Day 6/7: run until the breakpoint, show pending state, get human approval
# ---------------------------------------------------------------------------
def run_with_approval(user_text: str, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}

    print(f"\n>>> Running request: {user_text}")
    app.invoke(_fresh_input(user_text), config=config)

    # Day 6: state inspection -- see exactly what's queued to run next
    snapshot = app.get_state(config)
    print(f"[paused] next node(s) to run: {snapshot.next}")
    print(f"[paused] web_search_result: {snapshot.values.get('web_search_result')!r}")
    print(f"[paused] written_content:\n{snapshot.values.get('written_content')}\n")

    while True:
        choice = input("Approve and finalize? (yes / no / edit): ").strip().lower()

        if choice in ("yes", "y"):
            # Day 7: resume execution. Passing None means "continue from where
            # we paused" using the state already saved in the checkpoint.
            final = app.invoke(None, config=config)
            print(f"\n--- FINAL OUTPUT (approved) ---\n{final['final_output']}\n")
            return final

        elif choice in ("no", "n"):
            print("Rejected -- graph left paused, nothing finalized.")
            return None

        elif choice in ("edit", "e"):
            # Day 9: editing state after interruption. update_state writes new
            # values into the CURRENT checkpoint, as if a node had produced them.
            new_text = input("Enter replacement written_content: ").strip()
            app.update_state(config, {"written_content": new_text}, as_node="writer_agent")
            print("[state updated] re-inspecting before you decide again...")
            snapshot = app.get_state(config)
            print(f"written_content is now:\n{snapshot.values.get('written_content')}\n")
            # loop back to ask approve/reject again with the edited content

        else:
            print("Please type yes, no, or edit.")


# ---------------------------------------------------------------------------
# Day 8/9: Time travel -- inspect and replay from any past checkpoint
# ---------------------------------------------------------------------------
def time_travel_demo(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}

    history = list(app.get_state_history(config))
    if not history:
        print(f"No history found for thread '{thread_id}'. Run it once with run_with_approval() first.")
        return

    print(f"\n--- Checkpoint history for '{thread_id}' (most recent first) ---")
    for i, snap in enumerate(history):
        print(f"[{i}] step={snap.metadata.get('step')} next={snap.next} "
              f"written_content={bool(snap.values.get('written_content'))}")

    idx = input("\nEnter the index to rewind to: ").strip()
    try:
        idx = int(idx)
        target_snapshot = history[idx]
    except (ValueError, IndexError):
        print("Invalid index.")
        return

    # This IS the rewind: target_snapshot.config points at a specific past
    # checkpoint_id. Using it as `config` for update_state/invoke operates
    # on that point in history, not the latest one.
    rewind_config = target_snapshot.config
    print(f"\nRewound to step {target_snapshot.metadata.get('step')}.")
    print(f"State at that point -- written_content: {target_snapshot.values.get('written_content')!r}")

    edit = input("Edit written_content before replaying from here? (leave blank to skip): ").strip()
    if edit:
        # Day 9: modifying state before resuming -- this creates a NEW branch
        # forked off the old checkpoint; the original history is untouched.
        rewind_config = app.update_state(rewind_config, {"written_content": edit}, as_node="writer_agent")

    print("Replaying execution forward from this checkpoint...")
    replayed = app.invoke(None, config=rewind_config)
    print(f"\n--- Result after replay ---\n{replayed.get('final_output', '(graph paused again -- inspect and approve)')}\n")


# ---------------------------------------------------------------------------
# CLI menu tying Days 6-10 together
# ---------------------------------------------------------------------------
def main():
    print("HITL + Time Travel Demo (Days 6-10)")
    print("1) Run a request with human approval gate")
    print("2) Time-travel through a thread's checkpoint history")
    print("3) Quit")

    while True:
        choice = input("\nChoose 1/2/3: ").strip()

        if choice == "1":
            thread_id = input("Thread id: ").strip() or "default"
            user_text = input("What should the agents do? ").strip()
            run_with_approval(user_text, thread_id)

        elif choice == "2":
            thread_id = input("Thread id to inspect: ").strip() or "default"
            time_travel_demo(thread_id)

        elif choice == "3":
            print("Bye!")
            break

        else:
            print("Please choose 1, 2, or 3.")


if __name__ == "__main__":
    main()