"""
Day 5, Part B: Web UI with Streamlit
--------------------------------------
Wraps the same graph from chatbot_day5_memory.py in a simple web chat
interface instead of the terminal. Run with:

    streamlit run app_streamlit.py
"""

import streamlit as st
from langchain_core.messages import HumanMessage

from chatbot_day5_memory import app  # reuse the compiled graph + checkpointer

st.set_page_config(page_title="LangGraph Intent-Routing Chatbot", page_icon="🤖")
st.title("🤖 Intent-Routing Chatbot")
st.caption("Routes between support, sales, and general responders — built with LangGraph")

# Each browser session gets its own thread_id -> its own memory
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "streamlit-session-1"
if "history" not in st.session_state:
    st.session_state.history = []  # for rendering only; the checkpointer holds real state

config = {"configurable": {"thread_id": st.session_state.thread_id}}

# Render past messages
for role, content in st.session_state.history:
    with st.chat_message(role):
        st.markdown(content)

user_input = st.chat_input("Type a message...")

if user_input:
    st.session_state.history.append(("user", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = app.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)
            reply = result["messages"][-1].content
            intent = result["intent"]
        st.markdown(reply)
        st.caption(f"routed as: `{intent}`")

    st.session_state.history.append(("assistant", reply))