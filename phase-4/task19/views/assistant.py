import uuid

import streamlit as st

from api_client import ApiError
from components import alerts


def _new_session():
    st.session_state["chat_session"] = f"console-{uuid.uuid4().hex[:12]}"
    st.session_state["chat_messages"] = []


def render(client, cfg):
    st.header("HR Assistant")
    st.write("Ask general HR policy questions (leave, notice period, holidays…). The assistant answers only from the approved FAQ list. "
             "It cannot see any employee's records, so it cannot look up anyone's salary or leave balance.")
    st.caption("AI-generated responses may contain errors. Verify important HR decisions with an authorized human.")
    if "chat_session" not in st.session_state:
        _new_session()
    st.button("Start a new conversation", on_click=_new_session)

    if not st.session_state["chat_messages"]:
        alerts.empty_state('No messages yet. Try: "What is the notice period?"')
    for m in st.session_state["chat_messages"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] == "assistant":
                if m.get("fallback"):
                    st.caption("The assistant could not answer this confidently. Please contact HR for help.")
                elif m.get("intent"):
                    st.caption(f"Topic: {m['intent']} · confidence {m.get('confidence', 0):.0%}")
                alerts.review_label("chat")

    prompt = st.chat_input("Ask the HR assistant…", max_chars=500)
    if prompt and prompt.strip() and alerts.single_flight("chat_busy"):
        st.session_state["chat_messages"].append({"role": "user", "content": prompt.strip()})
        try:
            with st.spinner("Thinking…"):
                r = client.chat(prompt.strip(), st.session_state["chat_session"])
            st.session_state["chat_messages"].append({"role": "assistant", "content": r["reply"], "intent": r.get("intent"),
                                                       "confidence": r.get("confidence"), "fallback": bool(r["fallback"])})
        except ApiError as e:
            st.session_state["chat_messages"].append({"role": "assistant", "content": f"⚠️ {e.user_message}", "fallback": False})
        finally:
            alerts.release("chat_busy")
        st.rerun()
