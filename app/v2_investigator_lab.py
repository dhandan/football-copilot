"""Isolated read-only V2 investigator lab (NOT the existing V2 preview).

Launch manually in a PRIVATE Codespace on port 8502:
    python -m streamlit run app/v2_investigator_lab.py --server.port 8502

Only explicit chat submissions trigger requests. No model answer rewriting,
no automatic background calls, no production routes or prediction changes.
"""
from __future__ import annotations

import os
import streamlit as st

from agent.investigator_v2 import Investigator
from agent import groq_v2

st.set_page_config(page_title="Football Copilot | Investigator Lab",
                   page_icon="⚽", layout="centered")
st.title("⚽ Football Copilot — Investigator Lab")
st.caption("EXPERIMENTAL · Private Codespace only · Two-round bounded evidence investigation")
st.warning(
    "This is a research interface, not the production copilot. "
    "The AI only proposes read-only tools; Python calculates the football facts. "
    "A complete evidence plan is NOT proof of an accurate explanatory answer. "
    "No autonomous Groq model has passed promotion criteria."
)

ready = groq_v2.groq_ready()
st.sidebar.write("**Private runtime:** Research only")
st.sidebar.write("**Groq Free/ZDR/key flags:** " +
                 ("present (user-attested)" if ready else "not ready"))
st.sidebar.write("**Model:** " + os.getenv("FOOTBALL_GROQ_MODEL", "openai/gpt-oss-20b"))
st.sidebar.caption("Only use a Codespaces port marked PRIVATE.")
st.sidebar.caption("Bounded to two planning rounds and up to six read-only tool calls.")
if st.sidebar.button("Reset experiment"):
    st.session_state.pop("inv_lab", None)
    st.session_state.pop("inv_lab_messages", None)
    st.rerun()

if not ready:
    st.error(
        "The experimental Groq planner is unavailable. Confirm Groq Free, "
        "Zero Data Retention and the private Codespaces API secret. "
        "No fallback or remote requests were made."
    )
    st.stop()

if "inv_lab" not in st.session_state:
    # Shadow protocol is scoped to this dedicated experiment process.
    # Do not run the production preview in the same Python process.
    os.environ["FOOTBALL_GROQ_SHADOW_PROTOCOL"] = "strict_budget"
    st.session_state.inv_lab = Investigator(groq_v2.groq_planner)
if "inv_lab_messages" not in st.session_state:
    st.session_state.inv_lab_messages = []

examples = [
    "Compare Brighton and Brentford over their last seven matches. "
    "Who conceded more, and who faced stronger opponents?",
    "Compare Arsenal and Liverpool's last nine matches, "
    "including away defending and opponents faced.",
    "Show Chelsea and Tottenham's recent five-match form "
    "and historical head-to-head record.",
]
with st.expander("Sample investigative questions"):
    for idx, q in enumerate(examples):
        st.code(q, language="text")

for turn in st.session_state.inv_lab_messages:
    with st.chat_message(turn["role"]):
        st.markdown(turn["text"])
        if turn["role"] == "assistant":
            st.caption(turn["status"])
            with st.expander("Evidence audit"):
                st.write("Tool calls:", turn["tools"])
                st.write("Unresolved:", turn["gaps"])
                st.write("Trace:", turn["trace"])

question = st.chat_input("Ask a multi-part question about completed Premier League matches")
if question:
    st.session_state.inv_lab_messages.append({"role": "user", "text": question})
    with st.spinner("Investigating verified match evidence..."):
        before = groq_v2._calls
        try:
            result = st.session_state.inv_lab.ask(question, thread_id="private-investigator")
            used = groq_v2._calls - before
            turn = {
                "role": "assistant", "text": result["answer"],
                "status": f"{'Complete tool evidence' if result['complete'] else 'Partial or missing evidence'}"
                          f" · {result['rounds']} round(s) · {used} Groq request(s)",
                "tools": [e["tool"] for e in result["evidence"] if "result" in e],
                "gaps": [e["name"] for e in result["gaps"]],
                "trace": result["trace"],
            }
        except Exception:
            turn = {
                "role": "assistant",
                "text": "The investigation could not be completed safely. No football data was changed.",
                "status": "Safe failure", "tools": [], "gaps": [], "trace": [],
            }
        st.session_state.inv_lab_messages.append(turn)
    st.rerun()
