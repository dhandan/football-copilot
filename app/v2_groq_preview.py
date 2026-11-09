"""Private, opt-in, read-only V2 preview for GitHub Codespaces.

This app is NOT a public deployment. The GitHub Codespaces forwarded port must
stay PRIVATE; Groq authentication is configured with a Codespaces secret.
Responses remain subject to V2 deterministic provenance and grounding checks.
"""
from __future__ import annotations

import os

import streamlit as st

from agent import groq_v2
from agent.langgraph_v2 import Conversation

st.set_page_config(page_title="Football Copilot V2 | Private Preview",
                   page_icon="⚽", layout="centered")
st.title("⚽ Football Copilot V2")
st.caption("Private experimental preview · completed Premier League results · source-backed answers")
st.info(
    "This is the V2 research branch, not the production chatbot.  "
    "Recognised comparisons use deterministic Python tools; Groq is used "
    "only when planning or answer synthesis is needed.  "
    "Groq's answer rewriting is still NO-GO following failed grounding and JSON checks."
)

# Do not connect unless the user has deliberately set the Codespaces secret.
# The existing GitHub Actions secret is a different secret store.
ready = groq_v2.groq_ready()
st.sidebar.header("Experimental runtime")
st.sidebar.write("**Groq model:** " + os.getenv("FOOTBALL_GROQ_MODEL", groq_v2.DEFAULT_MODEL))
st.sidebar.write("**Groq setup:** " + ("Ready (user-attested)" if ready else "Not configured"))
st.sidebar.caption(
    "The Free plan and Zero Data Retention flags are user attestations, "
    "not a verified reading of your Groq account."
)
st.sidebar.warning(
    "Keep forwarded port 8501 PRIVATE. Do not put any API key in chat or GitHub files. "
    "No production or prediction assets are modified."
)

if st.sidebar.button("Clear conversation", use_container_width=True):
    st.session_state.pop("v2_private_conversation", None)
    st.session_state.pop("v2_private_messages", None)
    st.rerun()

if not ready:
    st.error(
        "Groq is disabled: the Codespaces secret GROQ_API_KEY is not available. "
        "GitHub Actions secrets do NOT automatically transfer to Codespaces."
    )
    st.markdown(
        "Add **GROQ_API_KEY** under your GitHub repository's **Settings → "
        "Secrets and variables → Codespaces**, then restart your Codespace.  "
        "Confirm your Groq account remains on the Free tier with Zero Data Retention enabled."
    )
    st.stop()

if "v2_private_conversation" not in st.session_state:
    st.session_state.v2_private_conversation = Conversation(
        planner=groq_v2.groq_planner,
        synthesizer=groq_v2.groq_synthesizer,
    )
if "v2_private_messages" not in st.session_state:
    st.session_state.v2_private_messages = []

st.markdown("##### Try a real question")
examples = [
    ("Compare form", "Compare Liverpool and Arsenal over their last 10 matches"),
    ("Compare defending", "Who has been defensively better over those 10 matches?"),
    ("Ask why", "Why is that?"),
    ("Ask H07 (Groq)", "Why was H07 rejected?"),
    ("Model performance", "How has the 40/60 champion performed so far?"),
]
sample = None
for label, question in examples:
    if st.button(label, key="sample_" + label):
        sample = question

for item in st.session_state.v2_private_messages:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])
        if item["role"] == "assistant":
            st.caption(item["status"])
            with st.expander("How this answer was produced", expanded=False):
                st.write("Read-only tools:", ", ".join(item["tools"]) if item["tools"] else "None")
                st.write("Execution trace:", ", ".join(item["trace"]))
                if item["sources"]:
                    st.write("Evidence sources:", "; ".join(item["sources"]))

typed = st.chat_input("Ask about completed results, form, GW8 or model evaluations")
question = typed or sample
if question:
    st.session_state.v2_private_messages.append({"role": "user", "content": question})
    with st.spinner("Checking verified football data..."):
        before = groq_v2._calls
        try:
            result = st.session_state.v2_private_conversation.ask(
                question, thread_id="private-preview"
            )
            evidence = result.get("evidence", [])
            trace = result.get("trace", [])
            calls = groq_v2._calls - before
            outcome = (
                "Groq synthesis accepted" if "synthesis_accepted" in trace else
                "Groq answer rejected; verified facts retained"
                if calls and any(t.startswith("synthesis_") for t in trace) else
                "Groq tool planning and/or deterministic facts"
                if calls else "Deterministic tools; no Groq call"
            )
            sources = list(dict.fromkeys(
                e["result"]["source"]
                for e in evidence
                if isinstance(e.get("result"), dict) and e["result"].get("source")
            ))
            item = {
                "role": "assistant",
                "content": result["answer"],
                "tools": [e["tool"] for e in evidence],
                "sources": sources,
                "trace": trace,
                "status": f"{outcome} · {calls} Groq request(s) this turn",
            }
        except Exception:
            # Never render potentially sensitive exception messages.
            item = {
                "role": "assistant",
                "content": "The experimental request could not be completed. No football data was changed.",
                "tools": [], "sources": [], "trace": [],
                "status": "Safe error; request details not shown",
            }
        st.session_state.v2_private_messages.append(item)
    st.rerun()
