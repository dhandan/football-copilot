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
from agent.product_conversation_v2 import ProductConversation

def split_answer_for_display(answer: str) -> tuple[str, str | None]:
    """Show the verified conclusion first; keep its source tables accessible."""
    marker = "### Verified football evidence"
    if not answer.startswith("### Answer\n") or marker not in answer:
        return answer, None
    before, after = answer.split(marker, 1)
    if not before.strip() or not after.strip():
        return answer, None
    return before.strip(), marker + after


st.set_page_config(page_title="Football Copilot V2 | Private Preview",
                   page_icon="⚽", layout="centered")
st.title("⚽ Football Copilot V2")
st.caption("Private V2 product preview · verified Premier League insights · natural-language follow-ups")
st.info(
    "Completed-match questions use verified Python calculations by default, "
    "with no AI request. Historical comparisons, trends, venues and opponent "
    "context are read-only. Groq remains optional for legacy V2 enquiries. "
    "This is a private preview, not a production deployment."
)

# Do not connect unless the user has deliberately set the Codespaces secret.
# The existing GitHub Actions secret is a different secret store.
ready = groq_v2.groq_ready()
st.sidebar.header("Experimental runtime")
st.sidebar.caption("Build: integrated facts-first V2 · 18 Nov 2026")
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
    st.session_state.pop("v2_queued_question", None)
    st.rerun()

if not ready:
    st.sidebar.info("Groq is off. Verified recent-form questions still work without it.")

if "v2_private_conversation" not in st.session_state:
    legacy = (Conversation(planner=groq_v2.groq_planner,
                           synthesizer=groq_v2.groq_synthesizer)
              if ready else None)
    st.session_state.v2_private_conversation = ProductConversation(legacy=legacy)
if "v2_private_messages" not in st.session_state:
    st.session_state.v2_private_messages = []
if "v2_queued_question" not in st.session_state:
    st.session_state.v2_queued_question = None

st.markdown("##### Try a real question")
examples = [
    ("Recent form", "Compare Liverpool and Arsenal over their last 10 matches"),
    ("Five-match trend", "Has Arsenal's defence improved in its most recent five matches compared with the previous five?"),
    ("Opponent context", "Compare Arsenal and Liverpool's last seven matches and the strength of opponents they faced."),
    ("Why question", "Why did Brighton score more than Brentford over their last seven matches?"),
    ("H07 (optional Groq)", "Why was H07 rejected?"),
]
sample = None
for label, question in examples:
    if st.button(label, key="sample_" + label):
        sample = question

messages = st.session_state.v2_private_messages
latest_answer_index = next(
    (i for i in range(len(messages) - 1, -1, -1)
     if messages[i]["role"] == "assistant"), None
)
for idx, item in enumerate(messages):
    with st.chat_message(item["role"]):
        summary, details = (split_answer_for_display(item["content"])
                            if item["role"] == "assistant"
                            else (item["content"], None))
        st.markdown(summary)
        if details:
            with st.expander("Verified statistics and sources", expanded=False):
                st.markdown(details)
        if item["role"] == "assistant":
            st.caption(item["status"])
            if idx == latest_answer_index and item.get("followups"):
                st.caption("Ask a follow-up")
                columns = st.columns(len(item["followups"]))
                for i, suggestion in enumerate(item["followups"]):
                    if columns[i].button(suggestion, key=f"followup_{idx}_{i}",
                                         use_container_width=True):
                        st.session_state.v2_queued_question = suggestion
                        st.rerun()
            with st.expander("How this answer was produced", expanded=False):
                st.write("Read-only tools:", ", ".join(item["tools"]) if item["tools"] else "None")
                st.write("Execution trace:", ", ".join(item["trace"]))
                if item["sources"]:
                    st.write("Evidence sources:", "; ".join(item["sources"]))

typed = st.chat_input("Ask about completed results, form, GW8 or model evaluations")
question = typed or st.session_state.v2_queued_question or sample
st.session_state.v2_queued_question = None
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
                "Verified Python investigation" if
                result.get("route") == "deterministic_investigation" else
                "Evidence audit rejected incomplete answer" if
                result.get("route") == "verified_data_rejected" else
                "Legacy V2 route (optional Groq)" if
                result.get("route") == "legacy_v2" else
                "Outside supported offline queries"
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
                "followups": result.get("followups", []) if result.get("complete") else [],
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
