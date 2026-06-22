import logging
import pandas as pd
import streamlit as st

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

from data_engine import (
    DATA_SUMMARY,
    best_value_players,
    compare_players,
    filter_players,
    team_aggregation,
    top_n_players,
)
from llm_handler import classify_intent, generate_summary

st.set_page_config(page_title="FIFA Data Assistant", page_icon="⚽", layout="wide")

EXAMPLE_QUESTIONS = [
    "Show me the top 10 players by overall rating",
    "Find the best young players under age 23",
    "Compare Messi and Ronaldo",
    "Show me the best strikers with pace above 85",
    "Which teams have the highest average player rating?",
    "Give me a short analysis of the best value players",
]

FILTER_KEYS = {"max_age", "min_pace", "min_overall", "position", "club", "nationality", "min_potential"}


def _fmt(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "value_eur" in df.columns:
        df["value_eur"] = df["value_eur"].apply(lambda x: f"€{x/1_000_000:.1f}M" if x > 0 else "—")
    if "wage_eur" in df.columns:
        df["wage_eur"] = df["wage_eur"].apply(lambda x: f"€{int(x):,}/wk" if x > 0 else "—")
    for col in ("overall", "potential"):
        if col in df.columns:
            df[col] = df[col].astype(int)
    if "value_score" in df.columns:
        df["value_score"] = df["value_score"].round(2)
    if "avg_value_eur" in df.columns:
        df["avg_value_eur"] = df["avg_value_eur"].apply(lambda x: f"€{x/1_000_000:.1f}M" if x > 0 else "—")
    return df


def _init_state():
    for k, v in [("question_text", ""), ("do_search", False)]:
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📊 Dataset Info")
    st.metric("Players", f"{DATA_SUMMARY['total_players']:,}")
    st.metric("Clubs", DATA_SUMMARY["total_clubs"])
    st.metric("Overall Range", f"{DATA_SUMMARY['overall_range'][0]}–{DATA_SUMMARY['overall_range'][1]}")

    st.divider()
    st.subheader("Try these questions:")
    for q in EXAMPLE_QUESTIONS:
        if st.button(q, key=f"ex_{q[:22]}", use_container_width=True):
            st.session_state.question_text = q
            st.session_state.do_search = True
            st.rerun()

    st.divider()
    with st.expander("How it works"):
        st.caption(
            "1. Your question → Llama 3.1 classifies intent + extracts params\n"
            "2. Pandas queries the FIFA 23 dataset\n"
            "3. Llama 3.1 writes a brief insight\n\n"
            "Data logic is deterministic Python — LLM only handles language."
        )

# ── Main ─────────────────────────────────────────────────────────────────────
st.title("⚽ FIFA Data Assistant")
st.caption("Ask questions about FIFA 23 player data")

question = st.text_input(
    "Ask a question:",
    key="question_text",
    placeholder="e.g. Compare Messi and Ronaldo",
)

col1, col2 = st.columns([1, 8])
with col1:
    if st.button("Search", type="primary"):
        st.session_state.do_search = True
        st.rerun()
with col2:
    if st.button("Clear"):
        st.session_state.question_text = ""
        st.session_state.do_search = False
        st.rerun()

if st.session_state.do_search and st.session_state.question_text.strip():
    st.session_state.do_search = False
    q = st.session_state.question_text.strip()
    log.info("QUERY: %s", q)

    with st.spinner("Classifying intent..."):
        intent_result = classify_intent(q)

    intent = intent_result.get("intent", "unknown")
    params = intent_result.get("params", {})
    confidence = intent_result.get("confidence", "low")
    log.info("INTENT: %s | confidence: %s | params: %s", intent, confidence, params)

    badge = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(confidence, "🔴")
    st.caption(f"{badge} Intent: **{intent}** | Confidence: {confidence}")

    result = None
    err = None

    with st.spinner("Querying data..."):
        try:
            if intent == "top_n":
                result = top_n_players(
                    n=params.get("n", 10),
                    position=params.get("position"),
                    min_overall=params.get("min_overall"),
                )
            elif intent == "compare":
                players = params.get("players", [])
                if len(players) < 2:
                    err = "Need at least 2 player names to compare."
                else:
                    result = compare_players(players[0], players[1])
            elif intent == "filter":
                clean = {k: v for k, v in params.items() if k in FILTER_KEYS and v is not None}
                result = filter_players(**clean)
            elif intent == "team_stats":
                result = team_aggregation(top_n=params.get("top_n", 10))
            elif intent == "best_value":
                result = best_value_players(
                    top_n=params.get("top_n", 15),
                    min_overall=params.get("min_overall", 80),
                )
            else:
                err = "Couldn't understand that question. Try asking about top players, player comparisons, or filtering by position/age/pace."
        except Exception as e:
            err = f"Data error: {e}"
            log.error("DATA ERROR: %s", e)

    if err:
        log.warning("ERR: %s", err)
        st.error(err)

    elif result is not None:
        rows = len(result) if isinstance(result, pd.DataFrame) else len(result.get("found", []))
        log.info("RESULT: %s rows/players", rows)
        if isinstance(result, pd.DataFrame):
            if result.empty:
                st.warning("No players match these filters. Try broadening your search.")
            else:
                st.success(f"{len(result)} result(s) found")
                st.dataframe(_fmt(result), use_container_width=True)

        elif isinstance(result, dict):
            comp = result.get("comparison")
            not_found = result.get("not_found", [])
            if not_found:
                st.warning(f"Player(s) not found: {', '.join(not_found)}")
            if comp is not None:
                st.success(f"Comparing: {' vs '.join(result.get('found', []))}")
                comp_orig = comp.copy()
                comp_disp = comp.copy().astype(object)
                for idx in comp_disp.index:
                    if idx == "value_eur":
                        comp_disp.loc[idx] = comp_disp.loc[idx].apply(lambda x: f"€{x/1_000_000:.1f}M")
                    elif idx == "wage_eur":
                        comp_disp.loc[idx] = comp_disp.loc[idx].apply(lambda x: f"€{int(x):,}/wk")
                    else:
                        comp_disp.loc[idx] = comp_disp.loc[idx].apply(lambda x: str(int(x)))

                def _hl(row):
                    orig = comp_orig.loc[row.name]
                    if orig.nunique() == 1:
                        return [""] * len(row)
                    return [
                        "background-color: #1a6b3c; color: white; font-weight: bold" if v == orig.max() else ""
                        for v in orig
                    ]

                st.dataframe(comp_disp.style.apply(_hl, axis=1), use_container_width=True)
            elif not not_found:
                st.warning("Could not find players to compare.")

        with st.spinner("Generating insight..."):
            summary = generate_summary(intent, params, result, q)
        if summary:
            log.info("SUMMARY: generated (%d chars)", len(summary))
            st.info(f"💡 {summary}")
        else:
            log.warning("SUMMARY: unavailable")
            st.caption("AI summary unavailable")

st.divider()
st.caption("Data: FIFA 23 | Model: Llama 3.1 (Groq) | Built for IQM Assignment")
