# FIFA Data Assistant

A Streamlit app that lets you ask natural language questions about FIFA 23 player and team data. Combines LLM-based intent classification with deterministic Pandas queries to return structured, accurate results.

---

## What I Built

A hybrid AI + data pipeline:

1. **User asks a question** in plain English
2. **Llama 3.1** (via Groq) classifies the intent and extracts structured parameters (e.g. `{intent: "filter", position: "ST", min_pace: 85}`)
3. **Pandas** executes the query deterministically against the FIFA 23 dataset
4. **Llama 3.1** writes a 2–3 sentence insight about the results
5. **Streamlit** displays the result as a ranked table, comparison grid, or team summary

The data logic is fully deterministic — LLMs only handle natural language input and output, not data retrieval. This makes results reliable and auditable.

**Supported query types:**

| Type | Example |
|---|---|
| Top N ranking | "Show me the top 10 players by overall rating" |
| Player comparison | "Compare Messi and Ronaldo" |
| Filtered search | "Show me the best strikers with pace above 85" |
| Team aggregation | "Which teams have the highest average player rating?" |
| Value analysis | "Give me a short analysis of the best value players" |

---

## How to Run

**Requirements:** Python 3.9+, a free [Groq API key](https://console.groq.com)

```bash
# 1. Clone the repo
git clone <repo-url>
cd fifa-assistant

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your Groq API key
echo "GROQ_API_KEY=your_key_here" > .env

# 4. Download the dataset (see Dataset section below)
#    Place it at: data/players_23.csv

# 5. Run
streamlit run app.py
```

App opens at `http://localhost:8501`

---

## Dataset

**Source:** [FIFA 23 Complete Player Dataset](https://www.kaggle.com/datasets/stefanoleone992/fifa-23-complete-player-dataset) by stefanoleone992 on Kaggle

**Preparation:**
- Original download: `male_players.csv` (~5.6 GB, all FIFA editions 15–23 combined)
- Filtered to `fifa_version == 23` and deduplicated by keeping the latest seasonal patch per player
- Result: **20,621 unique male players**, 110 attributes

**Key attributes used:**
`short_name`, `age`, `nationality_name`, `club_name`, `player_positions`, `overall`, `potential`, `value_eur`, `wage_eur`, `pace`, `shooting`, `passing`, `dribbling`, `defending`, `physic`

> **Note:** `data/players_23.csv` is excluded from the repo (large file). Run the filter script or download and filter manually as described above.

---

## Where I Used AI / LLMs

**Model:** `llama-3.1-8b-instant` via Groq API

| Use | Where | Why not pure Python |
|---|---|---|
| Intent classification | `llm_handler.classify_intent()` | Handles arbitrary natural language phrasing — regex would miss too many variants |
| Param extraction | Same call | Extracts structured params (age, pace, position, player names) from free text |
| Result summarization | `llm_handler.generate_summary()` | Writes 2–3 sentence insight after each query — adds narrative context to raw data |

**Not used for:**
- Data retrieval (pure Pandas)
- Player matching (regex word-boundary + rapidfuzz)
- Filtering / ranking / aggregation (all deterministic Python)

---

## Known Limitations

- **Dataset is FIFA 23 season only** — ratings reflect the 2022–23 edition, not current real-world form
- **Male players only** — female player data was not included
- **Comparison is 2-player only** — "Compare Messi, Ronaldo and Neymar" compares only the first two
- **LLM intent may misclassify edge cases** — ambiguous queries like "best German players" may route to wrong intent
- **Best value formula** (`overall / (value_eur/1M + 1)`) favors cheap players — a 80-rated player at €1M scores higher than a 91-rated player at €50M, which is mathematically correct but counterintuitive
- **Groq API key required** — without it, the app errors on startup. App does not degrade gracefully if the key is missing at load time

---

## Architecture

```
User question
     │
     ▼
classify_intent()  ←── Llama 3.1 (Groq)
     │  returns: {intent, params, confidence}
     ▼
Router (app.py)
     │  routes to one of 5 Pandas functions
     ▼
data_engine.py  ←── FIFA 23 CSV (20,621 players)
     │  returns: DataFrame or compare dict
     ▼
generate_summary()  ←── Llama 3.1 (Groq)
     │  returns: 2-3 sentence insight
     ▼
Streamlit display
```

---

## Project Structure

```
fifa-assistant/
├── app.py              # Streamlit UI and routing
├── data_engine.py      # 5 Pandas query functions + player fuzzy match
├── llm_handler.py      # Groq API: intent classifier + summary generator
├── requirements.txt
├── .env                # GROQ_API_KEY (not committed)
└── data/
    └── players_23.csv  # FIFA 23 filtered dataset (not committed)
```
