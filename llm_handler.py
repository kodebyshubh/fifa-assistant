import json
import os

import pandas as pd
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
_MODEL = "llama-3.1-8b-instant"

_INTENT_SYSTEM = """You are an intent classifier for a FIFA football data assistant. Classify the user question into one of these intents and extract parameters. Respond ONLY with valid JSON, no explanation.

Intents:
- top_n: user wants the TOP N highest-rated players with NO extra conditions. Example: "top 10 players", "best 5 midfielders". Params: n=int default 10, position=str or null (use FIFA position codes: GK/ST/CF/CAM/CM/CDM/LW/RW/LB/RB/CB/LM/RM/LWB/RWB), min_overall=int or null.
- compare: user wants to compare 2+ specific named players. Example: "compare Messi and Ronaldo". Params: players=list of player name strings.
- filter: user wants players matching CONDITIONS or THRESHOLDS like pace, age, nationality, club, potential. Example: "strikers with pace above 85", "German strikers", "young players under 23", "best players at Real Madrid". Params: max_age=int or null, min_pace=int or null, min_overall=int or null, position=str or null (FIFA codes), club=str or null, nationality=str or null, min_potential=int or null.
- team_stats: user wants team-level rankings or averages. Example: "teams with highest average rating". Params: top_n=int default 10.
- best_value: user wants value-for-money analysis. Example: "best value players", "underrated players". Params: top_n=int default 15, min_overall=int default 80.
- unknown: cannot map to above.

IMPORTANT: Use "filter" when the question has any condition beyond just ranking (pace threshold, nationality, club, age limit). Use "top_n" ONLY for pure ranking with no extra filter conditions.

Response JSON format:
{"intent": "<intent_name>", "params": {}, "confidence": "high|medium|low"}"""

_SUMMARY_SYSTEM = "You are a concise football data analyst. Given a user question and data result, write 2-3 sentences of insight. Be specific with numbers. No fluff."


def classify_intent(user_question: str) -> dict:
    try:
        response = _client.chat.completions.create(
            model=_MODEL,
            messages=[
                {"role": "system", "content": _INTENT_SYSTEM},
                {"role": "user", "content": user_question},
            ],
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)
    except Exception:
        return {"intent": "unknown", "params": {}, "confidence": "low"}


def generate_summary(intent: str, params: dict, result, user_question: str) -> str:
    try:
        if isinstance(result, pd.DataFrame):
            result_preview = result.head(5).to_string()
        else:
            result_preview = str(result)

        user_msg = f"Question: {user_question}\nIntent: {intent}\nParams used: {params}\nResult preview:\n{result_preview}\nWrite a brief insight."
        response = _client.chat.completions.create(
            model=_MODEL,
            messages=[
                {"role": "system", "content": _SUMMARY_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
        )
        return response.choices[0].message.content
    except Exception:
        return ""
