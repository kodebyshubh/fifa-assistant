import re
import pandas as pd
from pathlib import Path
from rapidfuzz import process, fuzz

DATA_PATH = Path(__file__).parent / "data" / "players_23.csv"

_NUMERIC_COLS = ["overall", "potential", "pace", "shooting", "passing", "dribbling", "defending", "physic", "age", "value_eur", "wage_eur"]

_POSITION_MAP = {
    "goalkeeper": "GK", "goalie": "GK",
    "striker": "ST", "forward": "ST", "centre forward": "CF",
    "winger": "LW", "left winger": "LW", "right winger": "RW",
    "midfielder": "CM", "central midfielder": "CM", "attacking midfielder": "CAM",
    "defensive midfielder": "CDM",
    "left back": "LB", "right back": "RB", "centre back": "CB", "defender": "CB",
}


def _normalize_position(pos: str) -> str:
    if pos is None:
        return None
    return _POSITION_MAP.get(pos.lower().strip(), pos)
_DISPLAY_COLS = ["short_name", "age", "club_name", "nationality_name", "position", "overall", "potential", "value_eur"]
_DISPLAY_RENAME = {"short_name": "name", "club_name": "club", "nationality_name": "nationality"}


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, low_memory=False)
    # keep only latest update per player (FIFA 23 has multiple seasonal patches)
    df = df.sort_values("fifa_update").drop_duplicates(subset="player_id", keep="last")
    df["position"] = df["player_positions"].str.split(",").str[0].str.strip()
    for col in _NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


_data = load_data()

DATA_SUMMARY = {
    "total_players": len(_data),
    "total_clubs": int(_data["club_name"].nunique()),
    "overall_range": (int(_data["overall"].min()), int(_data["overall"].max())),
    "positions": sorted(_data["position"].dropna().unique().tolist()),
}


def top_n_players(n: int = 10, position: str = None, min_overall: int = None) -> pd.DataFrame:
    df = _data
    if position:
        df = df[df["player_positions"].str.contains(_normalize_position(position), case=False, na=False)]
    if min_overall:
        df = df[df["overall"] >= min_overall]
    return df.nlargest(n, "overall")[_DISPLAY_COLS].rename(columns=_DISPLAY_RENAME)


def _find_player(query: str):
    """Return (display_name, row) or None. Prefers whole-word match by highest overall."""
    pattern = r"\b" + re.escape(query) + r"\b"
    for col in ("short_name", "long_name"):
        mask = _data[col].str.contains(pattern, case=False, regex=True, na=False)
        if mask.any():
            row = _data[mask].nlargest(1, "overall").iloc[0]
            return row["short_name"], row
    # fuzzy fallback
    match = process.extractOne(query, _data["short_name"].tolist(), scorer=fuzz.WRatio)
    if match and match[1] >= 75:
        row = _data[_data["short_name"] == match[0]].iloc[0]
        return match[0], row
    return None


def compare_players(name1: str, name2: str) -> dict:
    found = []
    not_found = []
    rows = []

    for name in [name1, name2]:
        result = _find_player(name)
        if result:
            display, row = result
            found.append(display)
            rows.append(row)
        else:
            not_found.append(name)

    if len(rows) < 2:
        return {"found": found, "not_found": not_found, "comparison": None}

    attrs = ["overall", "potential", "pace", "shooting", "passing", "dribbling", "defending", "physic", "age", "value_eur", "wage_eur"]
    comparison = pd.DataFrame(
        {row["short_name"]: [row[a] for a in attrs] for row in rows},
        index=attrs,
    )
    return {"found": found, "not_found": not_found, "comparison": comparison}


def filter_players(
    max_age: int = None,
    min_pace: int = None,
    min_overall: int = None,
    position: str = None,
    club: str = None,
    nationality: str = None,
    min_potential: int = None,
) -> pd.DataFrame:
    df = _data
    if max_age:
        df = df[df["age"] <= max_age]
    if min_pace:
        df = df[df["pace"] >= min_pace]
    if min_overall:
        df = df[df["overall"] >= min_overall]
    if position:
        df = df[df["player_positions"].str.contains(_normalize_position(position), case=False, na=False)]
    if club:
        df = df[df["club_name"].str.contains(club, case=False, na=False)]
    if nationality:
        df = df[df["nationality_name"].str.contains(nationality, case=False, na=False)]
    if min_potential:
        df = df[df["potential"] >= min_potential]
    return df.nlargest(20, "overall")[_DISPLAY_COLS].rename(columns=_DISPLAY_RENAME)


def team_aggregation(top_n: int = 10) -> pd.DataFrame:
    grouped = (
        _data.groupby("club_name")
        .agg(avg_overall=("overall", "mean"), avg_potential=("potential", "mean"), player_count=("short_name", "count"), avg_value_eur=("value_eur", "mean"))
        .reset_index()
    )
    grouped = grouped[grouped["player_count"] >= 5]
    result = grouped.nlargest(top_n, "avg_overall").copy()
    result["avg_overall"] = result["avg_overall"].round(1)
    result["avg_potential"] = result["avg_potential"].round(1)
    result["avg_value_eur"] = result["avg_value_eur"].round(0)
    return result.rename(columns={"club_name": "club"})


def get_player_info(name: str) -> dict:
    result = _find_player(name)
    if result is None:
        return {"found": None, "not_found": name, "player": None}
    display, row = result
    attrs = ["overall", "potential", "pace", "shooting", "passing", "dribbling", "defending", "physic", "age", "value_eur", "wage_eur"]
    extra = ["club_name", "nationality_name", "position"]
    info = pd.DataFrame(
        {"value": [row[a] for a in attrs]},
        index=attrs,
    )
    meta = {col: row[col] for col in extra}
    return {"found": display, "not_found": None, "player": info, "meta": meta}


def best_value_players(top_n: int = 15, min_overall: int = 80) -> pd.DataFrame:
    df = _data[(_data["overall"] >= min_overall) & (_data["value_eur"] > 0)].copy()
    df["value_score"] = df["overall"] / (df["value_eur"] / 1_000_000 + 1)
    cols = ["short_name", "age", "club_name", "position", "overall", "value_eur", "wage_eur", "value_score"]
    result = df.nlargest(top_n, "value_score")[cols].copy()
    result["value_score"] = result["value_score"].round(2)
    return result.rename(columns={"short_name": "name", "club_name": "club"})
