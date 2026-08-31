
import os, math, json, time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import requests
import pandas as pd
import streamlit as st

# ============================================================
# EDGE — Sports Betting Analytics MVP
# Supports: MLB, PGA Tour, NFL, NCAA Football
#
# This is an MVP architecture:
#   Data -> Model probability -> Fair price -> Market price
#        -> EV -> Edge Score -> AI-ready explanation
#
# Live odds require THE_ODDS_API_KEY.
# Weather uses Open-Meteo and requires no API key.
# ============================================================

st.set_page_config(
    page_title="EDGE Sports Analyzer",
    page_icon="📈",
    layout="wide",
)

SPORTS = {
    "MLB": "baseball_mlb",
    "NFL": "americanfootball_nfl",
    "NCAA Football": "americanfootball_ncaaf",
    "PGA Tour": "golf_pga",
}

DEFAULTS = {
    "mlb": {"home_adv": 0.045, "spread": 0.11},
    "nfl": {"home_adv": 0.055, "spread": 0.13},
    "ncaaf": {"home_adv": 0.065, "spread": 0.18},
    "pga": {"home_adv": 0.0, "spread": 0.20},
}

# ----------------------------
# Odds helpers
# ----------------------------

def american_to_prob(odds: float) -> float:
    if odds is None:
        return 0.5
    odds = float(odds)
    if odds > 0:
        return 100 / (odds + 100)
    return (-odds) / ((-odds) + 100)

def american_to_decimal(odds: float) -> float:
    odds = float(odds)
    return 1 + (odds / 100 if odds > 0 else 100 / abs(odds))

def prob_to_american(p: float) -> int:
    p = min(max(float(p), 0.0001), 0.9999)
    if p >= 0.5:
        return round(-100 * p / (1 - p))
    return round(100 * (1 - p) / p)

def ev_per_unit(p: float, odds: float) -> float:
    d = american_to_decimal(odds)
    return p * (d - 1) - (1 - p)

def remove_vig_two_way(p1: float, p2: float) -> Tuple[float, float]:
    total = p1 + p2
    if total <= 0:
        return 0.5, 0.5
    return p1 / total, p2 / total

# ----------------------------
# API
# ----------------------------

def get_odds_api_key() -> str:
    key = os.getenv("THE_ODDS_API_KEY", "")
    try:
        key = st.secrets.get("THE_ODDS_API_KEY", key)
    except Exception:
        pass
    return key

@st.cache_data(ttl=90)
def fetch_odds(sport_key: str, regions="us", markets="h2h,spreads,totals"):
    key = get_odds_api_key()
    if not key:
        return [], "No THE_ODDS_API_KEY configured. Demo mode is active."
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    params = {
        "apiKey": key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": "american",
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return [], f"Odds API error: {e}"

# ----------------------------
# Weather
# ----------------------------

@st.cache_data(ttl=900)
def fetch_weather(lat: float, lon: float):
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m,precipitation_probability,wind_speed_10m,wind_direction_10m",
            "forecast_days": 2,
            "timezone": "auto",
        }
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)

def weather_impact(temp_f: float, wind_mph: float, precip_pct: float, sport: str) -> Dict:
    """
    Conservative first-pass weather adjustment.
    This is intentionally bounded; production models should learn
    sport/venue-specific effects from historical data.
    """
    impact = 0.0
    notes = []

    if sport in ("MLB",):
        if temp_f >= 85:
            impact += 0.12
            notes.append("warm air can modestly support offense")
        elif temp_f <= 45:
            impact -= 0.08
            notes.append("cold air can modestly suppress offense")
        if wind_mph >= 15:
            impact += 0.10
            notes.append("strong wind increases variance")
    elif sport in ("NFL", "NCAA Football"):
        if wind_mph >= 15:
            impact -= 0.05
            notes.append("wind can suppress passing/kicking efficiency")
        if precip_pct >= 50:
            impact -= 0.03
            notes.append("meaningful precipitation risk")
        if temp_f <= 35:
            impact -= 0.02
            notes.append("cold conditions")
    elif sport == "PGA Tour":
        if wind_mph >= 15:
            impact += 0.08
            notes.append("wind increases importance of ball-striking/trajectory")
        if precip_pct >= 50:
            impact += 0.04
            notes.append("weather increases variance and course-condition uncertainty")

    return {"impact": impact, "notes": notes}

# ----------------------------
# Model
# ----------------------------

def logistic(x: float) -> float:
    x = max(min(x, 30), -30)
    return 1 / (1 + math.exp(-x))

def model_game_probability(
    sport: str,
    market_home_prob: float,
    form_edge: float,
    matchup_edge: float,
    injury_edge: float,
    rest_edge: float,
    weather_edge: float,
) -> float:
    """
    MVP probability model.

    In production, these coefficients are replaced by trained,
    sport-specific models and calibrated against historical results.
    """
    base_logit = math.log(market_home_prob / (1 - market_home_prob))
    s = sport.lower()
    spread = DEFAULTS["pga" if "pga" in s else s]["spread"]
    adjustment = (form_edge + matchup_edge + injury_edge + rest_edge + weather_edge) / spread
    return logistic(base_logit + adjustment)

def edge_score(
    probability_edge: float,
    model_confidence: float,
    data_quality: float,
    market_quality: float,
    situational: float,
    stability: float,
    line_quality: float,
) -> int:
    """
    Score components are 0-100.
    Probability edge is expressed as percentage points.
    """
    edge_component = min(max(probability_edge * 5, 0), 100)
    score = (
        edge_component * 0.35
        + model_confidence * 0.20
        + data_quality * 0.10
        + market_quality * 0.10
        + situational * 0.10
        + stability * 0.10
        + line_quality * 0.05
    )
    return int(round(min(max(score, 0), 100)))

def confidence_from_inputs(data_quality: float, stability: float, sample_quality: float) -> int:
    return int(round(0.4 * data_quality + 0.35 * stability + 0.25 * sample_quality))

# ----------------------------
# Demo dataset
# ----------------------------

def demo_bets(sport: str) -> List[Dict]:
    if sport == "MLB":
        return [
            {
                "event": "Demo MLB — Home vs Away",
                "market": "Home ML",
                "book": "Demo",
                "odds": -118,
                "market_prob": american_to_prob(-118),
                "model_prob": 0.617,
                "weather": "Warm / moderate wind",
                "why": "Illustrative starter + lineup + bullpen edge",
            },
            {
                "event": "Demo MLB — Game Total",
                "market": "Under 8.5",
                "book": "Demo",
                "odds": -105,
                "market_prob": american_to_prob(-105),
                "model_prob": 0.575,
                "weather": "Cool / low wind",
                "why": "Illustrative run-environment edge",
            },
        ]
    if sport == "NFL":
        return [
            {
                "event": "Demo NFL — Home vs Away",
                "market": "Home +3.5",
                "book": "Demo",
                "odds": -110,
                "market_prob": american_to_prob(-110),
                "model_prob": 0.572,
                "weather": "Clear",
                "why": "Illustrative matchup and home-field edge",
            }
        ]
    if sport == "NCAA Football":
        return [
            {
                "event": "Demo NCAA — Ranked Home vs Away",
                "market": "Home -6.5",
                "book": "Demo",
                "odds": -110,
                "market_prob": american_to_prob(-110),
                "model_prob": 0.566,
                "weather": "Clear",
                "why": "Illustrative talent, matchup and home-field edge",
            }
        ]
    return [
        {
            "event": "Demo PGA — Tournament",
            "market": "Player A Top 20",
            "book": "Demo",
            "odds": -110,
            "market_prob": american_to_prob(-110),
            "model_prob": 0.595,
            "weather": "Windy",
            "why": "Illustrative course-fit and ball-striking edge",
        },
        {
            "event": "Demo PGA — Tournament",
            "market": "Player B Winner",
            "book": "Demo",
            "odds": 4500,
            "market_prob": american_to_prob(4500),
            "model_prob": 0.038,
            "weather": "Windy",
            "why": "Illustrative dark-horse course-fit edge",
        },
    ]

def analyze_bet(row: Dict, sport: str) -> Dict:
    mp = float(row["market_prob"])
    p = float(row["model_prob"])
    edge = (p - mp) * 100
    ev = ev_per_unit(p, float(row["odds"])) * 100
    fair = prob_to_american(p)

    confidence = confidence_from_inputs(88, 86, 84)
    score = edge_score(edge, confidence, 90, 88, 84, 86, min(100, 70 + max(edge, 0) * 4))

    if score >= 80 and edge >= 3:
        verdict = "BET"
    elif score >= 70 and edge >= 1.5:
        verdict = "LEAN"
    else:
        verdict = "PASS"

    return {
        **row,
        "model_prob": p,
        "market_prob": mp,
        "edge": edge,
        "ev": ev,
        "fair_odds": fair,
        "confidence": confidence,
        "edge_score": score,
        "verdict": verdict,
    }

# ----------------------------
# Odds API parsing
# ----------------------------

def parse_h2h(events: List[Dict]) -> List[Dict]:
    rows = []
    for event in events:
        home = event.get("home_team", "")
        away = event.get("away_team", "")
        for book in event.get("bookmakers", []):
            for market in book.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                outcomes = market.get("outcomes", [])
                for out in outcomes:
                    odds = out.get("price")
                    if odds is None:
                        continue
                    rows.append({
                        "event": f"{away} @ {home}",
                        "market": f"{out.get('name')} ML",
                        "book": book.get("title", "Book"),
                        "odds": odds,
                        "market_prob": american_to_prob(odds),
                        "model_prob": american_to_prob(odds),
                        "weather": "Not linked to venue yet",
                        "why": "Model awaiting sport-specific feature data",
                    })
                break
    return rows

# ----------------------------
# UI
# ----------------------------

st.title("📈 EDGE")
st.caption("Quantitative sports-betting analytics — probability first, price second.")

with st.sidebar:
    st.header("Controls")
    sport = st.selectbox("Sport", list(SPORTS.keys()))
    mode = st.radio("Data mode", ["Demo / Prototype", "Live Odds API"])
    min_edge = st.slider("Minimum model edge (%)", 0.0, 15.0, 3.0, 0.5)
    min_score = st.slider("Minimum Edge Score", 0, 100, 70, 5)
    st.divider()
    st.write("**Model philosophy**")
    st.write("Find positive expected value, not simply likely winners.")
    st.caption("This is an analytical prototype, not a guarantee of profit.")

if mode == "Live Odds API":
    events, err = fetch_odds(SPORTS[sport])
    if err:
        st.warning(err)
    rows = parse_h2h(events)
    if not rows:
        st.info("No live moneyline markets returned. Demo data is shown so the interface remains testable.")
        rows = demo_bets(sport)
else:
    rows = demo_bets(sport)

results = [analyze_bet(r, sport) for r in rows]
df = pd.DataFrame(results)
df = df[(df["edge"] >= min_edge) & (df["edge_score"] >= min_score)].sort_values(
    ["edge_score", "edge"], ascending=False
)

# Summary cards
c1, c2, c3, c4 = st.columns(4)
c1.metric("Opportunities", len(df))
c2.metric("Best Edge", f"{df['edge'].max():.1f}%" if len(df) else "—")
c3.metric("Best Score", f"{int(df['edge_score'].max())}" if len(df) else "—")
c4.metric("Avg Confidence", f"{df['confidence'].mean():.0f}" if len(df) else "—")

st.subheader("🔥 Today's strongest opportunities")

if df.empty:
    st.success("NO BET — nothing currently clears the selected thresholds.")
else:
    display = df[
        ["event", "market", "book", "odds", "model_prob", "market_prob", "edge", "fair_odds", "edge_score", "confidence", "verdict"]
    ].copy()
    display["model_prob"] = (display["model_prob"] * 100).round(1).astype(str) + "%"
    display["market_prob"] = (display["market_prob"] * 100).round(1).astype(str) + "%"
    display["edge"] = display["edge"].round(1).astype(str) + "%"
    display["fair_odds"] = display["fair_odds"].astype(int)
    display["edge_score"] = display["edge_score"].astype(int)
    display["confidence"] = display["confidence"].astype(int)
    st.dataframe(display, use_container_width=True, hide_index=True)

    st.divider()
    for _, r in df.iterrows():
        with st.container(border=True):
            a, b, c = st.columns([3, 1, 1])
            a.markdown(f"### {r['market']} — {r['event']}")
            b.metric("EDGE SCORE", int(r["edge_score"]))
            c.metric("Confidence", int(r["confidence"]))
            st.write(
                f"**Market:** {r['odds']:+g}  ·  "
                f"**Model:** {r['model_prob']*100:.1f}%  ·  "
                f"**Implied:** {r['market_prob']*100:.1f}%  ·  "
                f"**Edge:** {r['edge']:+.1f}%  ·  "
                f"**Fair price:** {r['fair_odds']:+d}  ·  "
                f"**EV:** {r['ev']:+.1f}%"
            )
            st.write(f"**Verdict: {r['verdict']}**")
            st.caption(f"Why: {r['why']} | Conditions: {r['weather']}")

# Parlay section
st.subheader("💰 Parlay Lab")
st.write("The production version will evaluate joint probability and correlation rather than simply stacking the highest Edge Scores.")

if len(df) >= 2:
    legs = df.head(3)
    joint_p = 1.0
    for _, r in legs.iterrows():
        joint_p *= float(r["model_prob"])
    # Illustrative independent-leg payout calculation
    combined_decimal = 1.0
    for _, r in legs.iterrows():
        combined_decimal *= american_to_decimal(float(r["odds"]))
    fair_parlay_odds = prob_to_american(joint_p)
    implied = 1 / combined_decimal
    parlay_edge = (joint_p - implied) * 100
    st.write(f"**Suggested legs:** {len(legs)}")
    for _, r in legs.iterrows():
        st.write(f"- {r['market']} ({r['odds']:+g}) — model {r['model_prob']*100:.1f}%")
    st.write(
        f"**Illustrative joint probability:** {joint_p*100:.1f}%  ·  "
        f"**Fair parlay price:** {fair_parlay_odds:+d}  ·  "
        f"**Model edge:** {parlay_edge:+.1f}%"
    )
else:
    st.info("Need at least two qualifying legs for a parlay preview.")

# PGA dark horse
if sport == "PGA Tour":
    st.subheader("🐎 Dark Horse Board")
    pga = pd.DataFrame(results)
    pga["dark_horse_score"] = (
        (pga["model_prob"] - pga["market_prob"]).clip(lower=0) * 100 * 25
        + pga["confidence"] * 0.45
        + pga["edge_score"] * 0.30
    ).clip(0, 100).round(0).astype(int)
    pga = pga.sort_values("dark_horse_score", ascending=False)
    st.dataframe(
        pga[["market", "odds", "model_prob", "market_prob", "edge", "dark_horse_score"]]
        .assign(
            model_prob=lambda x: (x.model_prob*100).round(1).astype(str)+"%",
            market_prob=lambda x: (x.market_prob*100).round(1).astype(str)+"%",
            edge=lambda x: x.edge.round(1).astype(str)+"%"
        ),
        use_container_width=True,
        hide_index=True,
    )

# Chat-style analyst
st.subheader("💬 AI Analyst")
question = st.text_input(
    "Ask the analyst",
    placeholder="What are the strongest bets today? / Who is the best PGA dark horse?"
)
if question:
    q = question.lower()
    if "dark horse" in q and sport == "PGA Tour":
        top = pd.DataFrame(results).sort_values("edge_score", ascending=False).iloc[0]
        st.success(
            f"Top prototype dark horse: **{top['market']}** at **{top['odds']:+g}**. "
            f"Model probability {top['model_prob']*100:.1f}% vs market {top['market_prob']*100:.1f}%, "
            f"Edge {top['edge']:+.1f}%, Edge Score {top['edge_score']}/100."
        )
    elif "parlay" in q:
        st.info("Use Parlay Lab above. Production v2 will optimize the legs jointly for probability, correlation, price and EV.")
    else:
        st.info("Prototype analyst: use the ranked opportunities above. In the production version, this box will call the model API and explain the quantitative evidence behind each recommendation.")

st.divider()
st.caption(
    "EDGE MVP v1.0 — Demo calculations are intentionally conservative placeholders. "
    "Do not treat demo outputs as live betting advice. A production model must be trained, calibrated, backtested, and validated out-of-sample."
)
