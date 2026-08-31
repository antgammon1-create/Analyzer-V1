
import os, math, random, json
from datetime import date, datetime, timedelta
from collections import defaultdict

import requests
import pandas as pd
import numpy as np
import streamlit as st

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, log_loss
except Exception:
    LogisticRegression = None
    brier_score_loss = None
    log_loss = None

st.set_page_config(page_title="EDGE v3", page_icon="📈", layout="wide")

MLB = "https://statsapi.mlb.com/api/v1"
ODDS_SPORT = "baseball_mlb"
WEATHER = "https://api.open-meteo.com/v1/forecast"
SEASON = date.today().year

# ============================================================
# Core helpers
# ============================================================

def secret(name):
    value = os.getenv(name, "")
    try:
        value = st.secrets.get(name, value)
    except Exception:
        pass
    return value

@st.cache_data(ttl=300)
def get_json(url, params=None):
    r = requests.get(url, params=params or {}, timeout=25)
    r.raise_for_status()
    return r.json()

def implied_prob(odds):
    odds = float(odds)
    return 100 / (odds + 100) if odds >= 0 else -odds / (-odds + 100)

def american_decimal(odds):
    odds = float(odds)
    return 1 + (odds / 100 if odds >= 0 else 100 / abs(odds))

def fair_price(p):
    p = max(.0001, min(.9999, float(p)))
    return round(-100 * p / (1 - p)) if p >= .5 else round(100 * (1 - p) / p)

def ev_pct(p, odds):
    return (p * american_decimal(odds) - 1) * 100

def no_vig(p1, p2):
    s = p1 + p2
    if s <= 0:
        return .5, .5
    return p1 / s, p2 / s

def norm(s):
    return "".join(c.lower() for c in (s or "") if c.isalnum())

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

# ============================================================
# MLB schedule / outcomes
# ============================================================

@st.cache_data(ttl=300)
def get_schedule(d):
    return get_json(
        f"{MLB}/schedule",
        {"sportId": 1, "date": d, "hydrate": "probablePitcher,team,venue"},
    ).get("dates", [])

def games_for(d):
    out = []
    for day in get_schedule(d):
        for g in day.get("games", []):
            h = g.get("teams", {}).get("home", {})
            a = g.get("teams", {}).get("away", {})
            out.append({
                "gamePk": g.get("gamePk"),
                "gameDate": g.get("gameDate"),
                "status": g.get("status", {}).get("detailedState"),
                "home": h.get("team", {}).get("name", "Home"),
                "home_id": h.get("team", {}).get("id"),
                "home_score": h.get("score"),
                "away": a.get("team", {}).get("name", "Away"),
                "away_id": a.get("team", {}).get("id"),
                "away_score": a.get("score"),
                "hp": (h.get("probablePitcher") or {}).get("fullName"),
                "hp_id": (h.get("probablePitcher") or {}).get("id"),
                "ap": (a.get("probablePitcher") or {}).get("fullName"),
                "ap_id": (a.get("probablePitcher") or {}).get("id"),
                "venue_id": (g.get("venue") or {}).get("id"),
            })
    return out

# ============================================================
# Leakage-aware stats
# ============================================================

def prev_day(d):
    return (datetime.fromisoformat(d) - timedelta(days=1)).date().isoformat()

def season_start_for(d):
    yr = datetime.fromisoformat(d).year
    return f"{yr}-03-20"

@st.cache_data(ttl=1200)
def pitcher_stats_range(pid, start_date, end_date):
    if not pid:
        return {}
    try:
        data = get_json(
            f"{MLB}/people/{pid}/stats",
            {
                "stats": "byDateRange",
                "group": "pitching",
                "startDate": start_date,
                "endDate": end_date,
            },
        )
        splits = data.get("stats", [{}])[0].get("splits") or []
        s = splits[0].get("stat", {}) if splits else {}
    except Exception:
        s = {}
    return {
        "era": float(s.get("era", 4.30) or 4.30),
        "whip": float(s.get("whip", 1.30) or 1.30),
        "k9": float(s.get("strikeoutsPer9Inn", 8.5) or 8.5),
        "bb9": float(s.get("walksPer9Inn", 3.2) or 3.2),
        "hr9": float(s.get("homeRunsPer9", 1.2) or 1.2),
        "innings": float(s.get("inningsPitched", 0) or 0),
    }

@st.cache_data(ttl=1200)
def team_stats_range(tid, group, start_date, end_date):
    if not tid:
        return {}
    try:
        data = get_json(
            f"{MLB}/teams/{tid}/stats",
            {
                "stats": "byDateRange",
                "group": group,
                "startDate": start_date,
                "endDate": end_date,
                "sportIds": 1,
            },
        )
        splits = data.get("stats", [{}])[0].get("splits") or []
        s = splits[0].get("stat", {}) if splits else {}
    except Exception:
        s = {}
    if group == "hitting":
        return {
            "avg": float(s.get("avg", .240) or .240),
            "obp": float(s.get("obp", .310) or .310),
            "slg": float(s.get("slg", .400) or .400),
            "ops": float(s.get("ops", .710) or .710),
        }
    return {
        "era": float(s.get("era", 4.20) or 4.20),
        "whip": float(s.get("whip", 1.30) or 1.30),
        "k9": float(s.get("strikeoutsPer9Inn", 8.5) or 8.5),
    }

# ============================================================
# Venue + weather
# ============================================================

@st.cache_data(ttl=900)
def venue_info(vid):
    if not vid:
        return {}
    data = get_json(f"{MLB}/venues/{vid}", {"hydrate": "location"})
    venues = data.get("venues", [])
    if not venues:
        return {}
    v = venues[0]
    loc = v.get("location", {})
    coords = loc.get("defaultCoordinates", {})
    return {
        "name": v.get("name", ""),
        "city": loc.get("city", ""),
        "state": loc.get("stateAbbrev", ""),
        "lat": coords.get("latitude"),
        "lon": coords.get("longitude"),
    }

PARK_FALLBACKS = {
    "Globe Life Field": (32.7473, -97.0845),
    "Truist Park": (33.8908, -84.4677),
    "Oracle Park": (37.7786, -122.3893),
    "Wrigley Field": (41.9484, -87.6553),
    "Fenway Park": (42.3467, -71.0972),
    "Yankee Stadium": (40.8296, -73.9262),
    "Dodger Stadium": (34.0739, -118.2400),
    "Coors Field": (39.7559, -104.9942),
}

def validated_venue(v):
    if not v:
        return {}, False, "missing venue"
    lat, lon = v.get("lat"), v.get("lon")
    valid = (
        isinstance(lat, (int, float))
        and isinstance(lon, (int, float))
        and 20 <= float(lat) <= 50
        and -130 <= float(lon) <= -65
    )
    source = "MLB coordinates"
    if not valid and v.get("name") in PARK_FALLBACKS:
        lat, lon = PARK_FALLBACKS[v["name"]]
        source = "fallback park coordinates"
        valid = True
    return {**v, "lat": lat, "lon": lon}, valid, source

@st.cache_data(ttl=900)
def weather_forecast(lat, lon):
    return get_json(
        WEATHER,
        {
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m,precipitation_probability,wind_speed_10m,wind_direction_10m",
            "forecast_days": 3,
            "timezone": "auto",
        },
    )

def first_pitch_weather(game_date_utc, venue):
    v, coord_ok, coord_source = validated_venue(venue)
    if not coord_ok or not game_date_utc:
        return 0.0, "Weather unavailable", 35, {}

    w = weather_forecast(v["lat"], v["lon"])
    h = w.get("hourly", {})
    times = h.get("time", [])
    temps_c = h.get("temperature_2m", [])
    winds = h.get("wind_speed_10m", [])
    rains = h.get("precipitation_probability", [])
    if not times or not temps_c:
        return 0.0, "Weather unavailable", 35, {}

    dt_utc = datetime.fromisoformat(game_date_utc.replace("Z", "+00:00"))
    offset = int(w.get("utc_offset_seconds", 0) or 0)
    target_local = (dt_utc + timedelta(seconds=offset)).replace(tzinfo=None)

    parsed = [datetime.fromisoformat(t) for t in times]
    idx = min(range(len(parsed)), key=lambda i: abs((parsed[i] - target_local).total_seconds()))

    temp_f = float(temps_c[idx]) * 9 / 5 + 32
    wind = float(winds[idx]) if idx < len(winds) else 0.0
    rain = float(rains[idx]) if idx < len(rains) else 0.0

    hour_delta = abs((parsed[idx] - target_local).total_seconds()) / 3600
    plausible_temp = -10 <= temp_f <= 115
    seasonal_ok = not (
        target_local.month in (6, 7, 8, 9)
        and float(v["lat"]) < 36
        and temp_f < 50
    )
    trusted = plausible_temp and hour_delta <= 1.5 and seasonal_ok

    quality = 90
    if coord_source != "MLB coordinates":
        quality -= 5
    if not plausible_temp:
        quality -= 40
    if hour_delta > 1.5:
        quality -= 20
    if not seasonal_ok:
        quality -= 35

    adj = 0.0
    if trusted:
        if temp_f >= 85:
            adj += .10
        elif temp_f <= 45:
            adj -= .08
        if wind >= 15:
            adj += .04
        if rain >= 60:
            adj -= .05

    note = f"{temp_f:.0f}°F · {wind:.0f} mph wind · {rain:.0f}% rain at first pitch"
    if not trusted:
        note += " · ⚠️ ignored by model"

    return adj, note, clamp(quality, 20, 95), {
        "trusted": trusted,
        "coord_source": coord_source,
        "target_local": str(target_local),
        "matched_hour": str(parsed[idx]),
    }

# ============================================================
# Lineups
# ============================================================

@st.cache_data(ttl=120)
def boxscore(game_pk):
    try:
        return get_json(f"{MLB}/game/{game_pk}/boxscore")
    except Exception:
        return {}

def lineup_status(game_pk):
    data = boxscore(game_pk)
    teams = data.get("teams", {})
    out = {}
    for side in ("home", "away"):
        players = teams.get(side, {}).get("players", {})
        starters = []
        for p in players.values():
            bo = p.get("battingOrder")
            if bo:
                starters.append((int(bo), p.get("person", {}).get("fullName", "")))
        starters.sort()
        out[side] = [name for _, name in starters[:9]]
    confirmed = len(out.get("home", [])) >= 9 and len(out.get("away", [])) >= 9
    return confirmed, out

# ============================================================
# Model
# ============================================================

def pitcher_quality(s):
    if not s:
        return 0
    x = (
        (4.30 - s["era"]) * .16
        + (1.30 - s["whip"]) * .65
        + (s["k9"] - 8.5) * .035
        + (3.2 - s["bb9"]) * .035
        + (1.20 - s["hr9"]) * .16
    )
    shrink = min(1.0, max(0, s.get("innings", 0)) / 80.0)
    return clamp(x * shrink, -1.25, 1.25)

def offense_quality(s):
    if not s:
        return 0
    return clamp(
        (s["ops"] - .710) * 2.4
        + (s["obp"] - .310) * 1.4
        + (s["slg"] - .400) * 1.2,
        -1.25, 1.25,
    )

def bullpen_proxy(s):
    if not s:
        return 0
    return clamp(
        (4.20 - s["era"]) * .10
        + (1.30 - s["whip"]) * .35
        + (s["k9"] - 8.5) * .02,
        -.60, .60,
    )

def project_game(g, asof_date=None, include_weather=True):
    d = asof_date or date.today().isoformat()
    start = season_start_for(d)
    end = prev_day(d)

    hp = pitcher_stats_range(g["hp_id"], start, end)
    ap = pitcher_stats_range(g["ap_id"], start, end)
    ho = team_stats_range(g["home_id"], "hitting", start, end)
    ao = team_stats_range(g["away_id"], "hitting", start, end)
    htp = team_stats_range(g["home_id"], "pitching", start, end)
    atp = team_stats_range(g["away_id"], "pitching", start, end)

    venue = venue_info(g["venue_id"])
    if include_weather:
        weather_adj, weather_note, weather_q, weather_debug = first_pitch_weather(g["gameDate"], venue)
    else:
        weather_adj, weather_note, weather_q, weather_debug = 0.0, "Weather omitted in backtest", 70, {}

    confirmed, lineups = lineup_status(g["gamePk"])
    lineup_q = 95 if confirmed else 58

    hpq, apq = pitcher_quality(hp), pitcher_quality(ap)
    hoq, aoq = offense_quality(ho), offense_quality(ao)
    hbp, abp = bullpen_proxy(htp), bullpen_proxy(atp)

    home_components = {
        "baseline": 4.45,
        "home_field": .20,
        "offense": .58 * hoq,
        "opp_starter": -.72 * apq,
        "opp_bullpen": -.18 * abp,
        "own_starter_context": .05 * hpq,
        "weather": weather_adj / 2,
    }
    away_components = {
        "baseline": 4.45,
        "offense": .58 * aoq,
        "opp_starter": -.72 * hpq,
        "opp_bullpen": -.18 * hbp,
        "own_starter_context": .05 * apq,
        "weather": weather_adj / 2,
    }

    home = clamp(sum(home_components.values()), 1.5, 8.5)
    away = clamp(sum(away_components.values()), 1.5, 8.5)

    starter_q = 90 if g["hp_id"] and g["ap_id"] else 55
    data_quality = round(.35 * starter_q + .25 * lineup_q + .20 * weather_q + .20 * 78)

    return home, away, {
        "weather": weather_note,
        "lineups_confirmed": confirmed,
        "data_quality": data_quality,
        "home_components": home_components,
        "away_components": away_components,
        "weather_debug": weather_debug,
    }

def poisson(rng, lam):
    L = math.exp(-lam)
    k, p = 0, 1.0
    while p > L:
        k += 1
        p *= rng.random()
    return k - 1

def simulate_scores(home_runs, away_runs, n, seed):
    rng = random.Random(seed)
    return [(poisson(rng, home_runs), poisson(rng, away_runs)) for _ in range(n)]

def model_moneyline(scores):
    n = len(scores)
    home_w = sum(h > a for h, a in scores) / n
    away_w = sum(a > h for h, a in scores) / n
    ties = max(0.0, 1 - home_w - away_w)
    return home_w + ties / 2, away_w + ties / 2

def model_total(scores, side, point):
    n = len(scores)
    if side.lower() == "over":
        wins = sum(h + a > point for h, a in scores)
        pushes = sum(h + a == point for h, a in scores)
    else:
        wins = sum(h + a < point for h, a in scores)
        pushes = sum(h + a == point for h, a in scores)
    return wins / max(1, n - pushes)

def model_spread(scores, team_is_home, point):
    n = len(scores)
    wins = pushes = 0
    for h, a in scores:
        margin = h - a if team_is_home else a - h
        result = margin + point
        wins += result > 0
        pushes += result == 0
    return wins / max(1, n - pushes)

# ============================================================
# Odds API
# ============================================================

@st.cache_data(ttl=90)
def odds_feed():
    key = secret("THE_ODDS_API_KEY")
    if not key:
        return [], "THE_ODDS_API_KEY is missing from Streamlit Secrets."
    try:
        r = requests.get(
            f"https://api.the-odds-api.com/v4/sports/{ODDS_SPORT}/odds",
            params={
                "apiKey": key,
                "regions": "us",
                "markets": "h2h,spreads,totals",
                "oddsFormat": "american",
            },
            timeout=20,
        )
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return [], f"Odds API error: {e}"

@st.cache_data(ttl=3600)
def historical_odds(snapshot_iso):
    key = secret("THE_ODDS_API_KEY")
    if not key:
        return [], "Missing key"
    try:
        r = requests.get(
            f"https://api.the-odds-api.com/v4/historical/sports/{ODDS_SPORT}/odds",
            params={
                "apiKey": key,
                "regions": "us",
                "markets": "h2h",
                "oddsFormat": "american",
                "date": snapshot_iso,
            },
            timeout=25,
        )
        r.raise_for_status()
        data = r.json()
        # Historical endpoint commonly nests events in "data".
        return data.get("data", data if isinstance(data, list) else []), None
    except Exception as e:
        return [], str(e)

def moneyline_market(event):
    book_probs = defaultdict(list)
    offers = defaultdict(list)
    for book in event.get("bookmakers", []):
        outcomes = None
        for m in book.get("markets", []):
            if m.get("key") == "h2h":
                outcomes = m.get("outcomes", [])
                break
        if not outcomes or len(outcomes) < 2:
            continue
        pair = [o for o in outcomes if o.get("price") is not None][:2]
        if len(pair) < 2:
            continue
        nv = no_vig(implied_prob(pair[0]["price"]), implied_prob(pair[1]["price"]))
        for o, p in zip(pair, nv):
            team = o.get("name")
            book_probs[team].append(p)
            offers[team].append((float(o["price"]), book.get("title", "Book")))
    out = {}
    for team, probs in book_probs.items():
        best = min(offers[team], key=lambda x: implied_prob(x[0]))
        out[team] = {
            "consensus_prob": sum(probs) / len(probs),
            "best_odds": best[0],
            "best_book": best[1],
            "books": len(probs),
        }
    return out

def raw_markets(event, key):
    rows = []
    for book in event.get("bookmakers", []):
        for market in book.get("markets", []):
            if market.get("key") != key:
                continue
            for o in market.get("outcomes", []):
                if o.get("price") is None:
                    continue
                rows.append({
                    "name": o.get("name"),
                    "point": o.get("point"),
                    "odds": float(o.get("price")),
                    "book": book.get("title", "Book"),
                })
    return rows

def group_two_way_consensus(rows, kind):
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["book"], r["point"])].append(r)
    consensus = defaultdict(list)
    offers = defaultdict(list)

    for (book, point), pair in grouped.items():
        if len(pair) < 2:
            continue
        if kind == "totals":
            names = {p["name"].lower(): p for p in pair}
            if "over" not in names or "under" not in names:
                continue
            a, b = names["over"], names["under"]
            nva, nvb = no_vig(implied_prob(a["odds"]), implied_prob(b["odds"]))
            for key, item, prob in [(("Over", point), a, nva), (("Under", point), b, nvb)]:
                consensus[key].append(prob)
                offers[key].append((item["odds"], book))
        else:
            if len(pair) != 2:
                continue
            a, b = pair
            nva, nvb = no_vig(implied_prob(a["odds"]), implied_prob(b["odds"]))
            for key, item, prob in [((a["name"], a["point"]), a, nva), ((b["name"], b["point"]), b, nvb)]:
                consensus[key].append(prob)
                offers[key].append((item["odds"], book))

    out = {}
    for k, probs in consensus.items():
        best = min(offers[k], key=lambda x: implied_prob(x[0]))
        out[k] = {
            "consensus_prob": sum(probs) / len(probs),
            "best_odds": best[0],
            "best_book": best[1],
            "books": len(probs),
        }
    return out

# ============================================================
# Reliability / scoring / calibration
# ============================================================

def disagreement_penalty(edge_pp, data_quality, books):
    abs_edge = abs(edge_pp)
    penalty = 0
    if abs_edge > 8:
        penalty += min(25, (abs_edge - 8) * 1.5)
    if books >= 6 and abs_edge > 10:
        penalty += 8
    if data_quality < 75 and abs_edge > 6:
        penalty += 8
    return penalty

def edge_score(edge_pp, data_quality, market_books, stability, market_type):
    edge_component = min(100, max(0, edge_pp * 4.2))
    market_q = min(100, 55 + market_books * 6)
    type_q = 86 if market_type == "ML" else 80 if market_type == "TOTAL" else 77
    raw = (
        edge_component * .28
        + data_quality * .30
        + stability * .18
        + market_q * .10
        + type_q * .14
    )
    raw -= disagreement_penalty(edge_pp, data_quality, market_books)
    return int(round(clamp(raw, 0, 100)))

def reliability_label(edge_pp, data_quality, books, score):
    if data_quality < 65:
        return "LOW"
    if abs(edge_pp) > 12 and books >= 6:
        return "CAUTION"
    if score >= 80 and data_quality >= 80:
        return "HIGH"
    if score >= 68:
        return "MEDIUM"
    return "LOW"

def apply_calibration(p):
    cal = st.session_state.get("calibration")
    if not cal:
        return p
    a = cal["coef"]
    b = cal["intercept"]
    x = math.log(clamp(p, .001, .999) / (1 - clamp(p, .001, .999)))
    z = a * x + b
    return 1 / (1 + math.exp(-z))

def fractional_kelly(p, odds, fraction=.25):
    dec = american_decimal(odds)
    b = dec - 1
    q = 1 - p
    full = (b * p - q) / b if b > 0 else 0
    return max(0.0, full * fraction)

# ============================================================
# Live analysis builder
# ============================================================

def build_live_board(game_date, sims, markets):
    games = games_for(game_date.isoformat())
    events, odds_error = odds_feed()
    lookup = {(norm(e.get("away_team")), norm(e.get("home_team"))): e for e in events}
    rows = []

    for g in games:
        event = lookup.get((norm(g["away"]), norm(g["home"])))
        if not event:
            continue
        try:
            hr, ar, meta = project_game(g, game_date.isoformat(), include_weather=True)
            scores = simulate_scores(hr, ar, sims, int(g["gamePk"] or 1))
            home_p, away_p = model_moneyline(scores)
        except Exception:
            continue

        def add_row(market_type, selection, market, model_p):
            raw_model_p = model_p
            model_p = apply_calibration(model_p)
            edge = (model_p - market["consensus_prob"]) * 100
            stability = 78 if market_type == "ML" else 73 if market_type == "TOTAL" else 70
            sc = edge_score(edge, meta["data_quality"], market["books"], stability, market_type)
            rows.append({
                "game": f'{g["away"]} @ {g["home"]}',
                "market_type": market_type,
                "selection": selection,
                "book": market["best_book"],
                "odds": market["best_odds"],
                "raw_model_prob": raw_model_p,
                "model_prob": model_p,
                "market_prob": market["consensus_prob"],
                "edge": edge,
                "fair": fair_price(model_p),
                "ev": ev_pct(model_p, market["best_odds"]),
                "score": sc,
                "reliability": reliability_label(edge, meta["data_quality"], market["books"], sc),
                "books": market["books"],
                "proj_score": f"{g['away']} {ar:.1f} – {g['home']} {hr:.1f}",
                "pitchers": f'{g["ap"] or "TBD"} / {g["hp"] or "TBD"}',
                "weather": meta["weather"],
                "lineup": "Confirmed" if meta["lineups_confirmed"] else "Projected/TBD",
                "data_quality": meta["data_quality"],
                "home_components": meta["home_components"],
                "away_components": meta["away_components"],
            })

        if "Moneyline" in markets:
            for team, market in moneyline_market(event).items():
                p = home_p if norm(team) == norm(g["home"]) else away_p
                add_row("ML", f"{team} ML", market, p)

        if "Totals" in markets:
            for (side, point), market in group_two_way_consensus(raw_markets(event, "totals"), "totals").items():
                add_row("TOTAL", f"{side} {float(point):g}", market, model_total(scores, side, float(point)))

        if "Run line" in markets:
            for (team, point), market in group_two_way_consensus(raw_markets(event, "spreads"), "spreads").items():
                add_row(
                    "RL",
                    f"{team} {float(point):+g}",
                    market,
                    model_spread(scores, norm(team) == norm(g["home"]), float(point)),
                )

    return pd.DataFrame(rows), games, events, odds_error

# ============================================================
# Historical backtest
# ============================================================

def backtest_dates(start_date, end_date, sims=3000, use_hist_odds=False):
    records = []
    d = start_date
    total_days = (end_date - start_date).days + 1
    progress = st.progress(0)
    status = st.empty()

    for idx in range(total_days):
        cur = d + timedelta(days=idx)
        ds = cur.isoformat()
        status.write(f"Backtesting {ds} ({idx+1}/{total_days})")
        try:
            gs = games_for(ds)
        except Exception:
            gs = []

        hist_lookup = {}
        if use_hist_odds:
            snap = f"{ds}T16:00:00Z"
            events, _ = historical_odds(snap)
            hist_lookup = {(norm(e.get("away_team")), norm(e.get("home_team"))): e for e in events}

        for g in gs:
            if g.get("home_score") is None or g.get("away_score") is None:
                continue
            try:
                hr, ar, meta = project_game(g, ds, include_weather=False)
                scores = simulate_scores(hr, ar, sims, int(g["gamePk"] or 1))
                home_p, away_p = model_moneyline(scores)
            except Exception:
                continue

            y_home = 1 if g["home_score"] > g["away_score"] else 0
            records.append({
                "date": ds,
                "game": f'{g["away"]} @ {g["home"]}',
                "team": g["home"],
                "side": "home",
                "model_prob": home_p,
                "outcome": y_home,
                "data_quality": meta["data_quality"],
            })
            records.append({
                "date": ds,
                "game": f'{g["away"]} @ {g["home"]}',
                "team": g["away"],
                "side": "away",
                "model_prob": away_p,
                "outcome": 1-y_home,
                "data_quality": meta["data_quality"],
            })

            if use_hist_odds:
                e = hist_lookup.get((norm(g["away"]), norm(g["home"])))
                if e:
                    ml = moneyline_market(e)
                    for team, m in ml.items():
                        for rec in records[-2:]:
                            if norm(rec["team"]) == norm(team):
                                rec["market_prob"] = m["consensus_prob"]
                                rec["odds"] = m["best_odds"]
                                rec["edge"] = (rec["model_prob"] - m["consensus_prob"]) * 100
                                rec["bet_return"] = (
                                    american_decimal(m["best_odds"]) - 1
                                    if rec["outcome"] == 1 else -1
                                )

        progress.progress((idx+1)/total_days)

    status.empty()
    progress.empty()
    return pd.DataFrame(records)

# ============================================================
# UI
# ============================================================

st.title("📈 EDGE v3 — MLB Research Platform")
st.caption("Live analysis · no-vig markets · calibrated probabilities · historical backtesting · ROI/CLV-ready exports")

tab_live, tab_backtest, tab_cal, tab_track, tab_roadmap = st.tabs(
    ["🔥 Today", "🧪 Backtest", "🎯 Calibration", "📒 Tracking", "🧭 Other Sports"]
)

with tab_live:
    c1, c2, c3 = st.columns(3)
    game_date = c1.date_input("Game date", date.today(), key="live_date")
    sims = c2.selectbox("Simulations/game", [5000, 10000, 25000, 50000], index=2)
    markets = c3.multiselect("Markets", ["Moneyline", "Totals", "Run line"], default=["Moneyline", "Totals", "Run line"])

    min_edge = st.slider("Minimum edge", 0.0, 15.0, 2.5, .5)
    min_score = st.slider("Minimum Edge Score", 0, 100, 65, 5)

    with st.spinner("Building live MLB board..."):
        df, games, events, odds_error = build_live_board(game_date, sims, markets)

    if odds_error:
        st.warning(odds_error)

    candidates = (
        df[(df.edge >= min_edge) & (df.score >= min_score)].sort_values(["score", "edge"], ascending=False)
        if not df.empty else pd.DataFrame()
    )

    a,b,c,d = st.columns(4)
    a.metric("Games", len(games))
    b.metric("Markets", len(df))
    c.metric("Best edge", f"{candidates.edge.max():.1f}%" if not candidates.empty else "—")
    d.metric("Best score", int(candidates.score.max()) if not candidates.empty else "—")

    if st.session_state.get("calibration"):
        st.success("Calibration is ACTIVE on live model probabilities.")
    else:
        st.warning("Calibration is NOT active. Live probabilities are raw model outputs.")

    st.subheader("Strongest current signals")
    if candidates.empty:
        st.success("NO BET — no market clears the current thresholds.")
    else:
        for _, r in candidates.head(12).iterrows():
            with st.container(border=True):
                x,y,z = st.columns([4,1,1])
                x.markdown(f"### {r.selection} — {r.game}")
                y.metric("EDGE", f"{r.edge:+.1f}%")
                z.metric("SCORE", int(r.score))
                st.write(
                    f"**Best price:** {r.odds:+g} at {r.book} · "
                    f"**Model:** {r.model_prob*100:.1f}% · "
                    f"**No-vig market:** {r.market_prob*100:.1f}% · "
                    f"**Fair:** {r.fair:+d} · **EV:** {r.ev:+.1f}%"
                )
                st.write(
                    f"**Projection:** {r.proj_score} · **Pitchers:** {r.pitchers} · "
                    f"**Weather:** {r.weather}"
                )
                st.write(
                    f"**Reliability:** {r.reliability} · **Data quality:** {int(r.data_quality)}/100 · "
                    f"**Books:** {int(r.books)} · **Lineups:** {r.lineup}"
                )

                kelly = fractional_kelly(r.model_prob, r.odds, .25)
                st.caption(f"¼-Kelly reference: {kelly*100:.2f}% of bankroll before user-defined caps. Use only after validation.")

                with st.expander("Why this projection?"):
                    st.write("Away run components")
                    st.json({k: round(v,3) for k,v in r.away_components.items()})
                    st.write("Home run components")
                    st.json({k: round(v,3) for k,v in r.home_components.items()})

    st.subheader("Full market board")
    if not df.empty:
        board = df.copy()
        board["model_prob"] = (board.model_prob*100).round(1).astype(str)+"%"
        board["market_prob"] = (board.market_prob*100).round(1).astype(str)+"%"
        board["edge"] = board.edge.round(1).astype(str)+"%"
        board["ev"] = board.ev.round(1).astype(str)+"%"
        st.dataframe(
            board[
                ["game","market_type","selection","book","odds","model_prob","market_prob","edge",
                 "fair","ev","score","reliability","books","proj_score","lineup","weather"]
            ],
            use_container_width=True, hide_index=True
        )
        st.download_button(
            "Download live board CSV",
            data=df.to_csv(index=False).encode(),
            file_name=f"EDGE_live_{game_date.isoformat()}.csv",
            mime="text/csv",
        )

with tab_backtest:
    st.subheader("Historical prediction backtest")
    st.write("This uses only information available before each game date. Weather is omitted in backtests to avoid accidental look-ahead from forecast data.")

    c1,c2,c3 = st.columns(3)
    start = c1.date_input("Start date", date.today()-timedelta(days=14), key="bt_start")
    end = c2.date_input("End date", date.today()-timedelta(days=1), key="bt_end")
    bt_sims = c3.selectbox("Simulations/game", [1000, 3000, 5000], index=1, key="bt_sims")

    days = (end-start).days+1
    if days > 45:
        st.warning("For phone/Streamlit reliability, v3 limits one backtest run to 45 days. Run multiple windows and combine exports.")
    use_hist_odds = st.checkbox(
        "Attempt historical odds / ROI (requires The Odds API plan with historical endpoint)",
        value=False
    )

    if st.button("Run backtest", type="primary"):
        if end < start:
            st.error("End date must be after start date.")
        elif days > 45:
            st.error("Choose a range of 45 days or fewer.")
        else:
            bt = backtest_dates(start, end, bt_sims, use_hist_odds)
            st.session_state["backtest"] = bt

    bt = st.session_state.get("backtest")
    if isinstance(bt, pd.DataFrame) and not bt.empty:
        probs = bt["model_prob"].clip(.001,.999)
        outcomes = bt["outcome"]

        brier = float(np.mean((probs-outcomes)**2))
        ll = float(-np.mean(outcomes*np.log(probs)+(1-outcomes)*np.log(1-probs)))
        acc = float(((probs>=.5).astype(int)==outcomes).mean())

        a,b,c = st.columns(3)
        a.metric("Brier score", f"{brier:.4f}", help="Lower is better.")
        b.metric("Log loss", f"{ll:.4f}", help="Lower is better.")
        c.metric("50% classification accuracy", f"{acc*100:.1f}%")

        cal = bt.copy()
        cal["bucket"] = pd.cut(
            cal["model_prob"], bins=[0,.4,.45,.5,.55,.6,1],
            labels=["<40","40–45","45–50","50–55","55–60","60+"], include_lowest=True
        )
        grp = cal.groupby("bucket", observed=False).agg(
            predictions=("outcome","size"),
            avg_model=("model_prob","mean"),
            actual_win_rate=("outcome","mean"),
        ).reset_index()
        st.write("### Calibration by probability band")
        st.dataframe(grp, use_container_width=True, hide_index=True)

        if "bet_return" in bt.columns:
            bettable = bt.dropna(subset=["bet_return","edge","odds"])
            if not bettable.empty:
                st.write("### Historical odds performance")
                threshold = st.slider("Backtest edge threshold", 0.0, 15.0, 3.0, .5)
                bets = bettable[bettable.edge >= threshold]
                if not bets.empty:
                    roi = bets.bet_return.mean()
                    st.metric("Flat-stake ROI", f"{roi*100:.2f}%")
                    st.write(f"Bets: **{len(bets)}**")
                else:
                    st.info("No historical bets cleared that edge threshold.")

        st.download_button(
            "Download backtest CSV",
            data=bt.to_csv(index=False).encode(),
            file_name=f"EDGE_backtest_{start}_{end}.csv",
            mime="text/csv"
        )

with tab_cal:
    st.subheader("Probability calibration")
    st.write("Fit a simple logistic calibration layer to the most recent backtest. This adjusts overconfident or underconfident model probabilities without changing the underlying run model.")

    bt = st.session_state.get("backtest")
    if not isinstance(bt, pd.DataFrame) or bt.empty:
        st.info("Run a backtest first.")
    elif LogisticRegression is None:
        st.error("scikit-learn did not load. Check requirements.txt.")
    else:
        x = bt["model_prob"].clip(.001,.999)
        logits = np.log(x/(1-x)).values.reshape(-1,1)
        y = bt["outcome"].astype(int).values

        if len(np.unique(y)) < 2:
            st.error("Backtest needs both wins and losses.")
        else:
            model = LogisticRegression(C=1e6, solver="lbfgs")
            model.fit(logits, y)
            coef = float(model.coef_[0][0])
            intercept = float(model.intercept_[0])

            raw_brier = np.mean((x-y)**2)
            calibrated = model.predict_proba(logits)[:,1]
            cal_brier = np.mean((calibrated-y)**2)

            a,b = st.columns(2)
            a.metric("Raw Brier", f"{raw_brier:.4f}")
            b.metric("Calibrated Brier", f"{cal_brier:.4f}")

            st.write(f"Calibration coefficient: **{coef:.4f}**")
            st.write(f"Calibration intercept: **{intercept:.4f}**")

            if st.button("Activate this calibration for live EDGE"):
                st.session_state["calibration"] = {"coef":coef,"intercept":intercept}
                st.success("Calibration activated for this Streamlit session.")

            if st.button("Clear calibration"):
                st.session_state.pop("calibration", None)
                st.success("Calibration cleared.")

            export = json.dumps({"coef":coef,"intercept":intercept}, indent=2)
            st.download_button(
                "Download calibration JSON",
                export.encode(),
                file_name="edge_calibration.json",
                mime="application/json"
            )

with tab_track:
    st.subheader("Results tracking")
    st.write("Upload exported EDGE bets/results CSVs here to review performance. Streamlit Community Cloud storage is not guaranteed to persist, so v3 uses portable CSV exports.")

    uploaded = st.file_uploader("Upload tracking CSV", type=["csv"])
    if uploaded:
        track = pd.read_csv(uploaded)
        st.dataframe(track, use_container_width=True)
        if "bet_return" in track.columns:
            st.metric("ROI", f"{track.bet_return.mean()*100:.2f}%")
        if "outcome" in track.columns:
            st.metric("Win rate", f"{track.outcome.mean()*100:.1f}%")

    st.write("Recommended tracking columns:")
    st.code("date, sport, game, market, selection, odds, model_prob, market_prob, edge, score, result, bet_return, closing_odds, clv")

with tab_roadmap:
    st.subheader("NFL / NCAA Football / PGA Tour")
    st.write(
        "The application framework is ready for additional sports, but v3 intentionally does **not** fabricate "
        "NFL, NCAA, or PGA predictive probabilities from betting lines alone. Each sport needs its own historical "
        "feature set and backtesting pipeline before EDGE should publish confidence-rated bets."
    )
    st.markdown(
        """
**NFL engine requires:** EPA/play, QB efficiency, pressure/coverage, injuries, rest/travel, weather, opponent-adjusted efficiency.

**NCAA engine requires:** opponent-adjusted efficiency, returning production, transfers, QB/coaching changes, talent, pace, home-field.

**PGA engine requires:** strokes gained categories, course fit, field strength, tee-time weather, player form, outright/top-20 pricing.
        """
    )
    st.info("This is the correct stopping point for a validated MLB platform. Expanding sports should happen only after equivalent data sources are connected.")

st.divider()
st.caption(
    "EDGE v3 is a research and decision-support platform. No model guarantees profit. "
    "Historical calibration, sample size, price availability, and execution quality materially affect results."
)
