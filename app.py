
import os, math, random
from datetime import date, datetime, timedelta
from collections import defaultdict

import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="EDGE MLB v2.2", page_icon="📈", layout="wide")

MLB = "https://statsapi.mlb.com/api/v1"
ODDS_SPORT = "baseball_mlb"
WEATHER = "https://api.open-meteo.com/v1/forecast"
SEASON = date.today().year

# ============================================================
# Helpers
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
    r = requests.get(url, params=params or {}, timeout=20)
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

# ============================================================
# MLB data
# ============================================================

@st.cache_data(ttl=300)
def get_schedule(d):
    return get_json(
        f"{MLB}/schedule",
        {
            "sportId": 1,
            "date": d,
            "hydrate": "probablePitcher,team,venue",
        },
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
                "away": a.get("team", {}).get("name", "Away"),
                "away_id": a.get("team", {}).get("id"),
                "hp": (h.get("probablePitcher") or {}).get("fullName"),
                "hp_id": (h.get("probablePitcher") or {}).get("id"),
                "ap": (a.get("probablePitcher") or {}).get("fullName"),
                "ap_id": (a.get("probablePitcher") or {}).get("id"),
                "venue_id": (g.get("venue") or {}).get("id"),
            })
    return out

@st.cache_data(ttl=900)
def pitcher_stats(pid):
    if not pid:
        return {}
    data = get_json(
        f"{MLB}/people/{pid}/stats",
        {"stats": "season", "group": "pitching", "season": SEASON},
    )
    splits = data.get("stats", [{}])[0].get("splits") or []
    s = splits[0].get("stat", {}) if splits else {}
    return {
        "era": float(s.get("era", 4.30) or 4.30),
        "whip": float(s.get("whip", 1.30) or 1.30),
        "k9": float(s.get("strikeoutsPer9Inn", 8.5) or 8.5),
        "bb9": float(s.get("walksPer9Inn", 3.2) or 3.2),
        "hr9": float(s.get("homeRunsPer9", 1.2) or 1.2),
        "innings": float(s.get("inningsPitched", 0) or 0),
    }

@st.cache_data(ttl=900)
def team_hitting(tid):
    if not tid:
        return {}
    data = get_json(
        f"{MLB}/teams/{tid}/stats",
        {"stats": "season", "group": "hitting", "season": SEASON, "sportIds": 1},
    )
    splits = data.get("stats", [{}])[0].get("splits") or []
    s = splits[0].get("stat", {}) if splits else {}
    return {
        "avg": float(s.get("avg", .240) or .240),
        "obp": float(s.get("obp", .310) or .310),
        "slg": float(s.get("slg", .400) or .400),
        "ops": float(s.get("ops", .710) or .710),
    }

@st.cache_data(ttl=900)
def team_pitching(tid):
    if not tid:
        return {}
    data = get_json(
        f"{MLB}/teams/{tid}/stats",
        {"stats": "season", "group": "pitching", "season": SEASON, "sportIds": 1},
    )
    splits = data.get("stats", [{}])[0].get("splits") or []
    s = splits[0].get("stat", {}) if splits else {}
    return {
        "era": float(s.get("era", 4.20) or 4.20),
        "whip": float(s.get("whip", 1.30) or 1.30),
        "k9": float(s.get("strikeoutsPer9Inn", 8.5) or 8.5),
    }

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

# Known MLB venue sanity bounds / fallback coordinates.
# Used only if MLB venue coordinates are missing or clearly implausible.
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
    reason = "MLB coordinates"
    valid = (
        isinstance(lat, (int, float)) and isinstance(lon, (int, float))
        and 20 <= float(lat) <= 50 and -130 <= float(lon) <= -65
    )
    if not valid and v.get("name") in PARK_FALLBACKS:
        lat, lon = PARK_FALLBACKS[v["name"]]
        reason = "fallback park coordinates"
        valid = True
    return {**v, "lat": lat, "lon": lon}, valid, reason

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
# Weather at first pitch with sanity checks
# ============================================================

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

def fahrenheit(c):
    return c * 9 / 5 + 32

def first_pitch_weather(game_date_utc, venue):
    v, coord_ok, coord_source = validated_venue(venue)
    if not coord_ok or not game_date_utc:
        return 0.0, "Weather unavailable", 35, {"coord_source": coord_source}

    w = weather_forecast(v["lat"], v["lon"])
    h = w.get("hourly", {})
    times = h.get("time", [])
    temps_c = h.get("temperature_2m", [])
    winds = h.get("wind_speed_10m", [])
    rains = h.get("precipitation_probability", [])

    if not times or not temps_c:
        return 0.0, "Weather unavailable", 35, {"coord_source": coord_source}

    # Convert MLB UTC start to venue local clock using Open-Meteo offset.
    dt_utc = datetime.fromisoformat(game_date_utc.replace("Z", "+00:00"))
    offset = int(w.get("utc_offset_seconds", 0) or 0)
    target_local = (dt_utc + timedelta(seconds=offset)).replace(tzinfo=None)

    parsed = [datetime.fromisoformat(t) for t in times]
    idx = min(range(len(parsed)), key=lambda i: abs((parsed[i] - target_local).total_seconds()))

    temp_c = float(temps_c[idx])
    temp_f = fahrenheit(temp_c)
    wind = float(winds[idx]) if idx < len(winds) else 0.0
    rain = float(rains[idx]) if idx < len(rains) else 0.0

    # Plausibility checks.
    plausible_temp = -10 <= temp_f <= 115
    hour_delta = abs((parsed[idx] - target_local).total_seconds()) / 3600
    near_first_pitch = hour_delta <= 1.5

    # Additional seasonal sanity: extreme cold in Aug at southern parks is suspicious.
    month = target_local.month
    southern = float(v["lat"]) < 36
    seasonal_ok = not (month in (6, 7, 8, 9) and southern and temp_f < 50)

    quality = 90
    flags = []
    if coord_source != "MLB coordinates":
        quality -= 5
        flags.append(coord_source)
    if not plausible_temp:
        quality -= 40
        flags.append("temperature out of bounds")
    if not near_first_pitch:
        quality -= 20
        flags.append("forecast not near first pitch")
    if not seasonal_ok:
        quality -= 35
        flags.append("seasonal temperature sanity failed")

    # If weather fails sanity, don't let it move the projection.
    trustworthy = plausible_temp and near_first_pitch and seasonal_ok
    adj = 0.0
    if trustworthy:
        if temp_f >= 85:
            adj += .10
        elif temp_f <= 45:
            adj -= .08
        if wind >= 15:
            adj += .04
        if rain >= 60:
            adj -= .05

    note = f"{temp_f:.0f}°F · {wind:.0f} mph wind · {rain:.0f}% rain at first pitch"
    if not trustworthy:
        note += " · ⚠️ weather ignored by model"
    if flags:
        note += f" ({'; '.join(flags)})"

    return adj, note, max(20, min(95, quality)), {
        "coord_source": coord_source,
        "lat": v["lat"],
        "lon": v["lon"],
        "local_target": str(target_local),
        "matched_hour": str(parsed[idx]),
        "trusted": trustworthy,
    }

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
    innings = max(0.0, s.get("innings", 0))
    shrink = min(1.0, innings / 80.0)
    return max(-1.25, min(1.25, x * shrink))

def offense_quality(s):
    if not s:
        return 0
    x = (
        (s["ops"] - .710) * 2.4
        + (s["obp"] - .310) * 1.4
        + (s["slg"] - .400) * 1.2
    )
    return max(-1.25, min(1.25, x))

def bullpen_proxy(s):
    if not s:
        return 0
    x = (4.20 - s["era"]) * .10 + (1.30 - s["whip"]) * .35 + (s["k9"] - 8.5) * .02
    return max(-.60, min(.60, x))

def project_game(g):
    hp, ap = pitcher_stats(g["hp_id"]), pitcher_stats(g["ap_id"])
    ho, ao = team_hitting(g["home_id"]), team_hitting(g["away_id"])
    htp, atp = team_pitching(g["home_id"]), team_pitching(g["away_id"])

    venue = venue_info(g["venue_id"])
    weather_adj, weather_note, weather_q, weather_debug = first_pitch_weather(g["gameDate"], venue)

    confirmed, lineups = lineup_status(g["gamePk"])
    lineup_q = 95 if confirmed else 58

    league = 4.45
    hpq, apq = pitcher_quality(hp), pitcher_quality(ap)
    hoq, aoq = offense_quality(ho), offense_quality(ao)
    hbp, abp = bullpen_proxy(htp), bullpen_proxy(atp)

    home_components = {
        "baseline": league,
        "home_field": .20,
        "offense": .58 * hoq,
        "opp_starter": -.72 * apq,
        "opp_bullpen": -.18 * abp,
        "own_starter_context": .05 * hpq,
        "weather": weather_adj / 2,
    }
    away_components = {
        "baseline": league,
        "offense": .58 * aoq,
        "opp_starter": -.72 * hpq,
        "opp_bullpen": -.18 * hbp,
        "own_starter_context": .05 * apq,
        "weather": weather_adj / 2,
    }

    home = max(1.5, min(8.5, sum(home_components.values())))
    away = max(1.5, min(8.5, sum(away_components.values())))

    starter_q = 90 if g["hp_id"] and g["ap_id"] else 55
    data_quality = round(.35 * starter_q + .25 * lineup_q + .20 * weather_q + .20 * 78)

    return home, away, {
        "weather": weather_note,
        "venue": venue.get("name", ""),
        "lineups_confirmed": confirmed,
        "data_quality": data_quality,
        "weather_debug": weather_debug,
        "home_components": home_components,
        "away_components": away_components,
    }

# ============================================================
# Simulation
# ============================================================

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
    ties = max(0.0, 1.0 - home_w - away_w)
    return home_w + ties / 2, away_w + ties / 2

def model_total(scores, side, point):
    n = len(scores)
    if side.lower() == "over":
        wins = sum((h + a) > point for h, a in scores)
        pushes = sum((h + a) == point for h, a in scores)
    else:
        wins = sum((h + a) < point for h, a in scores)
        pushes = sum((h + a) == point for h, a in scores)
    return wins / max(1, n - pushes)

def model_spread(scores, team_is_home, point):
    n = len(scores)
    wins = pushes = 0
    for h, a in scores:
        margin = (h - a) if team_is_home else (a - h)
        result = margin + point
        wins += result > 0
        pushes += result == 0
    return wins / max(1, n - pushes)

# ============================================================
# Odds API / consensus
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
# Reliability / sanity layer
# ============================================================

def disagreement_penalty(edge_pp, data_quality, books):
    """
    Large disagreements with liquid markets are treated as a reason for
    caution, not automatically as confidence.
    """
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
    return int(round(max(0, min(100, raw))))

def reliability_label(edge_pp, data_quality, books, score):
    if data_quality < 65:
        return "LOW", "Data quality is too weak for a strong recommendation."
    if abs(edge_pp) > 12 and books >= 6:
        return "CAUTION", "Model strongly disagrees with a liquid market; verify inputs before trusting the signal."
    if score >= 80 and data_quality >= 80:
        return "HIGH", "Model and data quality meet the current research threshold."
    if score >= 68:
        return "MEDIUM", "Potential signal, but still requires calibration and review."
    return "LOW", "Signal is not strong enough after reliability penalties."

# ============================================================
# UI
# ============================================================

st.title("📈 EDGE MLB v2.2")
st.caption("Reliability-first MLB model · weather sanity checks · market disagreement penalties · explainable run projections")

with st.sidebar:
    game_date = st.date_input("Game date", date.today())
    min_edge = st.slider("Minimum model edge (%)", 0.0, 15.0, 2.5, .5)
    min_score = st.slider("Minimum Edge Score", 0, 100, 65, 5)
    sims = st.select_slider("Simulations", [5000, 10000, 25000, 50000], value=25000)
    markets = st.multiselect("Markets", ["Moneyline", "Totals", "Run line"], default=["Moneyline", "Totals", "Run line"])
    st.divider()
    st.caption("v2.2 intentionally penalizes large model-market disagreements until the model is historically calibrated.")

try:
    games = games_for(game_date.isoformat())
except Exception as e:
    games = []
    st.error(f"MLB data error: {e}")

events, odds_error = odds_feed()
if odds_error:
    st.warning(odds_error)

lookup = {(norm(e.get("away_team")), norm(e.get("home_team"))): e for e in events}
rows = []

for g in games:
    event = lookup.get((norm(g["away"]), norm(g["home"])))
    if not event:
        continue

    try:
        hr, ar, meta = project_game(g)
        scores = simulate_scores(hr, ar, sims, int(g["gamePk"] or 1))
        home_p, away_p = model_moneyline(scores)
    except Exception:
        continue

    def add_row(market_type, selection, market, model_p):
        edge = (model_p - market["consensus_prob"]) * 100
        stability = 78 if market_type == "ML" else 73 if market_type == "TOTAL" else 70
        sc = edge_score(edge, meta["data_quality"], market["books"], stability, market_type)
        reliability, rel_note = reliability_label(edge, meta["data_quality"], market["books"], sc)
        rows.append({
            "game": f'{g["away"]} @ {g["home"]}',
            "market_type": market_type,
            "selection": selection,
            "book": market["best_book"],
            "odds": market["best_odds"],
            "model_prob": model_p,
            "market_prob": market["consensus_prob"],
            "edge": edge,
            "fair": fair_price(model_p),
            "ev": ev_pct(model_p, market["best_odds"]),
            "score": sc,
            "reliability": reliability,
            "reliability_note": rel_note,
            "books": market["books"],
            "proj_score": f"{g['away']} {ar:.1f} – {g['home']} {hr:.1f}",
            "pitchers": f'{g["ap"] or "TBD"} / {g["hp"] or "TBD"}',
            "weather": meta["weather"],
            "lineup": "Confirmed" if meta["lineups_confirmed"] else "Projected/TBD",
            "data_quality": meta["data_quality"],
            "home_components": meta["home_components"],
            "away_components": meta["away_components"],
            "weather_debug": meta["weather_debug"],
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

df = pd.DataFrame(rows)
if not df.empty:
    candidates = df[(df.edge >= min_edge) & (df.score >= min_score)].sort_values(["score", "edge"], ascending=False)
else:
    candidates = pd.DataFrame()

c1, c2, c3, c4 = st.columns(4)
c1.metric("MLB games", len(games))
c2.metric("Markets analyzed", len(df))
c3.metric("Best edge", f"{candidates.edge.max():.1f}%" if not candidates.empty else "—")
c4.metric("Best score", int(candidates.score.max()) if not candidates.empty else "—")

st.subheader("🔥 Today's strongest MLB edges")

if candidates.empty:
    st.success("NO BET — no market clears the selected reliability thresholds.")
else:
    for _, r in candidates.head(12).iterrows():
        with st.container(border=True):
            a, b, c = st.columns([4, 1, 1])
            a.markdown(f"### {r.selection} — {r.game}")
            b.metric("EDGE", f"{r.edge:+.1f}%")
            c.metric("SCORE", int(r.score))

            st.write(
                f'**Best price:** {r.odds:+g} at {r.book} · '
                f'**Model:** {r.model_prob*100:.1f}% · '
                f'**No-vig market:** {r.market_prob*100:.1f}% · '
                f'**Fair:** {r.fair:+d} · '
                f'**Uncalibrated EV:** {r.ev:+.1f}%'
            )
            st.write(
                f'**Projection:** {r.proj_score} · '
                f'**Pitchers:** {r.pitchers} · '
                f'**Weather:** {r.weather}'
            )
            st.write(
                f'**Lineups:** {r.lineup} · '
                f'**Data quality:** {int(r.data_quality)}/100 · '
                f'**Consensus books:** {int(r.books)}'
            )

            if r.reliability == "CAUTION":
                st.warning(f"**Reliability: CAUTION** — {r.reliability_note}")
            elif r.reliability == "HIGH":
                st.success(f"**Reliability: HIGH** — {r.reliability_note}")
            elif r.reliability == "MEDIUM":
                st.info(f"**Reliability: MEDIUM** — {r.reliability_note}")
            else:
                st.warning(f"**Reliability: LOW** — {r.reliability_note}")

            with st.expander("Why does the model project this score?"):
                st.write("**Away-team run components**")
                st.json({k: round(v, 3) for k, v in r.away_components.items()})
                st.write("**Home-team run components**")
                st.json({k: round(v, 3) for k, v in r.home_components.items()})
                st.caption("These are transparent research coefficients, not yet fitted from historical training data.")

st.subheader("📋 Full market board")
if not df.empty:
    board = df.copy()
    board["model_prob"] = (board["model_prob"] * 100).round(1).astype(str) + "%"
    board["market_prob"] = (board["market_prob"] * 100).round(1).astype(str) + "%"
    board["edge"] = board["edge"].round(1).astype(str) + "%"
    board["ev"] = board["ev"].round(1).astype(str) + "%"
    st.dataframe(
        board[
            ["game", "market_type", "selection", "book", "odds",
             "model_prob", "market_prob", "edge", "fair", "ev",
             "score", "reliability", "books", "proj_score",
             "lineup", "weather"]
        ],
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No matching live MLB markets were returned.")

st.subheader("💬 EDGE Analyst")
q = st.text_input("Ask", "What are the strongest MLB bets today?")
if q:
    if candidates.empty:
        st.info("No qualifying signal under the current thresholds.")
    else:
        r = candidates.iloc[0]
        st.info(
            f"Top current signal: **{r.selection}** at **{r.odds:+g}**. "
            f"Model probability is **{r.model_prob*100:.1f}%** versus a "
            f"**{r.market_prob*100:.1f}% no-vig consensus**. "
            f"Estimated model edge is **{r.edge:+.1f} points**, but reliability is **{r.reliability}**. "
            f"Fair price is approximately **{r.fair:+d}**."
        )

with st.expander("🔎 Weather / model audit"):
    st.write(f"MLB games found: **{len(games)}**")
    st.write(f"Odds events returned: **{len(events)}**")
    st.write(f"Markets analyzed: **{len(df)}**")
    st.write(f"Simulations per game: **{sims:,}**")
    st.write("Weather sanity checks now validate MLB venue coordinates, convert Open-Meteo temperatures from Celsius to Fahrenheit, match the nearest hourly forecast to first pitch, and ignore implausible weather instead of feeding it into the model.")
    st.write("Large model-market disagreements now reduce Edge Score rather than automatically increasing confidence.")
    st.write("Each recommendation now exposes the run-projection components so you can see what is driving the model.")
    st.warning("v2.2 is still uncalibrated. Historical backtesting and out-of-sample validation remain necessary before real-money trust.")

st.caption("EDGE MLB v2.2 — reliability-first research prototype")
