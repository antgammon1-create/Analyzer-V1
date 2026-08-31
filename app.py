
import os, math, random
from datetime import date, datetime, timezone, timedelta
from collections import defaultdict

import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="EDGE MLB v2.1", page_icon="📈", layout="wide")

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
# MLB schedule / stats
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
        "runs": float(s.get("runs", 0) or 0),
        "games": float(s.get("gamesPlayed", 0) or 0),
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
        "bb9": float(s.get("walksPer9Inn", 3.2) or 3.2),
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
        "lat": coords.get("latitude"),
        "lon": coords.get("longitude"),
    }

# ============================================================
# Confirmed lineup proxy
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
# Weather matched to FIRST PITCH
# ============================================================

@st.cache_data(ttl=900)
def weather_forecast(lat, lon):
    if lat is None or lon is None:
        return {}
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
    if not venue or not venue.get("lat") or not game_date_utc:
        return 0.0, "Weather unavailable", 45

    w = weather_forecast(venue["lat"], venue["lon"])
    h = w.get("hourly", {})
    times = h.get("time", [])
    temps = h.get("temperature_2m", [])
    winds = h.get("wind_speed_10m", [])
    rains = h.get("precipitation_probability", [])
    if not times:
        return 0.0, "Weather unavailable", 45

    # MLB gameDate is UTC. Open-Meteo returns local venue times plus UTC offset.
    dt_utc = datetime.fromisoformat(game_date_utc.replace("Z", "+00:00"))
    offset = int(w.get("utc_offset_seconds", 0) or 0)
    local_dt = dt_utc + timedelta(seconds=offset)
    target = local_dt.replace(tzinfo=None)

    parsed = [datetime.fromisoformat(t) for t in times]
    idx = min(range(len(parsed)), key=lambda i: abs((parsed[i] - target).total_seconds()))

    temp = float(temps[idx]) if idx < len(temps) else 70.0
    wind = float(winds[idx]) if idx < len(winds) else 0.0
    rain = float(rains[idx]) if idx < len(rains) else 0.0

    # Conservative run-environment adjustment. Wind direction is shown
    # but not converted into in/out until park orientation is added.
    adj = 0.0
    if temp >= 85:
        adj += .10
    elif temp <= 45:
        adj -= .08
    if wind >= 15:
        adj += .04
    if rain >= 60:
        adj -= .05

    quality = 85 if abs((parsed[idx] - target).total_seconds()) <= 3600 else 65
    note = f"{temp:.0f}°F · {wind:.0f} mph wind · {rain:.0f}% rain at first pitch"
    return adj, note, quality

# ============================================================
# Model features
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
    # Small-sample shrinkage toward league average.
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
    weather_adj, weather_note, weather_q = first_pitch_weather(g["gameDate"], venue)

    confirmed, lineups = lineup_status(g["gamePk"])
    lineup_q = 95 if confirmed else 58

    # Transparent v2.1 research model.
    league = 4.45
    hpq, apq = pitcher_quality(hp), pitcher_quality(ap)
    hoq, aoq = offense_quality(ho), offense_quality(ao)
    hbp, abp = bullpen_proxy(htp), bullpen_proxy(atp)

    home = (
        league
        + .20
        + .58 * hoq
        - .72 * apq
        - .18 * abp
        + .05 * hpq
        + weather_adj / 2
    )
    away = (
        league
        + .58 * aoq
        - .72 * hpq
        - .18 * hbp
        + .05 * apq
        + weather_adj / 2
    )

    home = max(1.5, min(8.5, home))
    away = max(1.5, min(8.5, away))

    starter_q = 90 if g["hp_id"] and g["ap_id"] else 55
    data_quality = round(.35 * starter_q + .25 * lineup_q + .20 * weather_q + .20 * 78)

    return home, away, {
        "weather": weather_note,
        "venue": venue.get("name", ""),
        "lineups_confirmed": confirmed,
        "home_lineup": lineups.get("home", []),
        "away_lineup": lineups.get("away", []),
        "data_quality": data_quality,
        "starter_quality": starter_q,
        "lineup_quality": lineup_q,
        "weather_quality": weather_q,
    }

# ============================================================
# Simulation: store outcomes once, then price all markets
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
    scores = []
    for _ in range(n):
        scores.append((poisson(rng, home_runs), poisson(rng, away_runs)))
    return scores

def model_moneyline(scores):
    n = len(scores)
    home_w = sum(h > a for h, a in scores) / n
    away_w = sum(a > h for h, a in scores) / n
    # Regulation ties are redistributed evenly for MLB moneyline purposes.
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
    # Pushes are excluded when estimating win probability for a standard bet.
    decisions = max(1, n - pushes)
    return wins / decisions

def model_spread(scores, team_is_home, point):
    n = len(scores)
    wins = pushes = 0
    for h, a in scores:
        margin = (h - a) if team_is_home else (a - h)
        result = margin + point
        wins += result > 0
        pushes += result == 0
    decisions = max(1, n - pushes)
    return wins / decisions

# ============================================================
# Odds API + consensus market
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
    # Return best price + no-vig consensus probability by team.
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

        team_outcomes = [o for o in outcomes if o.get("price") is not None]
        if len(team_outcomes) < 2:
            continue

        # MLB H2H should be two-way. Normalize bookmaker vig out.
        p1 = implied_prob(team_outcomes[0]["price"])
        p2 = implied_prob(team_outcomes[1]["price"])
        nv1, nv2 = no_vig(p1, p2)

        for o, nv in zip(team_outcomes[:2], [nv1, nv2]):
            team = o.get("name")
            book_probs[team].append(nv)
            offers[team].append((float(o["price"]), book.get("title", "Book")))

    result = {}
    for team, probs in book_probs.items():
        if not probs:
            continue
        best = min(offers[team], key=lambda x: implied_prob(x[0]))
        result[team] = {
            "consensus_prob": sum(probs) / len(probs),
            "best_odds": best[0],
            "best_book": best[1],
            "books": len(probs),
        }
    return result

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
    # Group matching total/spread pairs by point and book, de-vig within each book,
    # then average across books. Also retain best bettor price.
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["book"], r["point"])].append(r)

    consensus = defaultdict(list)
    offers = defaultdict(list)

    for (book, point), pair in grouped.items():
        if len(pair) < 2:
            continue

        if kind == "totals":
            # Need Over + Under for same point.
            names = {p["name"].lower(): p for p in pair}
            if "over" not in names or "under" not in names:
                continue
            a, b = names["over"], names["under"]
            nva, nvb = no_vig(implied_prob(a["odds"]), implied_prob(b["odds"]))
            consensus[("Over", point)].append(nva)
            consensus[("Under", point)].append(nvb)
            offers[("Over", point)].append((a["odds"], book))
            offers[("Under", point)].append((b["odds"], book))
        else:
            # Spreads: use two team outcomes at same point family.
            if len(pair) != 2:
                continue
            a, b = pair[0], pair[1]
            nva, nvb = no_vig(implied_prob(a["odds"]), implied_prob(b["odds"]))
            consensus[(a["name"], a["point"])].append(nva)
            consensus[(b["name"], b["point"])].append(nvb)
            offers[(a["name"], a["point"])].append((a["odds"], book))
            offers[(b["name"], b["point"])].append((b["odds"], book))

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
# Edge scoring
# ============================================================

def edge_score(edge_pp, data_quality, market_books, stability, market_type):
    edge_component = min(100, max(0, edge_pp * 5))
    market_q = min(100, 55 + market_books * 6)
    type_q = 86 if market_type == "ML" else 80 if market_type == "TOTAL" else 77
    raw = (
        edge_component * .34
        + data_quality * .25
        + stability * .16
        + market_q * .10
        + type_q * .15
    )
    return int(round(max(0, min(100, raw))))

# ============================================================
# UI / analysis
# ============================================================

st.title("📈 EDGE MLB v2.1")
st.caption("First-pitch weather · no-vig consensus pricing · ML/totals/run lines · lineup/data quality")

with st.sidebar:
    game_date = st.date_input("Game date", date.today())
    min_edge = st.slider("Minimum model edge (%)", 0.0, 15.0, 2.5, .5)
    min_score = st.slider("Minimum Edge Score", 0, 100, 65, 5)
    sims = st.select_slider("Simulations", [5000, 10000, 25000, 50000], value=25000)
    market_filter = st.multiselect("Markets", ["Moneyline", "Totals", "Run line"], default=["Moneyline", "Totals", "Run line"])
    st.divider()
    st.caption("v2.1 is still uncalibrated. EV is labeled 'Uncalibrated EV' until historical validation is complete.")

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

    # MONEYLINE
    if "Moneyline" in market_filter:
        ml = moneyline_market(event)
        for team, market in ml.items():
            model_p = home_p if norm(team) == norm(g["home"]) else away_p
            edge = (model_p - market["consensus_prob"]) * 100
            sc = edge_score(edge, meta["data_quality"], market["books"], 78, "ML")
            rows.append({
                "game": f'{g["away"]} @ {g["home"]}',
                "market_type": "ML",
                "selection": f"{team} ML",
                "book": market["best_book"],
                "odds": market["best_odds"],
                "model_prob": model_p,
                "market_prob": market["consensus_prob"],
                "edge": edge,
                "fair": fair_price(model_p),
                "ev": ev_pct(model_p, market["best_odds"]),
                "score": sc,
                "books": market["books"],
                "proj_score": f"{g['away']} {ar:.1f} – {g['home']} {hr:.1f}",
                "pitchers": f'{g["ap"] or "TBD"} / {g["hp"] or "TBD"}',
                "weather": meta["weather"],
                "lineup": "Confirmed" if meta["lineups_confirmed"] else "Projected/TBD",
                "data_quality": meta["data_quality"],
            })

    # TOTALS
    if "Totals" in market_filter:
        totals = group_two_way_consensus(raw_markets(event, "totals"), "totals")
        for (side, point), market in totals.items():
            model_p = model_total(scores, side, float(point))
            edge = (model_p - market["consensus_prob"]) * 100
            sc = edge_score(edge, meta["data_quality"], market["books"], 73, "TOTAL")
            rows.append({
                "game": f'{g["away"]} @ {g["home"]}',
                "market_type": "TOTAL",
                "selection": f"{side} {float(point):g}",
                "book": market["best_book"],
                "odds": market["best_odds"],
                "model_prob": model_p,
                "market_prob": market["consensus_prob"],
                "edge": edge,
                "fair": fair_price(model_p),
                "ev": ev_pct(model_p, market["best_odds"]),
                "score": sc,
                "books": market["books"],
                "proj_score": f"{g['away']} {ar:.1f} – {g['home']} {hr:.1f}",
                "pitchers": f'{g["ap"] or "TBD"} / {g["hp"] or "TBD"}',
                "weather": meta["weather"],
                "lineup": "Confirmed" if meta["lineups_confirmed"] else "Projected/TBD",
                "data_quality": meta["data_quality"],
            })

    # RUN LINES / SPREADS
    if "Run line" in market_filter:
        spreads = group_two_way_consensus(raw_markets(event, "spreads"), "spreads")
        for (team, point), market in spreads.items():
            model_p = model_spread(scores, norm(team) == norm(g["home"]), float(point))
            edge = (model_p - market["consensus_prob"]) * 100
            sc = edge_score(edge, meta["data_quality"], market["books"], 70, "RL")
            rows.append({
                "game": f'{g["away"]} @ {g["home"]}',
                "market_type": "RL",
                "selection": f"{team} {float(point):+g}",
                "book": market["best_book"],
                "odds": market["best_odds"],
                "model_prob": model_p,
                "market_prob": market["consensus_prob"],
                "edge": edge,
                "fair": fair_price(model_p),
                "ev": ev_pct(model_p, market["best_odds"]),
                "score": sc,
                "books": market["books"],
                "proj_score": f"{g['away']} {ar:.1f} – {g['home']} {hr:.1f}",
                "pitchers": f'{g["ap"] or "TBD"} / {g["hp"] or "TBD"}',
                "weather": meta["weather"],
                "lineup": "Confirmed" if meta["lineups_confirmed"] else "Projected/TBD",
                "data_quality": meta["data_quality"],
            })

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
    st.success("NO BET — no market clears the selected thresholds.")
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
                f'**Lineups:** {r.lineup} · **Data quality:** {int(r.data_quality)}/100 · '
                f'**Consensus books:** {int(r.books)}'
            )
            st.caption("v2.1 improves data handling, but the model is not yet historically calibrated. Treat this as a research signal, not a proven betting edge.")

st.subheader("📋 Full market board")
if not df.empty:
    board = df.copy()
    board["model_prob"] = (board["model_prob"] * 100).round(1).astype(str) + "%"
    board["market_prob"] = (board["market_prob"] * 100).round(1).astype(str) + "%"
    board["edge"] = board["edge"].round(1).astype(str) + "%"
    board["ev"] = board["ev"].round(1).astype(str) + "%"
    st.dataframe(
        board[
            ["game", "market_type", "selection", "book", "odds", "model_prob", "market_prob",
             "edge", "fair", "ev", "score", "books", "proj_score", "lineup", "weather"]
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
            f"EDGE estimates **{r.model_prob*100:.1f}%** versus a **{r.market_prob*100:.1f}% no-vig consensus**, "
            f"for **{r.edge:+.1f} percentage points** of model edge. "
            f"The fair price is approximately **{r.fair:+d}**. "
            f"Data quality is **{int(r.data_quality)}/100** and lineups are **{r.lineup.lower()}**."
        )

with st.expander("🔎 Model / data audit"):
    st.write(f"MLB games found: **{len(games)}**")
    st.write(f"Odds events returned: **{len(events)}**")
    st.write(f"Markets analyzed: **{len(df)}**")
    st.write(f"Monte Carlo simulations per game: **{sims:,}**")
    st.write("Weather is now matched to the scheduled first-pitch time using the venue's Open-Meteo UTC offset.")
    st.write("Moneyline, total and run-line market probabilities are de-vigged within sportsbooks, then averaged across books.")
    st.write("Confirmed batting orders are detected from MLB boxscore data when available; otherwise the model applies a data-quality penalty.")
    st.write("Bullpen strength is currently a team-pitching proxy. True reliever availability/fatigue is planned for the backtesting build.")
    st.warning("The model is still uncalibrated. Do not interpret displayed EV or Edge Score as proven profitability until historical backtesting and out-of-sample validation are complete.")

st.caption("EDGE MLB v2.1 — research prototype")
