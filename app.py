import os, math, random
from datetime import date
import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="EDGE MLB v2", page_icon="📈", layout="wide")

MLB = "https://statsapi.mlb.com/api/v1"
ODDS_SPORT = "baseball_mlb"
WEATHER = "https://api.open-meteo.com/v1/forecast"

def secret(name):
    v = os.getenv(name, "")
    try:
        v = st.secrets.get(name, v)
    except Exception:
        pass
    return v

@st.cache_data(ttl=300)
def get_json(url, params=None):
    r = requests.get(url, params=params or {}, timeout=20)
    r.raise_for_status()
    return r.json()

def implied_prob(odds):
    odds = float(odds)
    return 100/(odds+100) if odds >= 0 else -odds/(-odds+100)

def fair_price(p):
    p = max(.0001, min(.9999, float(p)))
    return round(-100*p/(1-p)) if p >= .5 else round(100*(1-p)/p)

def ev_pct(p, odds):
    odds = float(odds)
    dec = 1 + (odds/100 if odds >= 0 else 100/abs(odds))
    return (p*dec - 1)*100

@st.cache_data(ttl=300)
def get_schedule(d):
    return get_json(f"{MLB}/schedule", {
        "sportId": 1, "date": d,
        "hydrate": "probablePitcher,team,venue"
    }).get("dates", [])

def games_for(d):
    out = []
    for day in get_schedule(d):
        for g in day.get("games", []):
            h = g.get("teams", {}).get("home", {})
            a = g.get("teams", {}).get("away", {})
            out.append({
                "gamePk": g.get("gamePk"),
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

SEASON = date.today().year

@st.cache_data(ttl=900)
def pitcher_stats(pid):
    if not pid:
        return {}
    data = get_json(f"{MLB}/people/{pid}/stats", {
        "stats": "season", "group": "pitching", "season": SEASON
    })
    splits = data.get("stats", [{}])[0].get("splits") or []
    s = splits[0].get("stat", {}) if splits else {}
    return {
        "era": float(s.get("era", 4.30) or 4.30),
        "whip": float(s.get("whip", 1.30) or 1.30),
        "k9": float(s.get("strikeoutsPer9Inn", 8.5) or 8.5),
        "hr9": float(s.get("homeRunsPer9", 1.2) or 1.2),
    }

@st.cache_data(ttl=900)
def team_hitting(tid):
    if not tid:
        return {}
    data = get_json(f"{MLB}/teams/{tid}/stats", {
        "stats": "season", "group": "hitting",
        "season": SEASON, "sportIds": 1
    })
    splits = data.get("stats", [{}])[0].get("splits") or []
    s = splits[0].get("stat", {}) if splits else {}
    return {
        "avg": float(s.get("avg", .240) or .240),
        "obp": float(s.get("obp", .310) or .310),
        "slg": float(s.get("slg", .400) or .400),
        "ops": float(s.get("ops", .710) or .710),
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
    coords = v.get("location", {}).get("defaultCoordinates", {})
    return {
        "name": v.get("name", ""),
        "lat": coords.get("latitude"),
        "lon": coords.get("longitude"),
    }

@st.cache_data(ttl=900)
def weather(lat, lon):
    if lat is None or lon is None:
        return {}
    return get_json(WEATHER, {
        "latitude": lat, "longitude": lon,
        "hourly": "temperature_2m,precipitation_probability,wind_speed_10m",
        "forecast_days": 2, "timezone": "auto"
    })

def pitcher_quality(s):
    if not s:
        return 0
    x = ((4.30-s["era"])*.18 +
         (1.30-s["whip"])*.70 +
         (s["k9"]-8.5)*.035 +
         (1.20-s["hr9"])*.18)
    return max(-1.25, min(1.25, x))

def offense_quality(s):
    if not s:
        return 0
    x = ((s["ops"]-.710)*2.4 +
         (s["obp"]-.310)*1.4 +
         (s["slg"]-.400)*1.2)
    return max(-1.25, min(1.25, x))

def projection(g):
    hp, ap = pitcher_stats(g["hp_id"]), pitcher_stats(g["ap_id"])
    ho, ao = team_hitting(g["home_id"]), team_hitting(g["away_id"])

    v = venue_info(g["venue_id"])
    w = weather(v.get("lat"), v.get("lon")) if v else {}
    h = w.get("hourly", {})
    temp = (h.get("temperature_2m") or [70])[0]
    wind = (h.get("wind_speed_10m") or [0])[0]
    rain = (h.get("precipitation_probability") or [0])[0]

    weather_adj = (
        .10 if temp >= 85 else
        -.08 if temp <= 45 else 0
    )
    weather_adj += .08 if wind >= 15 else 0
    weather_adj -= .05 if rain >= 60 else 0

    home = 4.45 + .20 + .55*offense_quality(ho) - .75*pitcher_quality(ap) + .15*pitcher_quality(hp) + weather_adj/2
    away = 4.45 + .55*offense_quality(ao) - .75*pitcher_quality(hp) + .15*pitcher_quality(ap) + weather_adj/2

    return max(1.5, min(8.5, home)), max(1.5, min(8.5, away)), {
        "weather": f"{temp:.0f}°F · {wind:.0f} mph wind · {rain:.0f}% rain",
        "venue": v.get("name", "")
    }

def poisson(rng, lam):
    L = math.exp(-lam)
    k, p = 0, 1.0
    while p > L:
        k += 1
        p *= rng.random()
    return k-1

def simulate(hr, ar, n, seed):
    rng = random.Random(seed)
    hw = aw = over = 0
    for _ in range(n):
        h, a = poisson(rng, hr), poisson(rng, ar)
        hw += h > a
        aw += a > h
        over += h+a > 8.5
    return hw/n, aw/n, over/n

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
                "oddsFormat": "american"
            },
            timeout=20
        )
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return [], f"Odds API error: {e}"

def norm(s):
    return "".join(c.lower() for c in s if c.isalnum())

def best_moneylines(event):
    prices = {}
    for book in event.get("bookmakers", []):
        for market in book.get("markets", []):
            if market.get("key") != "h2h":
                continue
            for outcome in market.get("outcomes", []):
                prices.setdefault(outcome["name"], []).append(
                    (float(outcome["price"]), book.get("title", "Book"))
                )
    return [(team, min(vals, key=lambda x: implied_prob(x[0])))
            for team, vals in prices.items()]

def edge_score(edge, data=80, stability=75, market=88, situational=72):
    edge_component = min(100, max(0, edge*5))
    return int(round(max(0, min(100,
        edge_component*.35 +
        data*.20 +
        stability*.20 +
        market*.10 +
        situational*.15
    ))))

st.title("📈 EDGE MLB v2")
st.caption("Live MLB schedule + current sportsbook odds + season stats + weather + Monte Carlo model")

with st.sidebar:
    game_date = st.date_input("Game date", date.today())
    min_edge = st.slider("Minimum model edge (%)", 0.0, 15.0, 2.5, .5)
    min_score = st.slider("Minimum Edge Score", 0, 100, 65, 5)
    sims = st.select_slider("Simulations", [2500, 5000, 10000, 25000], value=10000)
    st.divider()
    st.caption("v2 is a transparent research prototype. Historical calibration is still required.")

try:
    games = games_for(game_date.isoformat())
except Exception as e:
    games = []
    st.error(f"MLB data error: {e}")

events, odds_error = odds_feed()
if odds_error:
    st.warning(odds_error)

lookup = {
    (norm(e.get("away_team", "")), norm(e.get("home_team", ""))): e
    for e in events
}

rows = []
for g in games:
    try:
        hr, ar, meta = projection(g)
        hw, aw, over = simulate(hr, ar, sims, int(g["gamePk"] or 1))
    except Exception as ex:
        continue

    event = lookup.get((norm(g["away"]), norm(g["home"])))
    if not event:
        continue

    for team, (price, book) in best_moneylines(event):
        model_p = hw if norm(team) == norm(g["home"]) else aw
        market_p = implied_prob(price)
        edge = (model_p - market_p) * 100
        data_q = 84 if g["hp_id"] and g["ap_id"] else 58
        sc = edge_score(edge, data_q, 78, 88, 72)
        rows.append({
            "game": f'{g["away"]} @ {g["home"]}',
            "selection": f"{team} ML",
            "book": book,
            "odds": price,
            "model_prob": model_p,
            "market_prob": market_p,
            "edge": edge,
            "fair": fair_price(model_p),
            "ev": ev_pct(model_p, price),
            "score": sc,
            "pitchers": f'{g["ap"] or "TBD"} / {g["hp"] or "TBD"}',
            "weather": meta["weather"]
        })

df = pd.DataFrame(rows)
if not df.empty:
    candidates = df[(df.edge >= min_edge) & (df.score >= min_score)].sort_values(
        ["score", "edge"], ascending=False
    )
else:
    candidates = pd.DataFrame()

c1, c2, c3, c4 = st.columns(4)
c1.metric("MLB games", len(games))
c2.metric("Markets analyzed", len(df))
c3.metric("Best edge", f"{candidates.edge.max():.1f}%" if not candidates.empty else "—")
c4.metric("Best score", int(candidates.score.max()) if not candidates.empty else "—")

st.subheader("🔥 Today's strongest MLB edges")

if candidates.empty:
    st.success("NO BET — no current moneyline market clears your selected thresholds.")
else:
    for _, r in candidates.head(10).iterrows():
        with st.container(border=True):
            a, b, c = st.columns([4, 1, 1])
            a.markdown(f"### {r.selection} — {r.game}")
            b.metric("EDGE", f"{r.edge:+.1f}%")
            c.metric("SCORE", int(r.score))
            st.write(
                f'**Best price:** {r.odds:+g} at {r.book} · '
                f'**Model:** {r.model_prob*100:.1f}% · '
                f'**Market:** {r.market_prob*100:.1f}% · '
                f'**Fair:** {r.fair:+d} · **EV:** {r.ev:+.1f}%'
            )
            st.write(f'**Pitchers:** {r.pitchers} · **Weather:** {r.weather}')
            st.caption("Signal only; not a guarantee of profit. Model coefficients are not yet historically calibrated.")

st.subheader("📋 Full market board")
if not df.empty:
    board = df.copy()
    board["model_prob"] = (board["model_prob"]*100).round(1).astype(str) + "%"
    board["market_prob"] = (board["market_prob"]*100).round(1).astype(str) + "%"
    board["edge"] = board["edge"].round(1).astype(str) + "%"
    board["ev"] = board["ev"].round(1).astype(str) + "%"
    st.dataframe(board, use_container_width=True, hide_index=True)
else:
    st.info("No matching live MLB odds were returned.")

st.subheader("💬 EDGE Analyst")
question = st.text_input("Ask", "What are the strongest MLB bets today?")
if question:
    if candidates.empty:
        st.info("The model does not identify a qualifying bet under the current thresholds.")
    else:
        r = candidates.iloc[0]
        st.info(
            f"Top signal: {r.selection} at {r.odds:+g}. "
            f"Model probability {r.model_prob*100:.1f}% vs market {r.market_prob*100:.1f}%, "
            f"estimated edge {r.edge:+.1f} percentage points, fair price {r.fair:+d}."
        )

with st.expander("🔎 Data / model audit"):
    st.write(f"MLB games found: {len(games)}")
    st.write(f"Odds events returned: {len(events)}")
    st.write(f"Simulations per game: {sims}")
    st.write("Data sources: MLB Stats API · The Odds API · Open-Meteo")
    st.warning("Before real-money use, this model needs historical backtesting, probability calibration, closing-line-value tracking, and out-of-sample validation.")
