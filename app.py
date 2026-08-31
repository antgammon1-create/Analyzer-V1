
import math
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="EDGE v5 Multi-Sport", page_icon="📊", layout="wide")

ODDS_BASE = "https://api.the-odds-api.com/v4"
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports"
DATA_FILE = Path("data/market_snapshots.csv")

def secret(name, default=""):
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default

API_KEY = secret("THE_ODDS_API_KEY", "")

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def american_to_prob(price):
    try:
        p = float(price)
    except Exception:
        return np.nan
    if p < 0:
        return (-p) / ((-p) + 100.0)
    if p > 0:
        return 100.0 / (p + 100.0)
    return np.nan

def prob_to_american(p):
    p = clamp(float(p), 0.001, 0.999)
    if p >= 0.5:
        return int(round(-100.0 * p / (1.0 - p)))
    return int(round(100.0 * (1.0 - p) / p))

def ev_per_unit(p, price):
    try:
        price = float(price)
    except Exception:
        return np.nan
    profit = price / 100.0 if price > 0 else 100.0 / abs(price)
    return p * profit - (1.0 - p)

def fmt_odds(x):
    if pd.isna(x):
        return "—"
    x = int(round(float(x)))
    return f"+{x}" if x > 0 else str(x)

def norm_name(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

@st.cache_data(ttl=180)
def get_odds(sport_key, markets="h2h,spreads,totals"):
    if not API_KEY:
        return [], {"error": "THE_ODDS_API_KEY is missing from Streamlit Secrets."}
    url = f"{ODDS_BASE}/sports/{sport_key}/odds"
    params = {
        "apiKey": API_KEY,
        "regions": "us",
        "markets": markets,
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    r = requests.get(url, params=params, timeout=25)
    meta = {
        "status": r.status_code,
        "remaining": r.headers.get("x-requests-remaining"),
        "used": r.headers.get("x-requests-used"),
    }
    if r.status_code != 200:
        meta["error"] = r.text[:500]
        return [], meta
    return r.json(), meta

@st.cache_data(ttl=3600)
def get_sports():
    if not API_KEY:
        return []
    r = requests.get(f"{ODDS_BASE}/sports", params={"apiKey": API_KEY, "all": "true"}, timeout=20)
    return r.json() if r.status_code == 200 else []

@st.cache_data(ttl=1800)
def espn_scoreboard(league, dates=None, limit=1000):
    url = f"{ESPN_BASE}/football/{league}/scoreboard"
    params = {"limit": limit}
    if dates:
        params["dates"] = dates
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def season_games(league, year):
    data = espn_scoreboard(league, str(year), 1000)
    rows = []
    for ev in data.get("events", []):
        comp = (ev.get("competitions") or [{}])[0]
        status = comp.get("status", {}).get("type", {})
        if not status.get("completed"):
            continue
        competitors = comp.get("competitors", [])
        if len(competitors) != 2:
            continue
        home = next((x for x in competitors if x.get("homeAway") == "home"), None)
        away = next((x for x in competitors if x.get("homeAway") == "away"), None)
        if not home or not away:
            continue
        try:
            hs, as_ = float(home.get("score")), float(away.get("score"))
        except Exception:
            continue
        rows.append({
            "date": ev.get("date"),
            "home": home.get("team", {}).get("displayName"),
            "away": away.get("team", {}).get("displayName"),
            "home_score": hs,
            "away_score": as_,
        })
    return pd.DataFrame(rows)

@st.cache_data(ttl=1800)
def team_ratings(league, year):
    df = season_games(league, year)
    if df.empty:
        return {}, {}
    teams = sorted(set(df.home) | set(df.away))
    rating = {t: 0.0 for t in teams}
    # Simple iterative SRS: average adjusted scoring margin.
    for _ in range(12):
        new = {}
        for t in teams:
            vals = []
            for r in df.itertuples():
                if r.home == t:
                    vals.append((r.home_score-r.away_score) + rating.get(r.away, 0))
                elif r.away == t:
                    vals.append((r.away_score-r.home_score) + rating.get(r.home, 0))
            new[t] = float(np.mean(vals)) if vals else 0.0
        mean_r = float(np.mean(list(new.values()))) if new else 0.0
        rating = {k: v-mean_r for k,v in new.items()}
    stats = {}
    for t in teams:
        pts_for, pts_against, wins, n = [], [], 0, 0
        for r in df.itertuples():
            if r.home == t:
                pf, pa = r.home_score, r.away_score
            elif r.away == t:
                pf, pa = r.away_score, r.home_score
            else:
                continue
            pts_for.append(pf); pts_against.append(pa); n += 1; wins += int(pf > pa)
        stats[t] = {
            "games": n,
            "pf": float(np.mean(pts_for)) if pts_for else np.nan,
            "pa": float(np.mean(pts_against)) if pts_against else np.nan,
            "win_pct": wins/n if n else 0.5,
        }
    return rating, stats

def consensus_market(event, market_key):
    books = event.get("bookmakers", [])
    outcomes_by_key = {}
    best = {}
    book_count = 0
    for b in books:
        market = next((m for m in b.get("markets", []) if m.get("key") == market_key), None)
        if not market:
            continue
        book_count += 1
        for o in market.get("outcomes", []):
            name = o.get("name")
            point = o.get("point")
            key = (name, point)
            price = o.get("price")
            if price is None:
                continue
            outcomes_by_key.setdefault(key, []).append(american_to_prob(price))
            old = best.get(key)
            if old is None or float(price) > old["price"]:
                best[key] = {"price": float(price), "book": b.get("title", b.get("key",""))}
    return outcomes_by_key, best, book_count

def h2h_consensus(event):
    probs, best, nbooks = consensus_market(event, "h2h")
    home, away = event.get("home_team"), event.get("away_team")
    hp = [p for (n,_), arr in probs.items() if n == home for p in arr]
    ap = [p for (n,_), arr in probs.items() if n == away for p in arr]
    if not hp or not ap:
        return None
    raw_h, raw_a = float(np.mean(hp)), float(np.mean(ap))
    s = raw_h + raw_a
    return {
        "home_market": raw_h/s,
        "away_market": raw_a/s,
        "home_best": best.get((home, None), {}),
        "away_best": best.get((away, None), {}),
        "books": nbooks
    }

def normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def football_model(event, league, ratings, stats):
    home, away = event["home_team"], event["away_team"]
    rh, ra = ratings.get(home, 0.0), ratings.get(away, 0.0)
    hfa = 2.0 if league == "nfl" else 2.5
    scale = 13.5 if league == "nfl" else 17.0
    expected_margin = rh - ra + hfa
    p_home = normal_cdf(expected_margin / scale)

    sh, sa = stats.get(home, {}), stats.get(away, {})
    league_total = 44.5 if league == "nfl" else 55.0
    vals = []
    if sh:
        vals.extend([sh.get("pf"), sh.get("pa")])
    if sa:
        vals.extend([sa.get("pf"), sa.get("pa")])
    vals = [v for v in vals if v is not None and not pd.isna(v)]
    expected_total = float(np.mean(vals))*2 if vals else league_total
    expected_total = 0.65*expected_total + 0.35*league_total

    games_h = sh.get("games", 0) if sh else 0
    games_a = sa.get("games", 0) if sa else 0
    min_games = min(games_h, games_a)
    quality = int(clamp(45 + min_games*6, 45, 92))
    reliability = "HIGH" if min_games >= 8 else ("MEDIUM" if min_games >= 4 else "LOW")
    return expected_margin, expected_total, p_home, quality, reliability, games_h, games_a

def classify(edge, ev, quality, reliability):
    if quality < 65 or reliability == "LOW":
        return "PASS"
    if edge >= 0.075 and ev >= 0.08 and quality >= 80:
        return "BET CANDIDATE"
    if edge >= 0.05 and ev >= 0.05:
        return "WATCH"
    return "PASS"

def analyze_football(sport_key, league_label, espn_league):
    st.subheader(f"🏈 {league_label} — Current Betting Board")
    odds, meta = get_odds(sport_key)
    if meta.get("error"):
        st.error(meta["error"])
        return
    if not odds:
        st.info("No current games/odds were returned.")
        return
    now = datetime.now(timezone.utc)
    year = now.year
    ratings, stats = team_ratings(espn_league, year)
    rows = []
    for ev in odds:
        c = h2h_consensus(ev)
        if not c:
            continue
        margin, total, p_home, quality, reliability, gh, ga = football_model(ev, espn_league, ratings, stats)
        for side, p_model, p_market, best in [
            (ev["home_team"], p_home, c["home_market"], c["home_best"]),
            (ev["away_team"], 1-p_home, c["away_market"], c["away_best"]),
        ]:
            price = best.get("price", np.nan)
            edge = p_model-p_market
            evu = ev_per_unit(p_model, price)
            rows.append({
                "Game": f'{ev["away_team"]} @ {ev["home_team"]}',
                "Pick": f"{side} ML",
                "Model %": p_model*100,
                "Market %": p_market*100,
                "Edge %": edge*100,
                "Best odds": fmt_odds(price),
                "Book": best.get("book","—"),
                "EV %": evu*100 if not pd.isna(evu) else np.nan,
                "Fair odds": fmt_odds(prob_to_american(p_model)),
                "Model margin": margin if side == ev["home_team"] else -margin,
                "Model total": total,
                "Data quality": quality,
                "Reliability": reliability,
                "Books": c["books"],
                "Signal": classify(edge, evu if not pd.isna(evu) else -1, quality, reliability),
                "Start": ev.get("commence_time"),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        st.info("No two-way moneyline markets could be analyzed.")
        return
    order = {"BET CANDIDATE":0, "WATCH":1, "PASS":2}
    df["_o"] = df["Signal"].map(order).fillna(3)
    df = df.sort_values(["_o","Edge %"], ascending=[True,False]).drop(columns="_o")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Games", df["Game"].nunique())
    c2.metric("Bet candidates", int((df.Signal=="BET CANDIDATE").sum()))
    c3.metric("Watch", int((df.Signal=="WATCH").sum()))
    c4.metric("API remaining", meta.get("remaining") or "—")
    st.dataframe(df, use_container_width=True, hide_index=True,
                 column_config={
                     "Model %": st.column_config.NumberColumn(format="%.1f"),
                     "Market %": st.column_config.NumberColumn(format="%.1f"),
                     "Edge %": st.column_config.NumberColumn(format="%+.1f"),
                     "EV %": st.column_config.NumberColumn(format="%+.1f"),
                     "Model margin": st.column_config.NumberColumn(format="%+.1f"),
                     "Model total": st.column_config.NumberColumn(format="%.1f"),
                 })
    st.caption("Football model = simple opponent-adjusted scoring-margin (SRS) + home field + season scoring environment. Early-season signals are intentionally downgraded. This is an operational research model, not a proven betting edge.")

    st.markdown("#### Spread & total model")
    market_rows = []
    for ev in odds:
        margin, model_total, _, quality, reliability, *_ = football_model(ev, espn_league, ratings, stats)
        for mk in ["spreads","totals"]:
            probs, best, nbooks = consensus_market(ev, mk)
            if not probs:
                continue
            # Evaluate each distinct line using best available price for that exact outcome/point.
            for (name, point), info in best.items():
                price = info["price"]
                if point is None:
                    continue
                if mk == "spreads":
                    if name not in (ev["home_team"], ev["away_team"]):
                        continue
                    model_margin_side = margin if name == ev["home_team"] else -margin
                    cover_p = normal_cdf((model_margin_side + float(point)) / (13.5 if espn_league=="nfl" else 17.0))
                    pick = f"{name} {float(point):+g}"
                else:
                    sigma = 13.0 if espn_league=="nfl" else 17.0
                    if name.lower() == "over":
                        cover_p = normal_cdf((model_total-float(point))/sigma)
                    elif name.lower() == "under":
                        cover_p = normal_cdf((float(point)-model_total)/sigma)
                    else:
                        continue
                    pick = f"{name} {float(point):g}"
                imp = american_to_prob(price)
                edge = cover_p - imp
                evu = ev_per_unit(cover_p, price)
                market_rows.append({
                    "Game": f'{ev["away_team"]} @ {ev["home_team"]}',
                    "Market": "Spread" if mk=="spreads" else "Total",
                    "Pick": pick,
                    "Model %": cover_p*100,
                    "Price implied %": imp*100,
                    "Edge %": edge*100,
                    "Best odds": fmt_odds(price),
                    "Book": info["book"],
                    "EV %": evu*100,
                    "Data quality": quality,
                    "Reliability": reliability,
                    "Signal": classify(edge, evu, quality, reliability),
                })
    if market_rows:
        mdf = pd.DataFrame(market_rows).sort_values("Edge %", ascending=False)
        st.dataframe(mdf.head(40), use_container_width=True, hide_index=True,
                     column_config={
                         "Model %": st.column_config.NumberColumn(format="%.1f"),
                         "Price implied %": st.column_config.NumberColumn(format="%.1f"),
                         "Edge %": st.column_config.NumberColumn(format="%+.1f"),
                         "EV %": st.column_config.NumberColumn(format="%+.1f"),
                     })

@st.cache_data(ttl=900)
def golf_scoreboard(year):
    url = f"{ESPN_BASE}/golf/pga/scoreboard"
    r = requests.get(url, params={"dates": str(year), "limit": 100}, timeout=30)
    r.raise_for_status()
    return r.json()

def golf_current_event(data):
    events = data.get("events", [])
    now = datetime.now(timezone.utc)
    def dt(e):
        try: return datetime.fromisoformat(e.get("date","").replace("Z","+00:00"))
        except: return now + timedelta(days=999)
    active = [e for e in events if not (e.get("status",{}).get("type",{}).get("completed"))]
    if active:
        return sorted(active, key=lambda e: abs((dt(e)-now).total_seconds()))[0]
    return sorted(events, key=lambda e: abs((dt(e)-now).total_seconds()))[0] if events else None

def parse_golf_field(event):
    comps = event.get("competitions") or []
    if not comps:
        return []
    competitors = comps[0].get("competitors") or []
    rows=[]
    for c in competitors:
        a=c.get("athlete",{})
        rows.append({
            "player": a.get("displayName"),
            "rank": a.get("rank") or a.get("position"),
            "score": c.get("score"),
            "status": c.get("status",{}).get("type",{}).get("description",""),
        })
    return [r for r in rows if r["player"]]

def golf_market_events():
    sports = get_sports()
    keys = [s["key"] for s in sports if str(s.get("group","")).lower()=="golf" and s.get("active")]
    out=[]
    for key in keys:
        evs, meta = get_odds(key, "outrights")
        if not meta.get("error"):
            for ev in evs:
                ev["_sport_key"] = key
                ev["_sport_title"] = next((s.get("title") for s in sports if s.get("key")==key), key)
                out.append(ev)
    return out

def analyze_golf():
    st.subheader("⛳ PGA Golf")
    year=datetime.now(timezone.utc).year
    try:
        data=golf_scoreboard(year)
    except Exception as e:
        st.error(f"Could not load ESPN golf data: {e}")
        return
    event=golf_current_event(data)
    if not event:
        st.info("No PGA event was returned.")
        return
    st.markdown(f"### {event.get('name','Current PGA event')}")
    st.caption(f"Start: {event.get('date','—')} • Status: {event.get('status',{}).get('type',{}).get('description','—')}")
    field=parse_golf_field(event)
    if field:
        fdf=pd.DataFrame(field)
        st.dataframe(fdf.head(40), use_container_width=True, hide_index=True)
    else:
        st.info("ESPN has not published the field/leaderboard in this event payload yet.")

    st.markdown("#### Sportsbook outright markets")
    markets=golf_market_events()
    if not markets:
        st.warning("No golf outright market is currently available from your free odds feed. The Odds API's golf coverage is mainly the four majors, so ordinary weekly PGA Tour events may have no sportsbook market here.")
        st.info("You can still use the ESPN tournament/leaderboard view above. EDGE will only show a betting-value comparison when a supported golf outright market is actually available.")
        return

    allrows=[]
    for ev in markets:
        books=ev.get("bookmakers",[])
        by_player={}
        best={}
        for b in books:
            m=next((m for m in b.get("markets",[]) if m.get("key") in ("outrights","h2h")),None)
            if not m: continue
            for o in m.get("outcomes",[]):
                name=o.get("name"); price=o.get("price")
                if not name or price is None: continue
                by_player.setdefault(name,[]).append(american_to_prob(price))
                if name not in best or float(price)>best[name]["price"]:
                    best[name]={"price":float(price),"book":b.get("title",b.get("key",""))}
        if not by_player: continue
        raw={n:float(np.mean(ps)) for n,ps in by_player.items()}
        total=sum(raw.values()) or 1
        novig={n:p/total for n,p in raw.items()}
        for n,p in sorted(novig.items(), key=lambda x:x[1], reverse=True):
            allrows.append({
                "Tournament":ev.get("_sport_title","Golf"),
                "Player":n,
                "No-vig market %":p*100,
                "Best odds":fmt_odds(best[n]["price"]),
                "Book":best[n]["book"],
                "Market fair odds":fmt_odds(prob_to_american(p)),
            })
    if allrows:
        st.dataframe(pd.DataFrame(allrows),use_container_width=True,hide_index=True,
                     column_config={"No-vig market %":st.column_config.NumberColumn(format="%.2f")})
        st.caption("Golf currently shows market consensus rather than a fake independent 'edge'. A proper PGA betting model needs player-level strokes-gained/form/course-fit inputs. We deliberately do not label market-only differences as a betting edge.")

def snapshot_status():
    st.subheader("📡 Prospective Market Database")
    if not DATA_FILE.exists():
        st.info("No snapshot database is present yet. Run the GitHub Actions collector.")
        return
    try:
        df=pd.read_csv(DATA_FILE)
    except Exception as e:
        st.error(str(e)); return
    if df.empty:
        st.info("Snapshot file exists but has no rows.")
        return
    c1,c2,c3=st.columns(3)
    c1.metric("Odds observations",len(df))
    c2.metric("Collection times",df["snapshot_time"].nunique() if "snapshot_time" in df else "—")
    c3.metric("Unique events",df["event_id"].nunique() if "event_id" in df else "—")
    if "sport_key" in df:
        st.dataframe(df.groupby("sport_key").agg(observations=("event_id","size"),events=("event_id","nunique")).reset_index(),hide_index=True,use_container_width=True)
    st.dataframe(df.tail(100),hide_index=True,use_container_width=True)

st.title("📊 EDGE v5 — Multi-Sport Betting Research")
st.caption("MLB • NFL • NCAA Football • PGA Golf | Current odds + independent football research model + prospective market collection")

sport=st.sidebar.radio("Sport",["MLB snapshot database","NFL","NCAA Football","PGA Golf"])
st.sidebar.markdown("---")
st.sidebar.caption("BET CANDIDATE is a research label, not proof of profitability. Prospective validation remains the standard before increasing stake size.")

if sport=="MLB snapshot database":
    snapshot_status()
    st.info("Your existing MLB analyzer/collector remains compatible. This v5 database view focuses on the shared prospective odds history.")
elif sport=="NFL":
    analyze_football("americanfootball_nfl","NFL","nfl")
elif sport=="NCAA Football":
    analyze_football("americanfootball_ncaaf","NCAA Football","college-football")
else:
    analyze_golf()
