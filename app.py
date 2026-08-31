
from datetime import datetime, timezone, timedelta
from pathlib import Path
import math, re

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="EDGE Final", page_icon="🎯", layout="wide")

ODDS_BASE = "https://api.the-odds-api.com/v4"
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports"
DATA_FILE = Path("data/market_snapshots.csv")

def secret(name, default=""):
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default

API_KEY = secret("THE_ODDS_API_KEY", "")

def clamp(x, lo, hi): return max(lo, min(hi, x))

def american_to_prob(price):
    try: p=float(price)
    except: return np.nan
    if p < 0: return (-p)/((-p)+100)
    if p > 0: return 100/(p+100)
    return np.nan

def prob_to_american(p):
    p=clamp(float(p),.001,.999)
    if p>=.5: return int(round(-100*p/(1-p)))
    return int(round(100*(1-p)/p))

def ev_per_unit(p, price):
    try: price=float(price)
    except: return np.nan
    profit=price/100 if price>0 else 100/abs(price)
    return p*profit-(1-p)

def fmt_odds(x):
    if pd.isna(x): return "—"
    x=int(round(float(x)))
    return f"+{x}" if x>0 else str(x)

def norm_name(s): return re.sub(r"[^a-z0-9]","",str(s).lower())

@st.cache_data(ttl=180)
def get_odds(sport_key, markets):
    if not API_KEY:
        return [], {"error":"THE_ODDS_API_KEY is missing from Streamlit Secrets."}
    r=requests.get(f"{ODDS_BASE}/sports/{sport_key}/odds",params={
        "apiKey":API_KEY,"regions":"us","markets":markets,
        "oddsFormat":"american","dateFormat":"iso"
    },timeout=30)
    meta={"status":r.status_code,
          "remaining":r.headers.get("x-requests-remaining"),
          "used":r.headers.get("x-requests-used")}
    if r.status_code!=200:
        meta["error"]=r.text[:500]
        return [],meta
    return r.json(),meta

@st.cache_data(ttl=3600)
def get_sports():
    if not API_KEY: return []
    r=requests.get(f"{ODDS_BASE}/sports",params={"apiKey":API_KEY,"all":"true"},timeout=20)
    return r.json() if r.status_code==200 else []

@st.cache_data(ttl=1800)
def espn_scoreboard(league, dates=None, limit=1000):
    r=requests.get(f"{ESPN_BASE}/football/{league}/scoreboard",
                   params={"limit":limit, **({"dates":dates} if dates else {})}, timeout=30)
    r.raise_for_status()
    return r.json()

def season_games(league, year, season_type=2):
    data=espn_scoreboard(league,str(year),1000)
    rows=[]
    for ev in data.get("events",[]):
        if ev.get("season",{}).get("type") != season_type: continue
        comp=(ev.get("competitions") or [{}])[0]
        if not comp.get("status",{}).get("type",{}).get("completed"): continue
        cs=comp.get("competitors",[])
        if len(cs)!=2: continue
        h=next((x for x in cs if x.get("homeAway")=="home"),None)
        a=next((x for x in cs if x.get("homeAway")=="away"),None)
        if not h or not a: continue
        try: hs,as_=float(h.get("score")),float(a.get("score"))
        except: continue
        rows.append({"home":h["team"]["displayName"],"away":a["team"]["displayName"],
                     "home_score":hs,"away_score":as_})
    return pd.DataFrame(rows)

def srs(df):
    if df.empty: return {},{}
    teams=sorted(set(df.home)|set(df.away))
    rating={t:0. for t in teams}
    for _ in range(12):
        new={}
        for t in teams:
            vals=[]
            for r in df.itertuples():
                if r.home==t: vals.append((r.home_score-r.away_score)+rating.get(r.away,0))
                elif r.away==t: vals.append((r.away_score-r.home_score)+rating.get(r.home,0))
            new[t]=float(np.mean(vals)) if vals else 0.
        m=np.mean(list(new.values())) if new else 0
        rating={k:v-m for k,v in new.items()}
    stats={}
    for t in teams:
        pf=[]; pa=[]; n=0
        for r in df.itertuples():
            if r.home==t: x,y=r.home_score,r.away_score
            elif r.away==t: x,y=r.away_score,r.home_score
            else: continue
            pf.append(x); pa.append(y); n+=1
        stats[t]={"games":n,"pf":float(np.mean(pf)) if pf else np.nan,
                  "pa":float(np.mean(pa)) if pa else np.nan}
    return rating,stats

@st.cache_data(ttl=1800)
def blended_ratings(league,year):
    prev_r,prev_s=srs(season_games(league,year-1,2))
    cur_r,cur_s=srs(season_games(league,year,2))
    teams=sorted(set(prev_r)|set(cur_r))
    out_r={}; out_s={}
    for t in teams:
        n=cur_s.get(t,{}).get("games",0)
        w=clamp(n/8,0,1)
        out_r[t]=(1-w)*prev_r.get(t,0)+w*cur_r.get(t,prev_r.get(t,0))
        ps=prev_s.get(t,{}); cs=cur_s.get(t,{})
        out_s[t]={
            "games":n,
            "prior_games":ps.get("games",0),
            "pf":(1-w)*ps.get("pf",0)+w*cs.get("pf",ps.get("pf",0)),
            "pa":(1-w)*ps.get("pa",0)+w*cs.get("pa",ps.get("pa",0)),
        }
    return out_r,out_s

@st.cache_data(ttl=900)
def event_stage_map(league, dates):
    out={}
    for d in sorted(set(dates)):
        try: data=espn_scoreboard(league,d.strftime("%Y%m%d"),100)
        except: continue
        for ev in data.get("events",[]):
            comp=(ev.get("competitions") or [{}])[0]
            cs=comp.get("competitors",[])
            if len(cs)!=2: continue
            h=next((x for x in cs if x.get("homeAway")=="home"),None)
            a=next((x for x in cs if x.get("homeAway")=="away"),None)
            if not h or not a: continue
            typ=ev.get("season",{}).get("type")
            stage={1:"PRESEASON",2:"REGULAR",3:"POSTSEASON"}.get(typ,"UNKNOWN")
            out[(norm_name(a["team"]["displayName"]),norm_name(h["team"]["displayName"]),d.isoformat())]=stage
    return out

def next_slate(events,league):
    now=datetime.now(timezone.utc)
    parsed=[]
    for e in events:
        try: dt=datetime.fromisoformat(e.get("commence_time","").replace("Z","+00:00"))
        except: continue
        if dt>=now-timedelta(hours=2): parsed.append((dt,e))
    if not parsed: return [],[],"No upcoming games"
    meta=event_stage_map(league,[dt.date() for dt,_ in parsed])
    reg=[]; pre=[]
    for dt,e in parsed:
        stage=meta.get((norm_name(e.get("away_team")),norm_name(e.get("home_team")),dt.date().isoformat()),"UNKNOWN")
        e=dict(e); e["_stage"]=stage
        (pre if stage=="PRESEASON" else reg).append((dt,e))
    pool=reg if reg else pre
    first=min(dt for dt,_ in pool)
    end=first+timedelta(days=7)
    return [e for dt,e in pool if dt<=end],[e for _,e in pre],f"{first:%b %d} – {end:%b %d}"

def h2h_book_data(event):
    rows=[]
    for b in event.get("bookmakers",[]):
        m=next((m for m in b.get("markets",[]) if m.get("key")=="h2h"),None)
        if not m: continue
        outs={o.get("name"):o.get("price") for o in m.get("outcomes",[]) if o.get("price") is not None}
        h,a=event.get("home_team"),event.get("away_team")
        if h in outs and a in outs:
            ph,pa=american_to_prob(outs[h]),american_to_prob(outs[a])
            s=ph+pa
            if s>0:
                rows.append({"book":b.get("title",b.get("key","")),"home_price":outs[h],"away_price":outs[a],
                             "home_novig":ph/s,"away_novig":pa/s})
    return rows

def consensus_h2h(event):
    rows=h2h_book_data(event)
    if not rows: return None
    h,a=event["home_team"],event["away_team"]
    hp=np.median([r["home_novig"] for r in rows]); ap=1-hp
    best_h=max(rows,key=lambda r:r["home_price"])
    best_a=max(rows,key=lambda r:r["away_price"])
    dispersion=float(np.std([r["home_novig"] for r in rows])) if len(rows)>1 else .03
    return {"home_p":hp,"away_p":ap,
            "home_best":best_h,"away_best":best_a,
            "books":len(rows),"dispersion":dispersion}

def football_lean(event,league,ratings,stats):
    h,a=event["home_team"],event["away_team"]
    rh,ra=ratings.get(h,0),ratings.get(a,0)
    hfa=2.0 if league=="nfl" else 2.5
    margin=rh-ra+hfa
    scale=13.5 if league=="nfl" else 17.
    raw=0.5*(1+math.erf((margin/scale)/math.sqrt(2)))
    nh=stats.get(h,{}).get("games",0); na=stats.get(a,{}).get("games",0)
    maturity=min(nh,na)
    return raw,margin,maturity

def final_probability(market_p, model_p, maturity, dispersion):
    # Market is the anchor. The research model may only move probability modestly.
    # This prevents unvalidated model outputs from creating unrealistic 15–25 point "edges".
    model_weight=0.10 if maturity<3 else (0.18 if maturity<6 else 0.25)
    if dispersion>.035: model_weight*=0.75
    delta=clamp(model_p-market_p,-0.06,0.06)
    return clamp(market_p + model_weight*delta, .03, .97)

def signal_label(value_ev, books, dispersion, maturity, model_agrees):
    # Actionability is based primarily on price shopping vs consensus, not a giant model claim.
    if books < 4 or dispersion > .055:
        return "PASS"
    if value_ev >= .035 and model_agrees and maturity >= 2:
        return "BET CANDIDATE"
    if value_ev >= .015:
        return "WATCH"
    return "PASS"

def confidence_score(books,dispersion,maturity,model_agrees,value_ev):
    score=50
    score += min(15,books*2)
    score += min(15,maturity*2)
    score += 8 if model_agrees else -5
    score += min(10,max(0,value_ev)*100)
    score -= min(15,dispersion*250)
    return int(clamp(round(score),35,95))

def football_board(sport_key,label,league):
    st.subheader(f"🏈 {label}")
    odds,meta=get_odds(sport_key,"h2h,spreads,totals")
    if meta.get("error"):
        st.error(meta["error"]); return
    slate,pre,slate_label=next_slate(odds,league)
    if not slate:
        st.info("No upcoming slate found."); return

    ratings,stats=blended_ratings(league,datetime.now(timezone.utc).year)
    st.caption(f"Main board: {slate_label}. Preseason is excluded from actionable signals.")
    rows=[]
    for e in slate:
        c=consensus_h2h(e)
        if not c: continue
        model_home,margin,maturity=football_lean(e,league,ratings,stats)
        for side,market_p,model_p,best in [
            (e["home_team"],c["home_p"],model_home,c["home_best"]),
            (e["away_team"],c["away_p"],1-model_home,c["away_best"])
        ]:
            price=best["home_price"] if side==e["home_team"] else best["away_price"]
            p_final=final_probability(market_p,model_p,maturity,c["dispersion"])
            price_ev=ev_per_unit(market_p,price)   # exact cross-book value vs consensus
            adj_ev=ev_per_unit(p_final,price)      # conservative research-adjusted value
            agrees=(model_p-market_p)>0
            sig=signal_label(price_ev,c["books"],c["dispersion"],maturity,agrees)
            conf=confidence_score(c["books"],c["dispersion"],maturity,agrees,price_ev)
            rows.append({
                "Game":f'{e["away_team"]} @ {e["home_team"]}',
                "Signal":sig,
                "Pick":f"{side} ML",
                "Best odds":fmt_odds(price),
                "Book":best["book"],
                "Market fair %":market_p*100,
                "Final %":p_final*100,
                "Market-value EV %":price_ev*100,
                "Conservative EV %":adj_ev*100,
                "Fair odds":fmt_odds(prob_to_american(p_final)),
                "Model lean":("supports" if agrees else "opposes"),
                "Model margin":margin if side==e["home_team"] else -margin,
                "Books":c["books"],
                "Market dispersion":c["dispersion"]*100,
                "Season games":maturity,
                "Confidence":conf,
                "Start":e.get("commence_time")
            })
    df=pd.DataFrame(rows)
    if df.empty:
        st.info("No analyzable two-way moneylines."); return
    order={"BET CANDIDATE":0,"WATCH":1,"PASS":2}
    df["_o"]=df.Signal.map(order).fillna(3)
    df=df.sort_values(["_o","Market-value EV %","Confidence"],ascending=[True,False,False]).drop(columns="_o")

    c1,c2,c3,c4=st.columns(4)
    c1.metric("Games",df.Game.nunique())
    c2.metric("Bet candidates",(df.Signal=="BET CANDIDATE").sum())
    c3.metric("Watch",(df.Signal=="WATCH").sum())
    c4.metric("API remaining",meta.get("remaining") or "—")

    st.markdown("### Action board")
    for _,r in df.head(16).iterrows():
        badge="🟢" if r.Signal=="BET CANDIDATE" else ("🟡" if r.Signal=="WATCH" else "⚪")
        st.markdown(
            f'**{badge} {r.Signal} — {r.Pick} {r["Best odds"]}**  \n'
            f'Market-value EV **{r["Market-value EV %"]:+.1f}%** • '
            f'Conservative EV **{r["Conservative EV %"]:+.1f}%** • '
            f'Confidence **{r.Confidence}/100**'
        )
        with st.expander(f'Details: {r.Game}'):
            st.write(f'Best book: {r.Book}')
            st.write(f'Consensus no-vig probability: {r["Market fair %"]:.1f}%')
            st.write(f'Final conservative probability: {r["Final %"]:.1f}%')
            st.write(f'Final fair odds: {r["Fair odds"]}')
            st.write(f'Research model: {r["Model lean"]} this side')
            st.write(f'Model scoring-margin lean: {r["Model margin"]:+.1f}')
            st.write(f'Books in consensus: {int(r.Books)}')
            st.write(f'Market dispersion: {r["Market dispersion"]:.1f} pp')
            st.write(f'Current regular-season games in rating blend: {int(r["Season games"])}')

    st.info("Interpretation: Market-value EV is the cleanest number here—it asks whether the best available sportsbook price is better than the no-vig consensus of the other books. The football model is only allowed to make a small, capped adjustment. This prevents unrealistic +20% probability edges.")

    with st.expander("Full table"):
        st.dataframe(df,use_container_width=True,hide_index=True)

    if pre:
        with st.expander(f"Preseason excluded ({len(pre)})"):
            st.warning("No actionable preseason signals are issued.")
            st.dataframe(pd.DataFrame([{"Game":f'{e["away_team"]} @ {e["home_team"]}',"Start":e.get("commence_time")} for e in pre]),
                         use_container_width=True,hide_index=True)

def snapshot_status():
    st.subheader("📡 Prospective Odds Database")
    if not DATA_FILE.exists():
        st.info("No market_snapshots.csv yet."); return
    try: df=pd.read_csv(DATA_FILE)
    except Exception as e: st.error(str(e)); return
    if df.empty: st.info("Snapshot file is empty."); return
    c1,c2,c3=st.columns(3)
    c1.metric("Observations",len(df))
    c2.metric("Collection times",df.snapshot_time.nunique() if "snapshot_time" in df else "—")
    c3.metric("Unique events",df.event_id.nunique() if "event_id" in df else "—")
    if "sport_key" in df:
        st.dataframe(df.groupby("sport_key").agg(observations=("event_id","size"),events=("event_id","nunique")).reset_index(),
                     use_container_width=True,hide_index=True)
    st.caption("This database is for prospective validation. It does not retroactively create historical sportsbook prices.")

def golf_board():
    st.subheader("⛳ PGA Golf")
    sports=get_sports()
    golf=[s for s in sports if str(s.get("group","")).lower()=="golf" and s.get("active")]
    if not golf:
        st.info("No current golf market is exposed by the odds feed right now.")
        return
    rows=[]
    for s in golf:
        evs,meta=get_odds(s["key"],"outrights")
        if meta.get("error"): continue
        for e in evs:
            by={}
            for b in e.get("bookmakers",[]):
                m=next((m for m in b.get("markets",[]) if m.get("key")=="outrights"),None)
                if not m: continue
                for o in m.get("outcomes",[]):
                    if o.get("price") is None: continue
                    by.setdefault(o["name"],[]).append((b.get("title",b.get("key","")),float(o["price"])))
            raw={}
            for p,vals in by.items():
                raw[p]=np.mean([american_to_prob(v) for _,v in vals])
            total=sum(raw.values()) or 1
            for p,imp in raw.items():
                market_p=imp/total
                best_book,best_price=max(by[p],key=lambda x:x[1])
                value_ev=ev_per_unit(market_p,best_price)
                rows.append({"Tournament":s.get("title",s["key"]),"Player":p,
                             "Consensus fair %":market_p*100,
                             "Best odds":fmt_odds(best_price),"Book":best_book,
                             "Price value EV %":value_ev*100,
                             "Books":len(by[p]),
                             "Signal":"WATCH" if value_ev>=.02 and len(by[p])>=4 else "PASS"})
    if rows:
        df=pd.DataFrame(rows).sort_values(["Signal","Price value EV %"],ascending=[True,False])
        st.dataframe(df,use_container_width=True,hide_index=True)
        st.info("Golf signals are price-shopping signals versus no-vig consensus only. EDGE does not claim an independent player-performance edge until a validated strokes-gained/course-fit model is available.")
    else:
        st.info("No current outright market returned.")

st.title("🎯 EDGE — Final Conservative Multi-Sport Model")
st.caption("Market-anchored probabilities • best-price value • capped model adjustments • prospective validation")

sport=st.sidebar.radio("Sport",["NFL","NCAA Football","PGA Golf","Prospective database"])
st.sidebar.markdown("---")
st.sidebar.caption("A green label means a price is attractive versus current market consensus and the research lean agrees. It is not a guarantee of profit.")

if sport=="NFL":
    football_board("americanfootball_nfl","NFL","nfl")
elif sport=="NCAA Football":
    football_board("americanfootball_ncaaf","NCAA Football","college-football")
elif sport=="PGA Golf":
    golf_board()
else:
    snapshot_status()
