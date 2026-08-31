
import os, math, random
from datetime import date, datetime, timedelta, timezone
from collections import defaultdict

import requests
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

st.set_page_config(page_title="EDGE v4.1", page_icon="📈", layout="wide")

MLB = "https://statsapi.mlb.com/api/v1"
ODDS_SPORT = "baseball_mlb"

def secret(name):
    value = os.getenv(name, "")
    try:
        value = st.secrets.get(name, value)
    except Exception:
        pass
    return value

@st.cache_data(ttl=300)
def get_json(url, params=None):
    r = requests.get(url, params=params or {}, timeout=30)
    r.raise_for_status()
    return r.json()

def implied_prob(odds):
    odds = float(odds)
    return 100/(odds+100) if odds >= 0 else -odds/(-odds+100)

def american_decimal(odds):
    odds = float(odds)
    return 1 + (odds/100 if odds >= 0 else 100/abs(odds))

def no_vig(p1,p2):
    s=p1+p2
    return (p1/s,p2/s) if s>0 else (.5,.5)

def norm(s):
    return "".join(c.lower() for c in (s or "") if c.isalnum())

def clamp(x,lo,hi):
    return max(lo,min(hi,x))

@st.cache_data(ttl=300)
def schedule(d):
    return get_json(
        f"{MLB}/schedule",
        {"sportId":1,"date":d,"hydrate":"probablePitcher,team,venue"}
    ).get("dates",[])

def games_for(d):
    out=[]
    for day in schedule(d):
        for g in day.get("games",[]):
            h=g.get("teams",{}).get("home",{})
            a=g.get("teams",{}).get("away",{})
            out.append({
                "gamePk":g.get("gamePk"),
                "gameDate":g.get("gameDate"),
                "status":g.get("status",{}).get("detailedState"),
                "home":h.get("team",{}).get("name","Home"),
                "home_id":h.get("team",{}).get("id"),
                "home_score":h.get("score"),
                "away":a.get("team",{}).get("name","Away"),
                "away_id":a.get("team",{}).get("id"),
                "away_score":a.get("score"),
                "hp":(h.get("probablePitcher") or {}).get("fullName"),
                "hp_id":(h.get("probablePitcher") or {}).get("id"),
                "ap":(a.get("probablePitcher") or {}).get("fullName"),
                "ap_id":(a.get("probablePitcher") or {}).get("id"),
            })
    return out

def season_start(ds):
    return f"{datetime.fromisoformat(ds).year}-03-20"

def day_before(ds):
    return (datetime.fromisoformat(ds)-timedelta(days=1)).date().isoformat()

@st.cache_data(ttl=1800)
def pitcher_range(pid,start,end):
    if not pid:return {}
    try:
        d=get_json(f"{MLB}/people/{pid}/stats",
                   {"stats":"byDateRange","group":"pitching","startDate":start,"endDate":end})
        sp=d.get("stats",[{}])[0].get("splits") or []
        s=sp[0].get("stat",{}) if sp else {}
    except Exception:
        s={}
    return {
        "era":float(s.get("era",4.3) or 4.3),
        "whip":float(s.get("whip",1.3) or 1.3),
        "k9":float(s.get("strikeoutsPer9Inn",8.5) or 8.5),
        "bb9":float(s.get("walksPer9Inn",3.2) or 3.2),
        "hr9":float(s.get("homeRunsPer9",1.2) or 1.2),
        "innings":float(s.get("inningsPitched",0) or 0),
    }

@st.cache_data(ttl=1800)
def team_range(tid,group,start,end):
    if not tid:return {}
    try:
        d=get_json(f"{MLB}/teams/{tid}/stats",
                   {"stats":"byDateRange","group":group,"startDate":start,"endDate":end,"sportIds":1})
        sp=d.get("stats",[{}])[0].get("splits") or []
        s=sp[0].get("stat",{}) if sp else {}
    except Exception:
        s={}
    if group=="hitting":
        return {"avg":float(s.get("avg",.240) or .240),
                "obp":float(s.get("obp",.310) or .310),
                "slg":float(s.get("slg",.400) or .400),
                "ops":float(s.get("ops",.710) or .710)}
    return {"era":float(s.get("era",4.2) or 4.2),
            "whip":float(s.get("whip",1.3) or 1.3),
            "k9":float(s.get("strikeoutsPer9Inn",8.5) or 8.5)}

def pitcher_quality(s):
    if not s:return 0
    raw=((4.3-s["era"])*.16+(1.3-s["whip"])*.65+
         (s["k9"]-8.5)*.035+(3.2-s["bb9"])*.035+(1.2-s["hr9"])*.16)
    shrink=min(1.0,max(0,s.get("innings",0))/80)
    return clamp(raw*shrink,-1.25,1.25)

def offense_quality(s):
    if not s:return 0
    return clamp((s["ops"]-.710)*2.4+(s["obp"]-.310)*1.4+(s["slg"]-.400)*1.2,-1.25,1.25)

def bullpen_proxy(s):
    if not s:return 0
    return clamp((4.2-s["era"])*.10+(1.3-s["whip"])*.35+(s["k9"]-8.5)*.02,-.60,.60)

def features_for(g,ds):
    start,end=season_start(ds),day_before(ds)
    hp=pitcher_range(g["hp_id"],start,end)
    ap=pitcher_range(g["ap_id"],start,end)
    ho=team_range(g["home_id"],"hitting",start,end)
    ao=team_range(g["away_id"],"hitting",start,end)
    htp=team_range(g["home_id"],"pitching",start,end)
    atp=team_range(g["away_id"],"pitching",start,end)
    return {
        "home_off":offense_quality(ho),"away_off":offense_quality(ao),
        "home_sp":pitcher_quality(hp),"away_sp":pitcher_quality(ap),
        "home_bp":bullpen_proxy(htp),"away_bp":bullpen_proxy(atp),
        "home_field":1.0,
    }

def run_projection(f):
    hr=4.45+.20+.58*f["home_off"]-.72*f["away_sp"]-.18*f["away_bp"]+.05*f["home_sp"]
    ar=4.45+.58*f["away_off"]-.72*f["home_sp"]-.18*f["home_bp"]+.05*f["away_sp"]
    return clamp(hr,1.5,8.5),clamp(ar,1.5,8.5)

def poisson(rng,lam):
    L=math.exp(-lam); k=0; p=1.0
    while p>L:
        k+=1; p*=rng.random()
    return k-1

def simulate_ml(hr,ar,n,seed):
    rng=random.Random(seed); hw=aw=0
    for _ in range(n):
        h,a=poisson(rng,hr),poisson(rng,ar)
        hw+=h>a; aw+=a>h
    ties=max(0,n-hw-aw)
    return (hw+ties/2)/n,(aw+ties/2)/n

def historical_snapshot_for_game(game_date_utc, minutes_before=90):
    dt=datetime.fromisoformat(game_date_utc.replace("Z","+00:00"))
    snap=dt-timedelta(minutes=minutes_before)
    return snap.astimezone(timezone.utc).isoformat().replace("+00:00","Z")

@st.cache_data(ttl=3600)
def historical_odds(snapshot_iso):
    key=secret("THE_ODDS_API_KEY")
    if not key:
        return [], {"ok":False,"message":"THE_ODDS_API_KEY is missing."}
    try:
        r=requests.get(
            f"https://api.the-odds-api.com/v4/historical/sports/{ODDS_SPORT}/odds",
            params={"apiKey":key,"regions":"us","markets":"h2h","oddsFormat":"american","date":snapshot_iso},
            timeout=30
        )
        meta={"http_status":r.status_code,
              "remaining":r.headers.get("x-requests-remaining"),
              "used":r.headers.get("x-requests-used")}
        if r.status_code!=200:
            try: body=r.json()
            except Exception: body={"message":r.text[:500]}
            return [],{"ok":False,"message":body.get("message") or body.get("error") or str(body),**meta}
        d=r.json()
        events=d.get("data",d if isinstance(d,list) else [])
        return events,{"ok":True,"message":"Historical odds returned.",**meta}
    except Exception as e:
        return [],{"ok":False,"message":str(e)}

def moneyline_market(event):
    probs=defaultdict(list); offers=defaultdict(list)
    for book in event.get("bookmakers",[]):
        outs=None
        for m in book.get("markets",[]):
            if m.get("key")=="h2h":
                outs=m.get("outcomes",[]); break
        if not outs or len(outs)<2: continue
        pair=[o for o in outs if o.get("price") is not None][:2]
        if len(pair)<2: continue
        nv=no_vig(implied_prob(pair[0]["price"]),implied_prob(pair[1]["price"]))
        for o,p in zip(pair,nv):
            probs[o["name"]].append(p)
            offers[o["name"]].append((float(o["price"]),book.get("title","Book")))
    out={}
    for team,ps in probs.items():
        best=min(offers[team],key=lambda x: implied_prob(x[0]))
        out[team]={"consensus_prob":sum(ps)/len(ps),"best_odds":best[0],"best_book":best[1],"books":len(ps)}
    return out

def match_event(events,g):
    lookup={(norm(e.get("away_team")),norm(e.get("home_team"))):e for e in events}
    return lookup.get((norm(g["away"]),norm(g["home"])))

def diagnostic_for_date(test_date):
    gs=games_for(test_date.isoformat())
    details=[]; matched=0; tested=0; api_ok=True; msgs=[]
    for g in gs[:5]:
        if not g.get("gameDate"): continue
        snap=historical_snapshot_for_game(g["gameDate"],90)
        events,meta=historical_odds(snap)
        tested+=1
        if not meta.get("ok"):
            api_ok=False; msgs.append(meta.get("message"))
            details.append({"game":f'{g["away"]} @ {g["home"]}',"snapshot":snap,"events_returned":0,
                            "matched":False,"books":0,"api_message":meta.get("message")})
            continue
        e=match_event(events,g); books=0
        if e:
            ml=moneyline_market(e)
            books=max([m["books"] for m in ml.values()],default=0)
            matched+=1
        details.append({"game":f'{g["away"]} @ {g["home"]}',"snapshot":snap,"events_returned":len(events),
                        "matched":bool(e),"books":books,"api_message":meta.get("message")})
    return {"games":len(gs),"tested":tested,"matched":matched,"api_ok":api_ok,
            "message":"; ".join(sorted(set(msgs))) if msgs else "Historical odds diagnostic completed.",
            "details":details}

def build_dataset(start_date,end_date,sims=1000,include_hist_odds=False):
    recs=[]; days=(end_date-start_date).days+1
    prog=st.progress(0); status=st.empty()
    matched_games=0; attempted_games=0; failures=[]
    for i in range(days):
        cur=start_date+timedelta(days=i); ds=cur.isoformat()
        status.write(f"Building {ds} ({i+1}/{days})")
        try: gs=games_for(ds)
        except Exception: gs=[]
        for g in gs:
            if g.get("home_score") is None or g.get("away_score") is None: continue
            try:
                feat=features_for(g,ds); hr,ar=run_projection(feat)
                hp,ap=simulate_ml(hr,ar,sims,int(g["gamePk"] or 1))
            except Exception: continue
            base={"date":ds,"game":f'{g["away"]} @ {g["home"]}',
                  "home_team":g["home"],"away_team":g["away"],
                  "home_score":g["home_score"],"away_score":g["away_score"],**feat,
                  "proj_home_runs":hr,"proj_away_runs":ar,
                  "home_model_prob":hp,"away_model_prob":ap,
                  "home_outcome":int(g["home_score"]>g["away_score"]),
                  "away_outcome":int(g["away_score"]>g["home_score"])}
            if include_hist_odds:
                attempted_games+=1
                if not g.get("gameDate"):
                    failures.append("missing gameDate")
                else:
                    snap=historical_snapshot_for_game(g["gameDate"],90)
                    events,meta=historical_odds(snap)
                    if not meta.get("ok"):
                        failures.append(meta.get("message","historical odds error"))
                    else:
                        e=match_event(events,g)
                        if e:
                            ml=moneyline_market(e)
                            hm=next((m for t,m in ml.items() if norm(t)==norm(g["home"])),None)
                            am=next((m for t,m in ml.items() if norm(t)==norm(g["away"])),None)
                            if hm and am:
                                matched_games+=1
                                base.update({"home_market_prob":hm["consensus_prob"],"home_odds":hm["best_odds"],
                                             "home_books":hm["books"],"away_market_prob":am["consensus_prob"],
                                             "away_odds":am["best_odds"],"away_books":am["books"],
                                             "historical_snapshot":snap})
            recs.append(base)
        prog.progress((i+1)/days)
    status.empty(); prog.empty()
    df=pd.DataFrame(recs)
    diag={"attempted_games":attempted_games,"matched_games":matched_games,
          "match_rate":matched_games/attempted_games if attempted_games else None,
          "failures":failures[:10]}
    if include_hist_odds:
        if attempted_games==0:
            raise RuntimeError("Historical odds requested, but no completed games were available.")
        if matched_games==0:
            msg=failures[0] if failures else "No historical odds matched any MLB game."
            raise RuntimeError("Historical odds were requested but ZERO games matched. First diagnostic message: "+msg)
        if matched_games/attempted_games<.50:
            st.warning(f"Historical odds match rate is {matched_games}/{attempted_games} ({matched_games/attempted_games:.1%}).")
    return df,diag

def to_team_rows(df):
    rows=[]
    for _,r in df.iterrows():
        rows.append({"date":r["date"],"game":r["game"],"team":r["home_team"],"side":"home",
                     "f_off":r["home_off"]-r["away_off"],"f_sp":r["home_sp"]-r["away_sp"],
                     "f_bp":r["home_bp"]-r["away_bp"],"home":1.0,"outcome":r["home_outcome"],
                     "market_prob":r.get("home_market_prob",np.nan),"odds":r.get("home_odds",np.nan)})
        rows.append({"date":r["date"],"game":r["game"],"team":r["away_team"],"side":"away",
                     "f_off":r["away_off"]-r["home_off"],"f_sp":r["away_sp"]-r["home_sp"],
                     "f_bp":r["away_bp"]-r["home_bp"],"home":0.0,"outcome":r["away_outcome"],
                     "market_prob":r.get("away_market_prob",np.nan),"odds":r.get("away_odds",np.nan)})
    return pd.DataFrame(rows)

FEATURES=["f_off","f_sp","f_bp","home"]

def walk_forward(rows,train_days=45,test_days=14,step_days=14,calibrate=True):
    rows=rows.copy(); rows["date_dt"]=pd.to_datetime(rows["date"])
    start,end=rows["date_dt"].min(),rows["date_dt"].max()
    outs=[]; cursor=start; wid=0
    while True:
        tr0=cursor; tr1=tr0+pd.Timedelta(days=train_days-1)
        te0=tr1+pd.Timedelta(days=1); te1=te0+pd.Timedelta(days=test_days-1)
        if te1>end: break
        train=rows[(rows.date_dt>=tr0)&(rows.date_dt<=tr1)].copy()
        test=rows[(rows.date_dt>=te0)&(rows.date_dt<=te1)].copy()
        cursor+=pd.Timedelta(days=step_days)
        if len(train)<100 or len(test)<20 or train.outcome.nunique()<2: continue
        wid+=1
        model=LogisticRegression(C=1.0,solver="lbfgs")
        model.fit(train[FEATURES].values,train["outcome"].astype(int).values)
        trp=model.predict_proba(train[FEATURES].values)[:,1]
        tep=model.predict_proba(test[FEATURES].values)[:,1]
        if calibrate:
            p=np.clip(trp,.001,.999); logits=np.log(p/(1-p)).reshape(-1,1)
            cal=LogisticRegression(C=1e6,solver="lbfgs")
            cal.fit(logits,train["outcome"].astype(int).values)
            q=np.clip(tep,.001,.999); pred=cal.predict_proba(np.log(q/(1-q)).reshape(-1,1))[:,1]
        else:
            pred=tep
        tmp=test.copy(); tmp["wf_prob"]=pred; tmp["window"]=wid
        outs.append(tmp)
    return pd.concat(outs,ignore_index=True) if outs else pd.DataFrame()

def metric_summary(df,col):
    p=np.clip(df[col].values,.001,.999); y=df["outcome"].astype(int).values
    return {"n":len(df),"brier":float(brier_score_loss(y,p)),
            "logloss":float(log_loss(y,p,labels=[0,1])),
            "accuracy":float(((p>=.5).astype(int)==y).mean())}

def market_summary(df):
    d=df.dropna(subset=["market_prob"])
    return metric_summary(d,"market_prob") if not d.empty else None

def roi_table(df):
    d=df.dropna(subset=["market_prob","odds"]).copy()
    if d.empty:return pd.DataFrame()
    d["edge"]=(d["wf_prob"]-d["market_prob"])*100
    d["bet_return"]=np.where(d["outcome"]==1,d["odds"].apply(american_decimal)-1,-1.0)
    rows=[]
    for th in [1,2,3,4,5,7.5,10]:
        b=d[d.edge>=th]
        if len(b):
            rows.append({"edge_threshold":th,"bets":len(b),"win_rate":b.outcome.mean(),
                         "roi":b.bet_return.mean(),"avg_edge":b.edge.mean()})
    return pd.DataFrame(rows)

st.title("📈 EDGE v4.1 — Historical Odds Diagnostics")
st.caption("Verify historical odds first · align snapshots to first pitch · never silently build a market-free dataset")

tab_diag,tab_build,tab_walk,tab_market=st.tabs(["🩺 Odds diagnostic","🧱 Build dataset","🚶 Walk-forward","⚖️ Market benchmark"])

with tab_diag:
    test_date=st.date_input("Test one historical MLB date",date.today()-timedelta(days=7))
    if st.button("Run historical odds diagnostic",type="primary"):
        st.session_state["hist_diag"]=diagnostic_for_date(test_date)
    result=st.session_state.get("hist_diag")
    if result:
        a,b,c=st.columns(3)
        a.metric("MLB games",result["games"]); b.metric("Games tested",result["tested"]); c.metric("Matched",result["matched"])
        if result["api_ok"] and result["matched"]>0:
            st.success("Historical odds API is available and MLB games are matching.")
        elif result["api_ok"] is False:
            st.error("Historical odds API failed: "+result["message"])
        else:
            st.warning("Historical odds API responded, but no games matched.")
        st.dataframe(pd.DataFrame(result["details"]),use_container_width=True,hide_index=True)

with tab_build:
    c1,c2,c3=st.columns(3)
    start=c1.date_input("Start",date.today()-timedelta(days=90),key="build_start")
    end=c2.date_input("End",date.today()-timedelta(days=1),key="build_end")
    sims=c3.selectbox("Simulations/game",[500,1000,1500,3000],index=1)
    include_hist=st.checkbox("Include historical sportsbook odds (STOP if zero match)",value=True)
    if st.button("Build dataset",type="primary"):
        try:
            ds,diag=build_dataset(start,end,sims,include_hist)
            st.session_state["dataset"]=ds; st.session_state["build_diag"]=diag
            st.success("Dataset built successfully.")
        except Exception as e:
            st.error(str(e))
    if st.session_state.get("build_diag"):
        st.json(st.session_state["build_diag"])
    ds=st.session_state.get("dataset")
    if isinstance(ds,pd.DataFrame) and not ds.empty:
        st.dataframe(ds.head(20),use_container_width=True,hide_index=True)
        st.download_button("Download dataset CSV",ds.to_csv(index=False).encode(),
                           file_name=f"EDGE_v4_1_dataset_{start}_{end}.csv",mime="text/csv")

with tab_walk:
    uploaded=st.file_uploader("Upload an EDGE v4.1 dataset CSV",type=["csv"])
    if uploaded:
        st.session_state["dataset"]=pd.read_csv(uploaded)
    ds=st.session_state.get("dataset")
    if not isinstance(ds,pd.DataFrame) or ds.empty:
        st.info("Build or upload a dataset first.")
    else:
        rows=to_team_rows(ds)
        c1,c2,c3=st.columns(3)
        train_days=c1.selectbox("Training days",[30,45,60,90],index=1)
        test_days=c2.selectbox("Unseen test days",[7,14,21,30],index=1)
        step_days=c3.selectbox("Walk step",[7,14,21,30],index=1)
        calibrate=st.checkbox("Calibrate on training only",value=True)
        if st.button("Run walk-forward",type="primary"):
            st.session_state["wf"]=walk_forward(rows,train_days,test_days,step_days,calibrate)
        wf=st.session_state.get("wf")
        if isinstance(wf,pd.DataFrame) and not wf.empty:
            m=metric_summary(wf,"wf_prob")
            a,b,c,d=st.columns(4)
            a.metric("Unseen predictions",m["n"]); b.metric("Brier",f'{m["brier"]:.4f}')
            c.metric("Log loss",f'{m["logloss"]:.4f}'); d.metric("Accuracy",f'{m["accuracy"]*100:.1f}%')

with tab_market:
    wf=st.session_state.get("wf")
    if not isinstance(wf,pd.DataFrame) or wf.empty:
        st.info("Run walk-forward validation first.")
    else:
        edge_m=metric_summary(wf,"wf_prob"); market_m=market_summary(wf)
        if market_m is None:
            st.error("No historical market probabilities are present. Use Odds Diagnostic, then rebuild with historical odds enabled.")
        else:
            c1,c2,c3,c4=st.columns(4)
            c1.metric("EDGE Brier",f'{edge_m["brier"]:.4f}'); c2.metric("Market Brier",f'{market_m["brier"]:.4f}')
            c3.metric("EDGE log loss",f'{edge_m["logloss"]:.4f}'); c4.metric("Market log loss",f'{market_m["logloss"]:.4f}')
            if edge_m["brier"]<market_m["brier"]:
                st.success("EDGE beats market Brier in this unseen sample.")
            else:
                st.warning("EDGE does not beat market Brier in this unseen sample.")
            rt=roi_table(wf)
            if not rt.empty:
                show=rt.copy()
                show["win_rate"]=(show.win_rate*100).round(1).astype(str)+"%"
                show["roi"]=(show.roi*100).round(2).astype(str)+"%"
                show["avg_edge"]=show.avg_edge.round(2).astype(str)+"%"
                st.dataframe(show,use_container_width=True,hide_index=True)

st.caption("EDGE v4.1 — historical odds diagnostics and market-benchmark safeguards")
