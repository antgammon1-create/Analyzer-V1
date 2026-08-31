
import os, math, random
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

st.set_page_config(page_title="EDGE v4.2 Free Validation", page_icon="📈", layout="wide")

MLB = "https://statsapi.mlb.com/api/v1"
SNAPSHOT_FILE = "data/odds_snapshots.csv"

def get_json(url, params=None):
    r = requests.get(url, params=params or {}, timeout=30)
    r.raise_for_status()
    return r.json()

def norm(s):
    return "".join(c.lower() for c in (s or "") if c.isalnum())

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def implied_prob(odds):
    odds = float(odds)
    return 100/(odds+100) if odds >= 0 else -odds/(-odds+100)

def american_decimal(odds):
    odds = float(odds)
    return 1 + (odds/100 if odds >= 0 else 100/abs(odds))

def season_start(ds):
    return f"{datetime.fromisoformat(ds).year}-03-20"

def day_before(ds):
    return (datetime.fromisoformat(ds)-timedelta(days=1)).date().isoformat()

@st.cache_data(ttl=300)
def schedule(ds):
    return get_json(
        f"{MLB}/schedule",
        {"sportId":1, "date":ds, "hydrate":"probablePitcher,team"}
    ).get("dates", [])

def games_for(ds):
    out=[]
    for day in schedule(ds):
        for g in day.get("games",[]):
            h=g.get("teams",{}).get("home",{})
            a=g.get("teams",{}).get("away",{})
            out.append({
                "gamePk":g.get("gamePk"),
                "gameDate":g.get("gameDate"),
                "home":h.get("team",{}).get("name","Home"),
                "home_id":h.get("team",{}).get("id"),
                "home_score":h.get("score"),
                "away":a.get("team",{}).get("name","Away"),
                "away_id":a.get("team",{}).get("id"),
                "away_score":a.get("score"),
                "hp_id":(h.get("probablePitcher") or {}).get("id"),
                "ap_id":(a.get("probablePitcher") or {}).get("id"),
            })
    return out

@st.cache_data(ttl=1800)
def pitcher_range(pid,start,end):
    if not pid:return {}
    try:
        d=get_json(
            f"{MLB}/people/{pid}/stats",
            {"stats":"byDateRange","group":"pitching","startDate":start,"endDate":end}
        )
        splits=d.get("stats",[{}])[0].get("splits") or []
        s=splits[0].get("stat",{}) if splits else {}
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
def team_range(tid, group, start, end):
    if not tid:return {}
    try:
        d=get_json(
            f"{MLB}/teams/{tid}/stats",
            {"stats":"byDateRange","group":group,"startDate":start,"endDate":end,"sportIds":1}
        )
        splits=d.get("stats",[{}])[0].get("splits") or []
        s=splits[0].get("stat",{}) if splits else {}
    except Exception:
        s={}
    if group=="hitting":
        return {
            "obp":float(s.get("obp",.310) or .310),
            "slg":float(s.get("slg",.400) or .400),
            "ops":float(s.get("ops",.710) or .710),
        }
    return {
        "era":float(s.get("era",4.2) or 4.2),
        "whip":float(s.get("whip",1.3) or 1.3),
        "k9":float(s.get("strikeoutsPer9Inn",8.5) or 8.5),
    }

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

def features_for(g, ds):
    start,end=season_start(ds),day_before(ds)
    hp=pitcher_range(g["hp_id"],start,end)
    ap=pitcher_range(g["ap_id"],start,end)
    ho=team_range(g["home_id"],"hitting",start,end)
    ao=team_range(g["away_id"],"hitting",start,end)
    htp=team_range(g["home_id"],"pitching",start,end)
    atp=team_range(g["away_id"],"pitching",start,end)
    return {
        "home_off":offense_quality(ho),
        "away_off":offense_quality(ao),
        "home_sp":pitcher_quality(hp),
        "away_sp":pitcher_quality(ap),
        "home_bp":bullpen_proxy(htp),
        "away_bp":bullpen_proxy(atp),
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

def load_snapshots():
    try:
        df=pd.read_csv(SNAPSHOT_FILE)
    except Exception:
        return pd.DataFrame()
    for col in ["commence_time","snapshot_time"]:
        if col in df:
            df[col]=pd.to_datetime(df[col],utc=True,errors="coerce")
    return df

def nearest_pregame_snapshot(snaps, g, target_minutes=90):
    if snaps.empty:return None
    subset=snaps[
        (snaps.home_team.apply(norm)==norm(g["home"])) &
        (snaps.away_team.apply(norm)==norm(g["away"]))
    ].copy()
    if subset.empty:return None
    game_time=pd.to_datetime(g["gameDate"],utc=True)
    subset=subset[subset.snapshot_time < game_time]
    if subset.empty:return None
    subset["minutes_before"]=(game_time-subset.snapshot_time).dt.total_seconds()/60
    # Prefer 30-180 minutes pregame, nearest to 90.
    valid=subset[(subset.minutes_before>=30)&(subset.minutes_before<=180)]
    if valid.empty:
        valid=subset
    valid["distance"]=(valid.minutes_before-target_minutes).abs()
    return valid.sort_values(["distance","snapshot_time"]).iloc[0]

def build_prospective_dataset(start_date,end_date,sims=1500):
    snaps=load_snapshots()
    rows=[]
    days=(end_date-start_date).days+1
    prog=st.progress(0)
    status=st.empty()
    matched=0
    for i in range(days):
        cur=start_date+timedelta(days=i)
        ds=cur.isoformat()
        status.write(f"Processing {ds} ({i+1}/{days})")
        try: gs=games_for(ds)
        except Exception: gs=[]
        for g in gs:
            if g.get("home_score") is None or g.get("away_score") is None:
                continue
            snap=nearest_pregame_snapshot(snaps,g)
            if snap is None:
                continue
            try:
                feat=features_for(g,ds)
                hr,ar=run_projection(feat)
                hp,ap=simulate_ml(hr,ar,sims,int(g["gamePk"] or 1))
            except Exception:
                continue
            matched+=1
            rows.append({
                "date":ds,
                "game":f'{g["away"]} @ {g["home"]}',
                "home_team":g["home"],"away_team":g["away"],
                "home_outcome":int(g["home_score"]>g["away_score"]),
                "away_outcome":int(g["away_score"]>g["home_score"]),
                "home_model_prob":hp,"away_model_prob":ap,
                "home_market_prob":snap["home_no_vig_prob"],
                "away_market_prob":snap["away_no_vig_prob"],
                "home_odds":snap["home_best_odds"],
                "away_odds":snap["away_best_odds"],
                "snapshot_time":snap["snapshot_time"],
                "minutes_before":snap["minutes_before"],
                **feat
            })
        prog.progress((i+1)/days)
    status.empty(); prog.empty()
    return pd.DataFrame(rows), matched

def to_team_rows(df):
    rows=[]
    for _,r in df.iterrows():
        rows.append({
            "date":r["date"],"game":r["game"],"team":r["home_team"],"side":"home",
            "f_off":r["home_off"]-r["away_off"],
            "f_sp":r["home_sp"]-r["away_sp"],
            "f_bp":r["home_bp"]-r["away_bp"],
            "home":1.0,
            "outcome":r["home_outcome"],
            "market_prob":r["home_market_prob"],
            "odds":r["home_odds"],
        })
        rows.append({
            "date":r["date"],"game":r["game"],"team":r["away_team"],"side":"away",
            "f_off":r["away_off"]-r["home_off"],
            "f_sp":r["away_sp"]-r["home_sp"],
            "f_bp":r["away_bp"]-r["home_bp"],
            "home":0.0,
            "outcome":r["away_outcome"],
            "market_prob":r["away_market_prob"],
            "odds":r["away_odds"],
        })
    return pd.DataFrame(rows)

FEATURES=["f_off","f_sp","f_bp","home"]

def walk_forward(rows,train_days=45,test_days=14,step_days=14):
    rows=rows.copy()
    rows["date_dt"]=pd.to_datetime(rows["date"])
    start,end=rows.date_dt.min(),rows.date_dt.max()
    outs=[]; cursor=start; wid=0
    while True:
        tr0=cursor; tr1=tr0+pd.Timedelta(days=train_days-1)
        te0=tr1+pd.Timedelta(days=1); te1=te0+pd.Timedelta(days=test_days-1)
        if te1>end: break
        train=rows[(rows.date_dt>=tr0)&(rows.date_dt<=tr1)].copy()
        test=rows[(rows.date_dt>=te0)&(rows.date_dt<=te1)].copy()
        cursor+=pd.Timedelta(days=step_days)
        if len(train)<80 or len(test)<20 or train.outcome.nunique()<2: continue
        wid+=1
        model=LogisticRegression(C=1.0,solver="lbfgs")
        model.fit(train[FEATURES].values,train.outcome.astype(int).values)
        trp=model.predict_proba(train[FEATURES].values)[:,1]
        tep=model.predict_proba(test[FEATURES].values)[:,1]
        # Calibrate using training only.
        p=np.clip(trp,.001,.999)
        cal=LogisticRegression(C=1e6,solver="lbfgs")
        cal.fit(np.log(p/(1-p)).reshape(-1,1),train.outcome.astype(int).values)
        q=np.clip(tep,.001,.999)
        pred=cal.predict_proba(np.log(q/(1-q)).reshape(-1,1))[:,1]
        tmp=test.copy()
        tmp["wf_prob"]=pred
        tmp["window"]=wid
        outs.append(tmp)
    return pd.concat(outs,ignore_index=True) if outs else pd.DataFrame()

def metrics(df,col):
    p=np.clip(df[col].values,.001,.999)
    y=df.outcome.astype(int).values
    return {
        "brier":float(brier_score_loss(y,p)),
        "logloss":float(log_loss(y,p,labels=[0,1])),
        "accuracy":float(((p>=.5).astype(int)==y).mean())
    }

def roi_table(df):
    d=df.copy()
    d["edge"]=(d.wf_prob-d.market_prob)*100
    d["bet_return"]=np.where(d.outcome==1,d.odds.apply(american_decimal)-1,-1.0)
    rows=[]
    for th in [1,2,3,4,5,7.5,10]:
        b=d[d.edge>=th]
        if len(b):
            rows.append({
                "edge_threshold":th,
                "bets":len(b),
                "win_rate":b.outcome.mean(),
                "roi":b.bet_return.mean(),
                "avg_edge":b.edge.mean()
            })
    return pd.DataFrame(rows)

st.title("📈 EDGE v4.2 — Free Prospective Validation")
st.caption("No paid historical odds · collect current free odds from now on · build your own benchmark automatically")

tab_status,tab_validate,tab_market,tab_setup=st.tabs(
    ["📡 Snapshot status","🧪 Prospective test","⚖️ EDGE vs market","⚙️ Free setup"]
)

with tab_status:
    snaps=load_snapshots()
    if snaps.empty:
        st.warning("No odds snapshots have been collected yet.")
    else:
        st.success(f"Snapshots available: {len(snaps):,}")
        a,b,c=st.columns(3)
        a.metric("First snapshot",str(snaps.snapshot_time.min()))
        b.metric("Latest snapshot",str(snaps.snapshot_time.max()))
        c.metric("Unique games",snaps[["home_team","away_team","commence_time"]].drop_duplicates().shape[0])
        st.dataframe(snaps.tail(25),use_container_width=True,hide_index=True)

with tab_validate:
    snaps=load_snapshots()
    if snaps.empty:
        st.info("Set up the free GitHub collector first. Once it has accumulated completed games, validation will work here.")
    else:
        c1,c2,c3=st.columns(3)
        default_start=max(date.today()-timedelta(days=60), snaps.snapshot_time.min().date())
        start=c1.date_input("Start",default_start)
        end=c2.date_input("End",date.today()-timedelta(days=1))
        sims=c3.selectbox("Simulations/game",[500,1000,1500,3000],index=1)
        if st.button("Build prospective validation set",type="primary"):
            ds,matched=build_prospective_dataset(start,end,sims)
            st.session_state["prospective_ds"]=ds
            if matched:
                st.success(f"Matched {matched} completed games to free pregame snapshots.")
            else:
                st.warning("No completed games matched collected snapshots yet.")

        ds=st.session_state.get("prospective_ds")
        if isinstance(ds,pd.DataFrame) and not ds.empty:
            st.dataframe(ds.head(20),use_container_width=True,hide_index=True)
            rows=to_team_rows(ds)
            if rows.date.nunique() < 30:
                st.warning("You have less than 30 days of prospective data. Treat any performance numbers as preliminary.")
            train_days=st.selectbox("Training window",[21,30,45,60],index=1)
            test_days=st.selectbox("Unseen test window",[7,14,21],index=0)
            step_days=st.selectbox("Walk step",[7,14],index=0)
            if st.button("Run prospective walk-forward"):
                st.session_state["wf"]=walk_forward(rows,train_days,test_days,step_days)

            wf=st.session_state.get("wf")
            if isinstance(wf,pd.DataFrame) and not wf.empty:
                em=metrics(wf,"wf_prob")
                mm=metrics(wf,"market_prob")
                a,b,c,d=st.columns(4)
                a.metric("Unseen predictions",len(wf))
                b.metric("EDGE Brier",f'{em["brier"]:.4f}')
                c.metric("Market Brier",f'{mm["brier"]:.4f}')
                d.metric("EDGE accuracy",f'{em["accuracy"]*100:.1f}%')

with tab_market:
    wf=st.session_state.get("wf")
    if not isinstance(wf,pd.DataFrame) or wf.empty:
        st.info("Run the prospective walk-forward test first.")
    else:
        em=metrics(wf,"wf_prob")
        mm=metrics(wf,"market_prob")
        c1,c2,c3,c4=st.columns(4)
        c1.metric("EDGE Brier",f'{em["brier"]:.4f}')
        c2.metric("Market Brier",f'{mm["brier"]:.4f}')
        c3.metric("EDGE log loss",f'{em["logloss"]:.4f}')
        c4.metric("Market log loss",f'{mm["logloss"]:.4f}')
        if em["brier"]<mm["brier"]:
            st.success("EDGE beat the market benchmark on Brier score in this prospective unseen sample.")
        else:
            st.warning("EDGE did not beat the market benchmark on Brier score in this prospective unseen sample.")
        rt=roi_table(wf)
        if not rt.empty:
            show=rt.copy()
            show["win_rate"]=(show.win_rate*100).round(1).astype(str)+"%"
            show["roi"]=(show.roi*100).round(2).astype(str)+"%"
            show["avg_edge"]=show.avg_edge.round(2).astype(str)+"%"
            st.subheader("Flat-stake ROI by minimum EDGE")
            st.dataframe(show,use_container_width=True,hide_index=True)

with tab_setup:
    st.subheader("How the free version works")
    st.write(
        "GitHub Actions calls the normal CURRENT odds endpoint on a schedule and commits the snapshot CSV "
        "back into this repository. Current odds are available on The Odds API's free usage plan; the paid "
        "historical endpoint is no longer used by EDGE v4.2."
    )
    st.write(
        "Because the collector starts now, it cannot magically recreate past sportsbook prices. "
        "It builds a trustworthy historical benchmark prospectively from this point forward."
    )
    st.code(
        "Required GitHub repository secret:\n"
        "THE_ODDS_API_KEY = your existing Odds API key\n\n"
        "Workflow file included:\n"
        ".github/workflows/collect_odds.yml"
    )
    st.info("Once the GitHub Action runs, data/odds_snapshots.csv will begin filling automatically.")

st.caption("EDGE v4.2 — free prospective market validation")
