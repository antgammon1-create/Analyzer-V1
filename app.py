
import os, math, random, json
from datetime import date, datetime, timedelta
from collections import defaultdict

import requests
import pandas as pd
import numpy as np
import streamlit as st

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

st.set_page_config(page_title="EDGE v4", page_icon="📈", layout="wide")

MLB = "https://statsapi.mlb.com/api/v1"
ODDS_SPORT = "baseball_mlb"
WEATHER = "https://api.open-meteo.com/v1/forecast"

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
    r = requests.get(url, params=params or {}, timeout=30)
    r.raise_for_status()
    return r.json()

def implied_prob(odds):
    odds = float(odds)
    return 100/(odds+100) if odds >= 0 else -odds/(-odds+100)

def american_decimal(odds):
    odds = float(odds)
    return 1 + (odds/100 if odds >= 0 else 100/abs(odds))

def no_vig(p1, p2):
    s = p1+p2
    return (p1/s, p2/s) if s > 0 else (.5,.5)

def fair_price(p):
    p = min(max(float(p), .0001), .9999)
    return round(-100*p/(1-p)) if p >= .5 else round(100*(1-p)/p)

def norm(s):
    return "".join(c.lower() for c in (s or "") if c.isalnum())

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

# ============================================================
# MLB data
# ============================================================

@st.cache_data(ttl=300)
def schedule(d):
    return get_json(
        f"{MLB}/schedule",
        {"sportId": 1, "date": d, "hydrate": "probablePitcher,team,venue"}
    ).get("dates", [])

def games_for(d):
    out = []
    for day in schedule(d):
        for g in day.get("games", []):
            h = g.get("teams",{}).get("home",{})
            a = g.get("teams",{}).get("away",{})
            out.append({
                "gamePk": g.get("gamePk"),
                "gameDate": g.get("gameDate"),
                "status": g.get("status",{}).get("detailedState"),
                "home": h.get("team",{}).get("name","Home"),
                "home_id": h.get("team",{}).get("id"),
                "home_score": h.get("score"),
                "away": a.get("team",{}).get("name","Away"),
                "away_id": a.get("team",{}).get("id"),
                "away_score": a.get("score"),
                "hp": (h.get("probablePitcher") or {}).get("fullName"),
                "hp_id": (h.get("probablePitcher") or {}).get("id"),
                "ap": (a.get("probablePitcher") or {}).get("fullName"),
                "ap_id": (a.get("probablePitcher") or {}).get("id"),
                "venue_id": (g.get("venue") or {}).get("id"),
            })
    return out

def season_start(ds):
    y = datetime.fromisoformat(ds).year
    return f"{y}-03-20"

def day_before(ds):
    return (datetime.fromisoformat(ds)-timedelta(days=1)).date().isoformat()

@st.cache_data(ttl=1800)
def pitcher_range(pid, start, end):
    if not pid:
        return {}
    try:
        d = get_json(
            f"{MLB}/people/{pid}/stats",
            {"stats":"byDateRange","group":"pitching","startDate":start,"endDate":end}
        )
        sp = d.get("stats",[{}])[0].get("splits") or []
        s = sp[0].get("stat",{}) if sp else {}
    except Exception:
        s = {}
    return {
        "era": float(s.get("era",4.3) or 4.3),
        "whip": float(s.get("whip",1.3) or 1.3),
        "k9": float(s.get("strikeoutsPer9Inn",8.5) or 8.5),
        "bb9": float(s.get("walksPer9Inn",3.2) or 3.2),
        "hr9": float(s.get("homeRunsPer9",1.2) or 1.2),
        "innings": float(s.get("inningsPitched",0) or 0),
    }

@st.cache_data(ttl=1800)
def team_range(tid, group, start, end):
    if not tid:
        return {}
    try:
        d = get_json(
            f"{MLB}/teams/{tid}/stats",
            {"stats":"byDateRange","group":group,"startDate":start,"endDate":end,"sportIds":1}
        )
        sp = d.get("stats",[{}])[0].get("splits") or []
        s = sp[0].get("stat",{}) if sp else {}
    except Exception:
        s = {}
    if group == "hitting":
        return {
            "avg": float(s.get("avg",.240) or .240),
            "obp": float(s.get("obp",.310) or .310),
            "slg": float(s.get("slg",.400) or .400),
            "ops": float(s.get("ops",.710) or .710),
        }
    return {
        "era": float(s.get("era",4.2) or 4.2),
        "whip": float(s.get("whip",1.3) or 1.3),
        "k9": float(s.get("strikeoutsPer9Inn",8.5) or 8.5),
    }

@st.cache_data(ttl=900)
def venue_info(vid):
    if not vid:
        return {}
    d = get_json(f"{MLB}/venues/{vid}", {"hydrate":"location"})
    vs = d.get("venues",[])
    if not vs:
        return {}
    v = vs[0]
    loc = v.get("location",{})
    coords = loc.get("defaultCoordinates",{})
    return {
        "name":v.get("name",""),
        "lat":coords.get("latitude"),
        "lon":coords.get("longitude"),
    }

# ============================================================
# Features
# ============================================================

def pitcher_quality(s):
    if not s:
        return 0
    raw = (
        (4.3-s["era"])*.16 +
        (1.3-s["whip"])*.65 +
        (s["k9"]-8.5)*.035 +
        (3.2-s["bb9"])*.035 +
        (1.2-s["hr9"])*.16
    )
    shrink = min(1.0, max(0,s.get("innings",0))/80)
    return clamp(raw*shrink,-1.25,1.25)

def offense_quality(s):
    if not s:
        return 0
    return clamp(
        (s["ops"]-.710)*2.4 + (s["obp"]-.310)*1.4 + (s["slg"]-.400)*1.2,
        -1.25,1.25
    )

def bullpen_proxy(s):
    if not s:
        return 0
    return clamp(
        (4.2-s["era"])*.10 + (1.3-s["whip"])*.35 + (s["k9"]-8.5)*.02,
        -.60,.60
    )

def game_features(g, asof_date):
    start = season_start(asof_date)
    end = day_before(asof_date)

    hp = pitcher_range(g["hp_id"], start, end)
    ap = pitcher_range(g["ap_id"], start, end)
    ho = team_range(g["home_id"], "hitting", start, end)
    ao = team_range(g["away_id"], "hitting", start, end)
    htp = team_range(g["home_id"], "pitching", start, end)
    atp = team_range(g["away_id"], "pitching", start, end)

    return {
        "home_off": offense_quality(ho),
        "away_off": offense_quality(ao),
        "home_sp": pitcher_quality(hp),
        "away_sp": pitcher_quality(ap),
        "home_bp": bullpen_proxy(htp),
        "away_bp": bullpen_proxy(atp),
        "home_field": 1.0,
    }

def raw_run_projection(feat):
    hr = (
        4.45 + .20 +
        .58*feat["home_off"] -
        .72*feat["away_sp"] -
        .18*feat["away_bp"] +
        .05*feat["home_sp"]
    )
    ar = (
        4.45 +
        .58*feat["away_off"] -
        .72*feat["home_sp"] -
        .18*feat["home_bp"] +
        .05*feat["away_sp"]
    )
    return clamp(hr,1.5,8.5), clamp(ar,1.5,8.5)

def poisson(rng, lam):
    L = math.exp(-lam)
    k,p = 0,1.0
    while p > L:
        k += 1
        p *= rng.random()
    return k-1

def simulate_ml(hr, ar, n, seed):
    rng = random.Random(seed)
    hw=aw=0
    for _ in range(n):
        h = poisson(rng,hr)
        a = poisson(rng,ar)
        hw += h>a
        aw += a>h
    ties = max(0,n-hw-aw)
    return (hw+ties/2)/n, (aw+ties/2)/n

# ============================================================
# Historical odds
# ============================================================

@st.cache_data(ttl=3600)
def historical_odds(snapshot_iso):
    key = secret("THE_ODDS_API_KEY")
    if not key:
        return [], "Missing key"
    try:
        r = requests.get(
            f"https://api.the-odds-api.com/v4/historical/sports/{ODDS_SPORT}/odds",
            params={
                "apiKey":key,
                "regions":"us",
                "markets":"h2h",
                "oddsFormat":"american",
                "date":snapshot_iso
            },
            timeout=30
        )
        r.raise_for_status()
        d = r.json()
        return d.get("data", d if isinstance(d,list) else []), None
    except Exception as e:
        return [], str(e)

def moneyline_market(event):
    probs = defaultdict(list)
    offers = defaultdict(list)
    for book in event.get("bookmakers",[]):
        outs = None
        for m in book.get("markets",[]):
            if m.get("key")=="h2h":
                outs = m.get("outcomes",[])
                break
        if not outs or len(outs)<2:
            continue
        pair=[o for o in outs if o.get("price") is not None][:2]
        if len(pair)<2:
            continue
        nvp = no_vig(implied_prob(pair[0]["price"]), implied_prob(pair[1]["price"]))
        for o,p in zip(pair,nvp):
            probs[o["name"]].append(p)
            offers[o["name"]].append((float(o["price"]),book.get("title","Book")))
    out={}
    for team, ps in probs.items():
        best = min(offers[team], key=lambda x: implied_prob(x[0]))
        out[team] = {
            "consensus_prob": sum(ps)/len(ps),
            "best_odds": best[0],
            "best_book": best[1],
            "books": len(ps)
        }
    return out

# ============================================================
# Dataset builder
# ============================================================

def build_dataset(start_date, end_date, sims=1500, include_hist_odds=False):
    recs=[]
    days=(end_date-start_date).days+1
    prog=st.progress(0)
    status=st.empty()

    for i in range(days):
        cur=start_date+timedelta(days=i)
        ds=cur.isoformat()
        status.write(f"Building {ds} ({i+1}/{days})")
        try:
            gs=games_for(ds)
        except Exception:
            gs=[]

        lookup={}
        if include_hist_odds:
            # Noon/afternoon UTC snapshot; intentionally fixed and documented.
            evts,_=historical_odds(f"{ds}T16:00:00Z")
            lookup={(norm(e.get("away_team")),norm(e.get("home_team"))):e for e in evts}

        for g in gs:
            if g.get("home_score") is None or g.get("away_score") is None:
                continue
            try:
                feat=game_features(g,ds)
                hr,ar=raw_run_projection(feat)
                hp,ap=simulate_ml(hr,ar,sims,int(g["gamePk"] or 1))
            except Exception:
                continue

            home_win=int(g["home_score"]>g["away_score"])
            base={
                "date":ds,
                "game":f'{g["away"]} @ {g["home"]}',
                "home_team":g["home"],
                "away_team":g["away"],
                "home_score":g["home_score"],
                "away_score":g["away_score"],
                **feat,
                "proj_home_runs":hr,
                "proj_away_runs":ar,
                "home_model_prob":hp,
                "away_model_prob":ap,
                "home_outcome":home_win,
                "away_outcome":1-home_win,
            }

            if include_hist_odds:
                e=lookup.get((norm(g["away"]),norm(g["home"])))
                if e:
                    ml=moneyline_market(e)
                    for team,m in ml.items():
                        if norm(team)==norm(g["home"]):
                            base["home_market_prob"]=m["consensus_prob"]
                            base["home_odds"]=m["best_odds"]
                        elif norm(team)==norm(g["away"]):
                            base["away_market_prob"]=m["consensus_prob"]
                            base["away_odds"]=m["best_odds"]

            recs.append(base)

        prog.progress((i+1)/days)

    status.empty()
    prog.empty()
    return pd.DataFrame(recs)

# ============================================================
# Walk-forward training
# ============================================================

FEATURE_COLS = [
    "home_off","away_off","home_sp","away_sp","home_bp","away_bp","home_field"
]

def to_team_rows(df):
    rows=[]
    for _,r in df.iterrows():
        home_feat = [
            r["home_off"]-r["away_off"],
            r["home_sp"]-r["away_sp"],
            r["home_bp"]-r["away_bp"],
            1.0,
        ]
        away_feat = [
            r["away_off"]-r["home_off"],
            r["away_sp"]-r["home_sp"],
            r["away_bp"]-r["home_bp"],
            0.0,
        ]
        rows.append({
            "date":r["date"],"game":r["game"],"team":r["home_team"],"side":"home",
            "f_off":home_feat[0],"f_sp":home_feat[1],"f_bp":home_feat[2],"home":home_feat[3],
            "raw_model_prob":r["home_model_prob"],"outcome":r["home_outcome"],
            "market_prob":r.get("home_market_prob",np.nan),"odds":r.get("home_odds",np.nan)
        })
        rows.append({
            "date":r["date"],"game":r["game"],"team":r["away_team"],"side":"away",
            "f_off":away_feat[0],"f_sp":away_feat[1],"f_bp":away_feat[2],"home":away_feat[3],
            "raw_model_prob":r["away_model_prob"],"outcome":r["away_outcome"],
            "market_prob":r.get("away_market_prob",np.nan),"odds":r.get("away_odds",np.nan)
        })
    return pd.DataFrame(rows)

MODEL_FEATURES=["f_off","f_sp","f_bp","home"]

def fit_model(train_rows):
    X=train_rows[MODEL_FEATURES].values
    y=train_rows["outcome"].astype(int).values
    model=LogisticRegression(C=1.0,solver="lbfgs")
    model.fit(X,y)
    return model

def fit_calibrator(train_rows, base_probs):
    p=np.clip(base_probs,.001,.999)
    logits=np.log(p/(1-p)).reshape(-1,1)
    y=train_rows["outcome"].astype(int).values
    cal=LogisticRegression(C=1e6,solver="lbfgs")
    cal.fit(logits,y)
    return cal

def apply_calibrator(cal, probs):
    p=np.clip(np.asarray(probs),.001,.999)
    logits=np.log(p/(1-p)).reshape(-1,1)
    return cal.predict_proba(logits)[:,1]

def walk_forward(rows, train_days=45, test_days=14, step_days=14, calibrate=True):
    rows=rows.copy()
    rows["date_dt"]=pd.to_datetime(rows["date"])
    start=rows["date_dt"].min()
    end=rows["date_dt"].max()

    windows=[]
    cursor=start
    while True:
        train_start=cursor
        train_end=train_start+pd.Timedelta(days=train_days-1)
        test_start=train_end+pd.Timedelta(days=1)
        test_end=test_start+pd.Timedelta(days=test_days-1)
        if test_end>end:
            break
        windows.append((train_start,train_end,test_start,test_end))
        cursor=cursor+pd.Timedelta(days=step_days)

    outputs=[]
    for wid,(tr0,tr1,te0,te1) in enumerate(windows,1):
        train=rows[(rows.date_dt>=tr0)&(rows.date_dt<=tr1)].copy()
        test=rows[(rows.date_dt>=te0)&(rows.date_dt<=te1)].copy()
        if len(train)<100 or len(test)<20 or train.outcome.nunique()<2:
            continue

        model=fit_model(train)
        base_train=model.predict_proba(train[MODEL_FEATURES].values)[:,1]
        base_test=model.predict_proba(test[MODEL_FEATURES].values)[:,1]

        if calibrate:
            cal=fit_calibrator(train,base_train)
            pred=apply_calibrator(cal,base_test)
        else:
            pred=base_test

        tmp=test.copy()
        tmp["wf_prob"]=pred
        tmp["window"]=wid
        tmp["train_start"]=tr0.date().isoformat()
        tmp["train_end"]=tr1.date().isoformat()
        tmp["test_start"]=te0.date().isoformat()
        tmp["test_end"]=te1.date().isoformat()
        outputs.append(tmp)

    return pd.concat(outputs,ignore_index=True) if outputs else pd.DataFrame()

def metric_summary(df, prob_col):
    p=np.clip(df[prob_col].values,.001,.999)
    y=df["outcome"].astype(int).values
    return {
        "n":len(df),
        "brier":float(brier_score_loss(y,p)),
        "logloss":float(log_loss(y,p,labels=[0,1])),
        "accuracy":float(((p>=.5).astype(int)==y).mean()),
    }

def market_summary(df):
    d=df.dropna(subset=["market_prob"]).copy()
    if d.empty:
        return None
    return metric_summary(d,"market_prob")

def roi_table(df, prob_col="wf_prob"):
    d=df.dropna(subset=["market_prob","odds"]).copy()
    if d.empty:
        return pd.DataFrame()
    d["edge"]=(d[prob_col]-d["market_prob"])*100
    d["bet_return"]=np.where(
        d["outcome"]==1,
        d["odds"].apply(american_decimal)-1,
        -1.0
    )
    rows=[]
    for th in [1,2,3,4,5,7.5,10]:
        b=d[d.edge>=th]
        if len(b)==0:
            continue
        rows.append({
            "edge_threshold":th,
            "bets":len(b),
            "win_rate":b.outcome.mean(),
            "roi":b.bet_return.mean(),
            "avg_edge":b.edge.mean()
        })
    return pd.DataFrame(rows)

# ============================================================
# UI
# ============================================================

st.title("📈 EDGE v4 — Walk-Forward Validation Lab")
st.caption("Train → calibrate → freeze → test on unseen games · compare directly with the market")

tab_build, tab_walk, tab_compare, tab_export = st.tabs(
    ["🧱 Build dataset","🚶 Walk-forward","⚖️ Market benchmark","💾 Export"]
)

with tab_build:
    st.subheader("Build historical MLB dataset")
    c1,c2,c3=st.columns(3)
    start=c1.date_input("Start", date.today()-timedelta(days=90))
    end=c2.date_input("End", date.today()-timedelta(days=1))
    sims=c3.selectbox("Simulations/game",[500,1000,1500,3000],index=2)

    use_hist=st.checkbox(
        "Include historical sportsbook odds if your Odds API plan supports the historical endpoint",
        value=False
    )

    days=(end-start).days+1
    if days>120:
        st.warning("For Streamlit reliability, v4 limits a single dataset build to 120 days. Build multiple segments and export them.")
    if st.button("Build dataset",type="primary"):
        if end<start:
            st.error("End must be after start.")
        elif days>120:
            st.error("Choose 120 days or fewer.")
        else:
            ds=build_dataset(start,end,sims,use_hist)
            st.session_state["dataset"]=ds

    ds=st.session_state.get("dataset")
    if isinstance(ds,pd.DataFrame) and not ds.empty:
        st.success(f"Dataset ready: {len(ds)} completed games.")
        st.dataframe(ds.head(20),use_container_width=True,hide_index=True)
        st.download_button(
            "Download dataset CSV",
            ds.to_csv(index=False).encode(),
            file_name=f"EDGE_dataset_{start}_{end}.csv",
            mime="text/csv"
        )

with tab_walk:
    st.subheader("Walk-forward validation")
    ds=st.session_state.get("dataset")
    uploaded=st.file_uploader("Or upload a previously exported EDGE dataset CSV",type=["csv"],key="ds_upload")
    if uploaded is not None:
        ds=pd.read_csv(uploaded)
        st.session_state["dataset"]=ds

    if not isinstance(ds,pd.DataFrame) or ds.empty:
        st.info("Build or upload a historical dataset first.")
    else:
        rows=to_team_rows(ds)

        c1,c2,c3=st.columns(3)
        train_days=c1.selectbox("Training window (days)",[30,45,60,90],index=1)
        test_days=c2.selectbox("Unseen test window (days)",[7,14,21,30],index=1)
        step_days=c3.selectbox("Walk step (days)",[7,14,21,30],index=1)
        calibrate=st.checkbox("Calibrate on training window before testing",value=True)

        if st.button("Run walk-forward validation",type="primary"):
            wf=walk_forward(rows,train_days,test_days,step_days,calibrate)
            st.session_state["wf"]=wf

        wf=st.session_state.get("wf")
        if isinstance(wf,pd.DataFrame) and not wf.empty:
            m=metric_summary(wf,"wf_prob")
            a,b,c,d=st.columns(4)
            a.metric("Unseen predictions",m["n"])
            b.metric("Brier",f'{m["brier"]:.4f}')
            c.metric("Log loss",f'{m["logloss"]:.4f}')
            d.metric("Accuracy",f'{m["accuracy"]*100:.1f}%')

            st.write("### Probability calibration on unseen games")
            tmp=wf.copy()
            tmp["bucket"]=pd.cut(
                tmp["wf_prob"],bins=[0,.4,.45,.5,.55,.6,1],
                labels=["<40","40–45","45–50","50–55","55–60","60+"],include_lowest=True
            )
            caltab=tmp.groupby("bucket",observed=False).agg(
                predictions=("outcome","size"),
                avg_model=("wf_prob","mean"),
                actual_win_rate=("outcome","mean"),
            ).reset_index()
            st.dataframe(caltab,use_container_width=True,hide_index=True)

            st.write("### Window-by-window stability")
            windows=wf.groupby("window").apply(
                lambda x: pd.Series(metric_summary(x,"wf_prob"))
            ).reset_index()
            st.dataframe(windows,use_container_width=True,hide_index=True)

with tab_compare:
    st.subheader("EDGE vs market")
    wf=st.session_state.get("wf")
    if not isinstance(wf,pd.DataFrame) or wf.empty:
        st.info("Run walk-forward validation first.")
    else:
        edge_m=metric_summary(wf,"wf_prob")
        market_m=market_summary(wf)

        if market_m is None:
            st.warning("No historical market probabilities are present. Rebuild/upload a dataset with historical odds to compare EDGE against sportsbooks.")
        else:
            c1,c2,c3,c4=st.columns(4)
            c1.metric("EDGE Brier",f'{edge_m["brier"]:.4f}')
            c2.metric("Market Brier",f'{market_m["brier"]:.4f}')
            c3.metric("EDGE log loss",f'{edge_m["logloss"]:.4f}')
            c4.metric("Market log loss",f'{market_m["logloss"]:.4f}')

            if edge_m["brier"] < market_m["brier"]:
                st.success("EDGE beats the market benchmark on Brier score in this unseen sample.")
            else:
                st.warning("EDGE does not beat the market benchmark on Brier score in this unseen sample.")

            rt=roi_table(wf)
            if not rt.empty:
                st.write("### Flat-stake ROI by minimum model edge")
                rt2=rt.copy()
                rt2["win_rate"]=(rt2.win_rate*100).round(1).astype(str)+"%"
                rt2["roi"]=(rt2.roi*100).round(2).astype(str)+"%"
                rt2["avg_edge"]=rt2.avg_edge.round(2).astype(str)+"%"
                st.dataframe(rt2,use_container_width=True,hide_index=True)

                strong=rt[rt.edge_threshold>=5]
                if not strong.empty and (strong.roi>0).any():
                    st.info("Positive ROI appears in at least one ≥5% edge band. Check sample size and window stability before treating it as validated.")
                else:
                    st.warning("No robust positive ROI signal is visible in the tested ≥5% edge bands.")

with tab_export:
    st.subheader("Export validation artifacts")
    wf=st.session_state.get("wf")
    ds=st.session_state.get("dataset")

    if isinstance(wf,pd.DataFrame) and not wf.empty:
        st.download_button(
            "Download walk-forward predictions CSV",
            wf.to_csv(index=False).encode(),
            file_name="EDGE_walkforward_predictions.csv",
            mime="text/csv"
        )
        summary=metric_summary(wf,"wf_prob")
        ms=market_summary(wf)
        artifact={"edge":summary,"market":ms}
        st.download_button(
            "Download validation summary JSON",
            json.dumps(artifact,indent=2).encode(),
            file_name="EDGE_validation_summary.json",
            mime="application/json"
        )
    else:
        st.info("No walk-forward results yet.")

st.divider()
st.caption(
    "EDGE v4 is a validation lab, not a guarantee of profitability. "
    "The key standard is whether the model remains calibrated and competitive with the market on unseen data."
)
