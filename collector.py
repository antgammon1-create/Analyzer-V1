
from pathlib import Path
from datetime import datetime, timezone
import os, requests, pandas as pd

API_KEY=os.environ["THE_ODDS_API_KEY"]
BASE="https://api.the-odds-api.com/v4"
OUT=Path("data/market_snapshots.csv")
OUT.parent.mkdir(parents=True,exist_ok=True)

SPORTS=[
 ("baseball_mlb","h2h,spreads,totals"),
 ("americanfootball_nfl","h2h,spreads,totals"),
 ("americanfootball_ncaaf","h2h,spreads,totals"),
]

def fetch(key,markets):
    r=requests.get(f"{BASE}/sports/{key}/odds",params={
        "apiKey":API_KEY,"regions":"us","markets":markets,
        "oddsFormat":"american","dateFormat":"iso"
    },timeout=30)
    if r.status_code==422: return []
    r.raise_for_status()
    return r.json()

def golf_keys():
    r=requests.get(f"{BASE}/sports",params={"apiKey":API_KEY,"all":"true"},timeout=20)
    r.raise_for_status()
    return [s["key"] for s in r.json() if str(s.get("group","")).lower()=="golf" and s.get("active")]

snap=datetime.now(timezone.utc).isoformat()
rows=[]

def collect(key,markets):
    try: events=fetch(key,markets)
    except Exception as e:
        print(key,e); return
    for ev in events:
        for b in ev.get("bookmakers",[]):
            for m in b.get("markets",[]):
                for o in m.get("outcomes",[]):
                    rows.append({
                        "snapshot_time":snap,"sport_key":key,"event_id":ev.get("id"),
                        "commence_time":ev.get("commence_time"),"home_team":ev.get("home_team"),
                        "away_team":ev.get("away_team"),"book_key":b.get("key"),"book":b.get("title"),
                        "market":m.get("key"),"outcome":o.get("name"),"point":o.get("point"),
                        "price":o.get("price")
                    })

for k,m in SPORTS: collect(k,m)
for k in golf_keys(): collect(k,"outrights")

new=pd.DataFrame(rows)
if OUT.exists():
    try: old=pd.read_csv(OUT)
    except: old=pd.DataFrame()
    df=pd.concat([old,new],ignore_index=True)
else:
    df=new
if not df.empty:
    keys=[c for c in ["snapshot_time","sport_key","event_id","book_key","market","outcome","point"] if c in df]
    df=df.drop_duplicates(subset=keys,keep="last")
df.to_csv(OUT,index=False)
print(f"Wrote {len(new)} observations; database now {len(df)} rows.")
