
from pathlib import Path
from datetime import datetime, timezone
import os
import pandas as pd
import requests

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
    if r.status_code==422:
        # Some sports may not expose every requested featured market at a given time.
        return []
    r.raise_for_status()
    return r.json()

def active_golf_keys():
    r=requests.get(f"{BASE}/sports",params={"apiKey":API_KEY,"all":"true"},timeout=20)
    r.raise_for_status()
    return [x["key"] for x in r.json() if str(x.get("group","")).lower()=="golf" and x.get("active")]

snap=datetime.now(timezone.utc).isoformat()
rows=[]
for sport_key,markets in SPORTS:
    try:
        events=fetch(sport_key,markets)
    except Exception as e:
        print(f"{sport_key}: {e}")
        continue
    for ev in events:
        for book in ev.get("bookmakers",[]):
            for market in book.get("markets",[]):
                for o in market.get("outcomes",[]):
                    rows.append({
                        "snapshot_time":snap,
                        "sport_key":sport_key,
                        "event_id":ev.get("id"),
                        "commence_time":ev.get("commence_time"),
                        "home_team":ev.get("home_team"),
                        "away_team":ev.get("away_team"),
                        "book_key":book.get("key"),
                        "book":book.get("title"),
                        "market":market.get("key"),
                        "outcome":o.get("name"),
                        "point":o.get("point"),
                        "price":o.get("price"),
                    })

# Golf: collect any currently active golf outright feed exposed by the user's plan.
for key in active_golf_keys():
    try:
        events=fetch(key,"outrights")
    except Exception as e:
        print(f"{key}: {e}")
        continue
    for ev in events:
        for book in ev.get("bookmakers",[]):
            for market in book.get("markets",[]):
                for o in market.get("outcomes",[]):
                    rows.append({
                        "snapshot_time":snap,
                        "sport_key":key,
                        "event_id":ev.get("id"),
                        "commence_time":ev.get("commence_time"),
                        "home_team":ev.get("home_team"),
                        "away_team":ev.get("away_team"),
                        "book_key":book.get("key"),
                        "book":book.get("title"),
                        "market":market.get("key"),
                        "outcome":o.get("name"),
                        "point":o.get("point"),
                        "price":o.get("price"),
                    })

new=pd.DataFrame(rows)
if OUT.exists():
    old=pd.read_csv(OUT)
    df=pd.concat([old,new],ignore_index=True)
else:
    df=new
if not df.empty:
    dedupe=[c for c in ["snapshot_time","sport_key","event_id","book_key","market","outcome","point"] if c in df.columns]
    df=df.drop_duplicates(subset=dedupe,keep="last")
df.to_csv(OUT,index=False)
print(f"Wrote {len(new)} new observations; database now {len(df)} rows.")
