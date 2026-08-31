
import os, csv
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
import requests
import pandas as pd

API_KEY=os.environ["THE_ODDS_API_KEY"]
OUT=Path("data/odds_snapshots.csv")
OUT.parent.mkdir(parents=True,exist_ok=True)

def implied_prob(odds):
    odds=float(odds)
    return 100/(odds+100) if odds>=0 else -odds/(-odds+100)

def no_vig(p1,p2):
    s=p1+p2
    return p1/s,p2/s

r=requests.get(
    "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds",
    params={
        "apiKey":API_KEY,
        "regions":"us",
        "markets":"h2h",
        "oddsFormat":"american"
    },
    timeout=30
)
r.raise_for_status()
events=r.json()

snapshot_time=datetime.now(timezone.utc).isoformat()
rows=[]

for event in events:
    probs=defaultdict(list)
    offers=defaultdict(list)

    for book in event.get("bookmakers",[]):
        outcomes=None
        for market in book.get("markets",[]):
            if market.get("key")=="h2h":
                outcomes=market.get("outcomes",[])
                break
        if not outcomes or len(outcomes)<2:
            continue

        pair=[o for o in outcomes if o.get("price") is not None][:2]
        if len(pair)<2:
            continue

        p1,p2=no_vig(implied_prob(pair[0]["price"]),implied_prob(pair[1]["price"]))
        for o,p in zip(pair,[p1,p2]):
            probs[o["name"]].append(p)
            offers[o["name"]].append((float(o["price"]),book.get("title","Book")))

    home=event.get("home_team")
    away=event.get("away_team")
    if home not in probs or away not in probs:
        continue

    def side(team):
        best=min(offers[team],key=lambda x: implied_prob(x[0]))
        return sum(probs[team])/len(probs[team]),best[0],best[1],len(probs[team])

    hp,ho,hb,hbooks=side(home)
    ap,ao,ab,abooks=side(away)

    rows.append({
        "snapshot_time":snapshot_time,
        "event_id":event.get("id"),
        "commence_time":event.get("commence_time"),
        "home_team":home,
        "away_team":away,
        "home_no_vig_prob":hp,
        "away_no_vig_prob":ap,
        "home_best_odds":ho,
        "away_best_odds":ao,
        "home_best_book":hb,
        "away_best_book":ab,
        "books":min(hbooks,abooks),
    })

new=pd.DataFrame(rows)
if OUT.exists():
    old=pd.read_csv(OUT)
    combined=pd.concat([old,new],ignore_index=True)
else:
    combined=new

if not combined.empty:
    combined["snapshot_time"]=pd.to_datetime(combined["snapshot_time"],utc=True,errors="coerce")
    combined["commence_time"]=pd.to_datetime(combined["commence_time"],utc=True,errors="coerce")
    # Keep one row per game per collector run.
    combined=combined.drop_duplicates(
        subset=["event_id","snapshot_time"],keep="last"
    ).sort_values(["commence_time","snapshot_time"])
    combined.to_csv(OUT,index=False)
else:
    OUT.write_text("snapshot_time,event_id,commence_time,home_team,away_team,home_no_vig_prob,away_no_vig_prob,home_best_odds,away_best_odds,home_best_book,away_best_book,books\n")

print(f"Saved {len(new)} current MLB events; total rows {len(combined)}")
