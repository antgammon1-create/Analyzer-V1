# EDGE MLB v2.1

## What v2.1 fixes

### 1. Weather at first pitch
v2 accidentally used the first hourly weather value returned by Open-Meteo.
v2.1 converts MLB's UTC scheduled game time to the venue's local time using
Open-Meteo's UTC offset and selects the nearest hourly forecast.

### 2. No-vig consensus market
v2 compared the model against one sportsbook's vigged implied probability.
v2.1:
- removes vig inside each sportsbook's two-way market
- averages those probabilities across sportsbooks
- separately keeps the best available bettor price

### 3. More markets
v2.1 analyzes:
- Moneyline
- Totals
- Run lines / spreads

Each available market is priced from the same game simulation.

### 4. Data-quality scoring
The Edge Score now incorporates:
- probable-starter availability
- lineup confirmation
- weather quality
- number of sportsbooks in consensus
- market type
- model stability

### 5. Confirmed lineup detection
When MLB boxscore data contains batting orders, v2.1 marks the lineups confirmed.
If not, it explicitly shows Projected/TBD and lowers data quality.

### 6. Bullpen proxy
v2.1 adds team pitching quality as a conservative bullpen-strength proxy.
This is NOT yet true reliever fatigue/availability.

### 7. Better labeling
Expected value is displayed as **Uncalibrated EV** until historical validation is built.

## Deploying over your current app

In the GitHub repository currently connected to Streamlit:

1. Replace `app.py` with the v2.1 `app.py`.
2. Replace `requirements.txt`.
3. Commit the changes.
4. Streamlit should automatically redeploy.

Keep your existing Streamlit secret:
`THE_ODDS_API_KEY = "YOUR_EXISTING_KEY"`

Do not put the API key into GitHub.

## Still needed before real-money trust

The next major version should be historical validation rather than more cosmetics:

- historical MLB games
- historical odds snapshots
- closing lines
- out-of-sample train/test split
- probability calibration
- Brier score and log loss
- ROI by market
- CLV
- performance by Edge Score band
- true bullpen availability/fatigue
- batter-level lineups
- handedness/platoon splits
- park-specific wind orientation
- trained ensemble model

Until that exists, v2.1 should be treated as a research model, not a proven betting system.
