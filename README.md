# EDGE v3 — Complete MLB Research Platform

This is the largest EDGE build so far. It turns the live prototype into a research platform that can be tested rather than merely inspected.

## What is included

### Live MLB engine
- MLB schedule
- probable pitchers
- leakage-aware team and pitcher stats
- venue data
- first-pitch weather
- weather sanity checks
- lineup confirmation detection
- team pitching / bullpen proxy
- Monte Carlo simulations
- moneyline
- totals
- run lines
- no-vig consensus market probability
- best available price
- fair price
- EV
- Edge Score
- reliability label
- model-market disagreement penalty
- run-projection explanation
- downloadable live-board CSV

### Historical backtesting
- user-selectable historical windows up to 45 days per run
- stats are pulled only through the day BEFORE each game
- historical weather is omitted to prevent forecast look-ahead
- Brier score
- log loss
- classification accuracy
- probability-band calibration table
- downloadable backtest CSV

### Historical odds / ROI
Optional:
- attempts The Odds API historical endpoint
- requires an account/API plan that supports historical odds
- flat-stake ROI by user-defined edge threshold

If the historical endpoint is not available on the user's API plan, the prediction backtest still works without ROI.

### Probability calibration
- logistic/Platt-style calibration fitted from a completed backtest
- compares raw vs calibrated Brier score
- can activate calibrated probabilities for the current Streamlit session
- calibration JSON export

### Tracking
- CSV-based performance tracking
- suitable for ROI / win rate / CLV workflows
- portable because Streamlit Community Cloud local storage is not guaranteed to persist

## Important limitation

This is a complete **MLB research platform**, not a claim that the model has a proven betting edge.

The app now contains the machinery needed to test and calibrate the model. The user must run meaningful historical samples before trusting Edge Scores or EV.

## Other sports

NFL, NCAA Football, and PGA are intentionally not assigned fake model probabilities. They require their own:
- historical datasets
- sport-specific features
- backtests
- calibration

The app contains a roadmap tab explaining those inputs.

## Deployment

Replace the existing files in the GitHub repository used by Streamlit:

- `app.py`
- `requirements.txt`
- `README.md` (optional)

Keep the existing Streamlit Secret:

`THE_ODDS_API_KEY = "YOUR_KEY"`

Do not store the key in GitHub.

## Suggested validation workflow

1. Run a 14-day backtest.
2. Check that it completes successfully.
3. Run several non-overlapping 30-45 day windows.
4. Export each CSV.
5. Compare Brier score and calibration.
6. If historical odds are supported, review ROI by minimum edge.
7. Fit calibration only on one period and evaluate it on a later period.
8. Do not tune coefficients repeatedly against the same test sample.
9. Track closing-line value for live recommendations.
10. Only then consider raising confidence thresholds.

## Next technical step

A true v4 would replace the hand-set run coefficients with a trained ensemble using a durable historical database and would add:
- confirmed batter-level lineup projections
- platoon splits
- reliever availability / fatigue
- park factors
- Statcast-derived quality metrics
- historical closing odds
- scheduled model retraining
- persistent database
