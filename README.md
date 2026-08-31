# EDGE v4 — Walk-Forward Validation Lab

This build implements the validation methodology requested after the first backtest.

## Core upgrade

Instead of evaluating/calibrating on one sample, v4 uses:

**Train -> calibrate -> freeze -> test on unseen games -> walk forward**

This is designed to reduce overfitting and provide a much more honest estimate of model quality.

## Included

### Historical dataset builder
- completed MLB games
- pre-game team and pitcher stats only
- stats stop on the day before each game
- raw model probability
- optional historical sportsbook odds

### Walk-forward testing
User controls:
- training-window size
- test-window size
- walk step
- optional probability calibration

For every unseen test window:
- fit model only on prior training data
- optionally calibrate only on training data
- freeze the model
- predict the next unseen window
- advance forward

### Evaluation
- Brier score
- log loss
- classification accuracy
- probability-band calibration
- window-by-window stability

### Market comparison
If historical odds are available:
- no-vig market benchmark
- EDGE Brier vs sportsbook Brier
- EDGE log loss vs sportsbook log loss
- flat-stake ROI by minimum edge threshold
- sample-size reporting

### Exports
- dataset CSV
- walk-forward predictions CSV
- validation summary JSON

## Important methodological notes

1. This app still uses a relatively simple MLB feature set.
2. Historical sportsbook availability depends on the user's Odds API plan.
3. A model should not be considered validated because one edge threshold shows positive ROI.
4. Look for:
   - multiple non-overlapping test windows
   - stable calibration
   - Brier/log-loss competitiveness vs market
   - adequate sample size
   - positive ROI that persists across windows
5. Do not repeatedly tune on the same test period.

## Deployment

Replace your existing:
- app.py
- requirements.txt

Keep:
THE_ODDS_API_KEY = "YOUR_EXISTING_KEY"

inside Streamlit Secrets.

## Next step after v4

If walk-forward results are competitive with the market, v5 should replace the hand-built feature weights with a richer trained model using:
- Statcast quality metrics
- confirmed batter-level lineups
- platoon splits
- true bullpen availability/fatigue
- park factors
- historical closing prices
- persistent storage
