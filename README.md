# EDGE v4.2 — Free Prospective Validation

This redesign removes the dependency on a paid historical-odds subscription.

## How it works

The Odds API's paid historical endpoint is no longer used.

Instead, a GitHub Action periodically calls the normal current MLB moneyline endpoint and saves a timestamped snapshot to:

`data/odds_snapshots.csv`

Over time, your own repository becomes your historical odds database.

The Streamlit app then matches completed MLB games to a pregame snapshot (preferably about 90 minutes before first pitch) and performs:

- EDGE vs market Brier score
- EDGE vs market log loss
- prospective walk-forward testing
- ROI by model edge threshold

## Why this is more honest than finding random free historical odds online

Free public historical betting datasets often have unclear timestamps, missing books, survivorship problems, or licensing issues.

v4.2 records exactly what your own app could have seen at the time.

## Files

- `app.py` — Streamlit validation app
- `collector.py` — current odds collector
- `.github/workflows/collect_odds.yml` — scheduled collector
- `data/odds_snapshots.csv` — accumulated snapshots

## Required setup

You already have THE_ODDS_API_KEY in Streamlit Secrets.

You must ALSO add the same key as a GitHub repository Actions secret named:

`THE_ODDS_API_KEY`

Do not commit the API key into a file.

## GitHub Action schedule

The included workflow runs four times per day during typical MLB windows.

It can also be run manually from:

GitHub -> Actions -> Collect MLB odds snapshots -> Run workflow

## Important limitation

This method starts collecting history from the day you install it. It cannot reconstruct old sportsbook prices.

That is a feature, not a flaw: all benchmark data is genuinely time-stamped and observable prospectively.
