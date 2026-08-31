# EDGE v4.1 — Historical Odds Diagnostics

This version fixes the silent historical-odds failure from v4.

## Main improvements
- Dedicated one-date historical odds diagnostic
- Explicit API error/status reporting
- Historical snapshots aligned to 90 minutes before each game's scheduled first pitch
- Match-rate reporting
- Dataset build stops if historical odds were requested and zero games matched
- Market benchmark remains available once historical probabilities exist

## Recommended workflow
1. Open Odds diagnostic.
2. Pick a recent completed MLB date.
3. Run diagnostic.
4. If historical odds are unavailable, the app will show the API message.
5. If available and matched, build your 90-day dataset with historical odds enabled.
6. Run walk-forward.
7. Open Market benchmark.

Keep your current Streamlit secret:
THE_ODDS_API_KEY = "YOUR_EXISTING_KEY"
