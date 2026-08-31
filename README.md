# EDGE MLB v2

This is the next version of the EDGE MLB prototype.

## What changed from v1

v1 used the sportsbook market itself as the model probability, which meant it could not produce a genuine independent edge.

v2 separates:

**Sportsbook market probability**
from
**EDGE model probability**

The model currently uses:
- MLB schedule
- probable pitchers
- current-season pitcher statistics
- current-season team hitting statistics
- venue coordinates
- weather
- home-field effect
- Monte Carlo simulation
- sportsbook price

It calculates:
- model win probability
- market-implied probability
- percentage-point edge
- fair American odds
- expected value
- Edge Score

## Streamlit setup

Replace your existing `app.py` and `requirements.txt` with the files in this ZIP.

Keep your existing Streamlit Secret:

THE_ODDS_API_KEY = "YOUR_EXISTING_KEY"

Do not put the key into GitHub or app.py.

## Important

This is NOT yet a proven profitable betting model.

The coefficients are transparent research defaults. The next serious upgrade is historical backtesting/calibration using historical MLB games and odds.

After calibration, the roadmap is:

1. Confirmed lineups
2. Batter-level projections
3. Platoon splits
4. Bullpen availability and fatigue
5. Park factors
6. Better weather timing
7. Historical odds / closing-line value
8. Trained ensemble model
9. ROI and calibration dashboard
10. PGA, NFL, NCAA football
11. Correlation-aware parlay engine
