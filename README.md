
# EDGE Sports Betting Analyzer — MVP v1

A mobile-friendly Streamlit prototype for a quantitative sports betting app covering:

- MLB
- PGA Tour
- NFL
- NCAA Football

## What is implemented

- Mobile-friendly dashboard
- Sport selector
- Demo data so the app works immediately
- Optional live odds integration through The Odds API
- American odds -> implied probability
- Fair price calculation
- Expected value calculation
- Edge Score (0-100)
- Confidence score
- Bet / Lean / Pass classification
- Parlay Lab prototype
- PGA dark-horse board
- AI Analyst interface placeholder

## Important

The current v1 is an application **shell + analytics framework**, not a finished predictive model.

The demo probabilities are illustrative. They must NOT be used as real betting advice.

The next development stage is to replace the placeholder probability adjustments with trained, sport-specific models and historical backtesting.

## Run on a computer

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL shown by Streamlit.

## Live odds

Create an environment variable:

```bash
export THE_ODDS_API_KEY="YOUR_KEY"
```

Or add this to Streamlit secrets:

```toml
THE_ODDS_API_KEY = "YOUR_KEY"
```

The app uses the key only to retrieve live odds. You should verify the provider's current coverage, terms, limits, and pricing before deployment.

## Put it on an iPhone

The easiest first deployment is a hosted Streamlit app. Once deployed, open it in Safari and use:

Share -> Add to Home Screen

That creates an app-like icon on the iPhone.

For a true App Store product, the next phase should put this analytics engine behind a proper iOS/PWA front end and a secure backend. API keys should never be shipped inside an iPhone client.

## Production roadmap

### v2 — MLB
- Historical MLB database
- Starting pitcher projections
- Lineup projections
- Bullpen availability
- Park factors
- Weather by stadium
- Platoon splits
- xwOBA / wRC+ / SIERA / FIP
- Monte Carlo game simulation
- Moneyline / run line / total probabilities
- Historical odds and closing line
- Calibration
- Backtesting
- CLV tracking

### v3 — PGA
- Player database
- Strokes gained features
- Course-fit model
- Weather/wave model
- Win/top-5/top-10/top-20 probabilities
- Dark Horse model

### v4 — NFL
- EPA
- success rate
- QB model
- injuries
- matchup model
- weather
- spread / total / moneyline probabilities

### v5 — NCAA Football
- talent
- transfers
- returning production
- coaching
- opponent-adjusted efficiency
- matchup model

### v6 — Parlays
- Joint probability
- Correlation matrix
- Same-game parlay modeling
- Price shopping
- EV optimization

### v7 — AI Analyst
- Natural-language questions
- Explainable recommendations
- "Why do you like this?"
- "What changed?"
- "What is the minimum price?"
- "Build conservative/balanced/aggressive parlay"

### v8 — Tracking
- Every recommendation stored
- ROI
- units
- CLV
- calibration
- performance by sport / market / Edge Score band
