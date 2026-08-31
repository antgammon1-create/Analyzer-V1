# EDGE MLB v2.2

This release focuses on reliability and fixes the issues exposed in v2.1.

## Major fixes

### 1. Weather unit bug fixed
Open-Meteo returns temperatures in Celsius unless a Fahrenheit unit is explicitly requested.
v2.1 displayed that Celsius value as if it were Fahrenheit.

v2.2 converts Celsius to Fahrenheit before displaying or using it.

### 2. Venue coordinate validation
MLB venue coordinates are checked for plausible U.S./Canada bounds.
A small fallback table is included for several major parks if coordinates are missing or invalid.

### 3. Weather sanity layer
Weather must:
- match first pitch within roughly 1.5 hours
- pass plausible temperature bounds
- pass a simple seasonal sanity check

If weather fails validation, it is shown with a warning and ignored by the model.

### 4. Market-disagreement penalty
v2.1 could reward a huge model-market disagreement too aggressively.

v2.2 treats a large disagreement with a liquid multi-book market as a reason for caution:
- Edge Score is reduced
- reliability can be marked CAUTION
- the user is explicitly warned to verify inputs

### 5. Explainable run projection
Every recommended market includes an expander showing the run-projection components:
- baseline
- home field
- offense
- opposing starter
- opposing bullpen proxy
- weather

### 6. Reliability labels
Each signal is marked:
- HIGH
- MEDIUM
- LOW
- CAUTION

The label is based on:
- data quality
- size of model-market disagreement
- number of books
- Edge Score

## Deployment

Replace your existing `app.py` and `requirements.txt` in the GitHub repository connected to Streamlit.

Keep your current Streamlit secret:
`THE_ODDS_API_KEY = "YOUR_EXISTING_KEY"`

Do not put your API key in GitHub.

## Still not finished

v2.2 is a research model, not a proven profitable betting system.

The next major version should be historical validation:
- historical MLB results
- historical odds
- closing lines
- calibration
- Brier score / log loss
- ROI
- CLV
- performance by market
- performance by Edge Score band
- trained coefficients / ensemble model
