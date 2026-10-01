# AlgoHub NFL Game Algo

Spread / total / ML model for NFL. Data from api.nfldata.org (free, no key).

## Setup
```
pip install -r requirements.txt
python nfl_algo.py update      # first run pulls 2012-2026 (takes a while, then it's cached in data/)
python nfl_algo.py backtest    # walk-forward 2018 -> 2025, ATS / O-U record by edge size
python nfl_algo.py train       # fits models/nfl_bundle.joblib
```

## Weekly
```
python nfl_algo.py predict --week 5   # picks/2026_wk05.csv + cards/2026_wk05.html
python nfl_algo.py grade --week 5     # after games finish -> tracker.csv
python nfl_algo.py record             # season record by market + tier
```
Retrain (`train`) every week or two so the model sees new games.

## Website (Streamlit)
Local preview:
```
streamlit run app.py
```
Tabs: Cards (this week), Best Bets (plays ranked by tier/edge), Record (tracker + units curve),
Backtest. It only reads `picks/`, `tracker.csv`, `backtest_results.csv`, so the site never
pulls data or trains anything itself.

Go live (free): push this folder to GitHub (data/ and models/ are gitignored), then
share.streamlit.io -> New app -> pick the repo -> main file `app.py`.

Weekly loop: `predict` locally -> `git add picks tracker.csv` -> `git commit` -> `git push`.
Site updates on its own.

## Live lines + moneyline (optional)
Set `ODDS_API_KEY` as an environment variable and `predict` swaps in live consensus
spread/total and adds ML value picks. Without it, it uses the spread/total from nfldata.

## How it works
- Team features = this season's prior games only (EPA/play off + def, pass/rush split,
  success rate, explosives, turnovers, points, pace), shrunk toward league avg so a
  3-4 game sample doesn't run wild. No prior-season team data.
- QB: starter's season-to-date EPA/att + QB change flag. Upcoming starter = last week's
  starter; override at the top of `nfl_algo.py` with `QB_OVERRIDES = {"NYJ": "Taylor"}`.
- Context: rest, short week, bye, dome, temp, wind, division, travel, international.
- Two HistGradientBoostingRegressors: home margin + total. Win prob from margin.
- Tiers (points of edge vs Vegas): spread LEAN 2 / PLAY 3 / HAMMER 4.5,
  total LEAN 2.5 / PLAY 4 / HAMMER 6. Tune these after you see the backtest.

## Knobs (top of nfl_algo.py)
`FIRST_SEASON`, `MIN_WEEK_TRAIN`, `SPREAD_TIERS`, `TOTAL_TIERS`, `ML_EDGE`, `K_*` (shrinkage).
