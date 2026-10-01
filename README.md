# AlgoHub NFL Game Algo

Spread / total / ML model for NFL. Data from api.nfldata.org (free, no key).

## Setup
```
pip install -r requirements.txt
python nfl_algo.py update      # first run pulls 2012-2026 (takes a while, then it's cached in data/)
python nfl_algo.py backtest    # walk-forward 2018 -> 2025, ATS / O-U record by edge size
python nfl_algo.py train       # fits models/nfl_bundle.joblib
```

## Autopilot
`python nfl_algo.py auto` does the whole week: grades last week, retrains (weekly), makes this
week's picks + props + cards + post graphics, and pushes to GitHub (site updates itself).
- Double-click `run_algo.bat` to run it now (log in `logs/auto.log`)
- Double-click `schedule_tasks.bat` once to run it every Fri 5 PM, Sun 10:30 AM, Tue 10 AM
  (PC has to be on and logged in)

## Matchup Intel
On every card: each defense's man coverage %, most-used shell, pressure % (2025) and blitz % (2026).
On receiving props: player's yards/target vs MAN and vs ZONE, and COV ✓/✗ when this defense's
coverage mix helps/hurts him. Info only, not in the model yet (not enough seasons to validate).

## Post graphics
`predict` writes `posts/*_plays.png`, `*_props.png`, `*_tds.png`. Also on the site's Posts tab.

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

## Player props
Built automatically by `train` / `predict`. Tap **PLAYER PROPS** on any game card to open them.
- Markets: pass yds, rush yds, rec yds, receptions, anytime TD
- Stats: nflverse weekly player stats (free, pulled from GitHub, includes 2025 + 2026)
- Lines: Odds API player props. Pulled once per week and cached in `data/` so reruns are free.
  Re-pull fresh lines (costs credits, ~5 per game): `python nfl_algo.py predict --week 4 --refresh-lines`
- No `ODDS_API_KEY` = projections only, no picks
- Graded with `grade` like everything else (DNP = push)

## Starting QBs
Pulled from the nflverse schedule automatically. Only use `QB_OVERRIDES` at the top of
`nfl_algo.py` if the schedule hasn't caught up to news yet.

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
