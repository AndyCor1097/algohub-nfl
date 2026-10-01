#!/usr/bin/env python3
"""
NFL Game Algo - The AlgoHub
Data:  api.nfldata.org (nflverse gold layer, no key)
Lines: spread_line / total_line from the games endpoint; optional live lines + ML
       from The Odds API if ODDS_API_KEY is set as an env var.

Model: two HistGradientBoostingRegressors (home margin, game total).
       Team features are built ONLY from the current season's prior games,
       shrunk toward league average so 3-4 game samples don't run wild.

Usage:
  python nfl_algo.py update              # pull / refresh data cache
  python nfl_algo.py backtest            # walk-forward backtest (2018 -> last season)
  python nfl_algo.py train               # fit on all history, save models/
  python nfl_algo.py predict --week 5    # slate -> picks/*.csv + cards/*.html
  python nfl_algo.py grade --week 4      # grade a finished week into tracker.csv
"""
import argparse
import io
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_predict

# when run as `python nfl_algo.py`, make `import nfl_algo` (in props/intel) return THIS module,
# not a second fresh copy with empty state
if __name__ == "__main__":
    sys.modules.setdefault("nfl_algo", sys.modules[__name__])

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
CURRENT_SEASON = 2026
FIRST_SEASON = 2012          # training history start
BACKTEST_START = 2018
MIN_WEEK_TRAIN = 3           # don't train on weeks with ~no in-season sample

API = "https://api.nfldata.org/v1"
ODDS_URL = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds/"

ROOT = Path(__file__).resolve().parent
DATA, MODELS, PICKS, CARDS = (ROOT / d for d in ("data", "models", "picks", "cards"))
TRACKER = ROOT / "tracker.csv"

# Edge thresholds (points). Below the lowest = NO PLAY.
SPREAD_TIERS = [(6.0, "HAMMER"), (4.5, "PLAY"), (3.0, "LEAN")]  # sides are sharp: only fire on big gaps
TOTAL_TIERS = [(6.0, "HAMMER"), (4.0, "PLAY"), (2.5, "LEAN")]
ML_EDGE = 0.04               # model win prob minus no-vig implied prob

# Shrinkage strength (in plays / games). Bigger = pulls harder toward league avg.
K_PLAYS, K_PASS, K_RUSH, K_GAMES, K_QB = 170, 110, 90, 3, 150

# Starting QBs: by default each team keeps last week's starter. List QB changes here
# (full name). A QB with no snaps this season gets rated replacement level.
# Clear an entry once that QB has started a game.
QB_OVERRIDES = {"TB": "Jalon Daniels"}
QB_FROM_SCHEDULE = False   # True = also trust nflverse schedule's projected QBs (often stale)
REPLACEMENT_QB_PENALTY = 0.12   # EPA/play below league avg for a QB with no stats
QB_ACTIVE = dict(QB_OVERRIDES)  # schedule starters + manual overrides, filled in by predict()
LAST_FRAME = None               # game frame from the last build_all(), reused by props

TZ = {  # hours west of ET, for travel
    **dict.fromkeys("BUF MIA NE NYJ BAL CIN CLE PIT IND JAX NYG PHI WAS ATL CAR TB DET".split(), 0),
    **dict.fromkeys("HOU KC TEN DAL CHI GB MIN NO STL".split(), 1),
    **dict.fromkeys("DEN ARI".split(), 2),
    **dict.fromkeys("LA LAC LV SF SEA OAK SD".split(), 3),
}
DIVISION = {t: d for d, ts in {
    "AFCE": "BUF MIA NE NYJ", "AFCN": "BAL CIN CLE PIT", "AFCS": "HOU IND JAX TEN",
    "AFCW": "DEN KC LV LAC OAK SD", "NFCE": "DAL NYG PHI WAS", "NFCN": "CHI DET GB MIN",
    "NFCS": "ATL CAR NO TB", "NFCW": "ARI LA SF SEA STL"}.items() for t in ts.split()}
INTL = ["wembley", "tottenham", "twickenham", "allianz", "deutsche bank", "frankfurt",
        "azteca", "estadio", "croke", "bernab", "maracan", "corinthians", "melbourne",
        "olympiastadion", "dublin", "madrid", "munich", "london", "sao paulo"]

ODDS_NAMES = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Los Angeles Rams": "LA", "Los Angeles Chargers": "LAC",
    "Las Vegas Raiders": "LV", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "Seattle Seahawks": "SEA", "San Francisco 49ers": "SF", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}
TEAM_NAMES = {v: k for k, v in ODDS_NAMES.items()}

TEAM_FEATS = ["off_epa", "off_pass", "off_rush", "off_succ", "off_expl", "off_to",
              "def_epa", "def_pass", "def_rush", "def_succ", "def_expl", "def_to",
              "ppg", "papg", "pace", "gp", "rest", "qb_epa", "qb_change"]
GAME_FEATS = ["h_mu", "a_mu", "net_mu", "tot_mu", "rest_diff", "qb_diff", "ppg_net",
              "pace_sum", "dome", "temp", "wind", "div", "is_intl", "tz_travel", "week"]
FEATURES = [f"h_{c}" for c in TEAM_FEATS] + [f"a_{c}" for c in TEAM_FEATS] + GAME_FEATS

LEAGUE = {}  # league-average rates, filled in build_frame (used for display centering)


# ----------------------------------------------------------------------------
# DATA
# ----------------------------------------------------------------------------
SESSION = requests.Session()


def _get(path, params):
    for attempt in range(6):
        try:
            r = SESSION.get(f"{API}{path}", params=params, timeout=45)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if attempt == 5:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError(f"API failed: {path} {params}")


def fetch_all(path, params, page):
    rows, offset = [], 0
    while True:
        js = _get(path, {**params, "limit": page, "offset": offset})
        data = js.get("data", [])
        rows.extend(data)
        offset += len(data)
        if not data or offset >= js.get("total", 0):
            break
    return pd.DataFrame(rows)


def _cached(name, fetch, refresh):
    f = DATA / f"{name}.csv.gz"
    if f.exists() and not refresh:
        try:
            return pd.read_csv(f, low_memory=False)
        except pd.errors.EmptyDataError:
            f.unlink()                      # old blank file -> refetch
    try:
        df = fetch()
    except (requests.RequestException, OSError, RuntimeError) as ex:
        if f.exists():   # source down -> keep going on the last good copy
            print(f"  ! {name}: couldn't refresh ({type(ex).__name__}), using saved copy", flush=True)
            return pd.read_csv(f, low_memory=False)
        raise
    if df.empty:
        if refresh and f.exists():
            return pd.read_csv(f, low_memory=False)
        print(f"  ! {name}: API returned 0 rows (not cached, will retry next run)", flush=True)
        return df
    DATA.mkdir(exist_ok=True)
    df.to_csv(f, index=False)
    print(f"  {name}: {len(df):,} rows", flush=True)
    return df


def load_games(season, refresh=False):
    return _cached(f"games_{season}", lambda: fetch_all("/games", {"season": season}, 1000), refresh)


def load_plays(season, refresh=False):
    cols = ["game_id", "season", "week", "posteam", "defteam", "play_type",
            "yards_gained", "turnover", "epa"]

    def fetch():
        parts = []
        for pt in ("pass", "run"):
            print(f"  plays {season} {pt}...", flush=True)
            parts.append(fetch_all("/plays", {"season": season, "play_type": pt}, 500))
        df = pd.concat(parts, ignore_index=True)
        return df[[c for c in cols if c in df.columns]]
    return _cached(f"plays_{season}", fetch, refresh)


def load_passing(season, refresh=False):
    def fetch():
        df = fetch_all("/stats/passing", {"season": season}, 1000)
        if df.empty:   # nfldata's box scores stop at 2024 -> nflverse weekly player stats
            ps = load_player_week(season, refresh)
            if not ps.empty:
                ps = ps[pd.to_numeric(ps.attempts, errors="coerce").fillna(0) > 0]
                df = pd.DataFrame({"player_id": ps.player_id, "player_name": ps.player_display_name,
                                   "recent_team": ps.team, "season": ps.season, "week": ps.week,
                                   "season_type": ps.season_type, "attempts": ps.attempts,
                                   "passing_epa": ps.passing_epa})
                print(f"  passing_{season}: using nflverse player stats", flush=True)
        if df.empty:
            df = load_qbr_week(season)
            if not df.empty:
                print(f"  passing_{season}: using weekly QBR", flush=True)
        return df
    return _cached(f"passing_{season}", fetch, refresh)


NFLVERSE = "https://github.com/nflverse/nflverse-data/releases/download"


def _nflverse_csv(path):
    try:
        r = requests.get(f"{NFLVERSE}/{path}", timeout=120)
        if r.status_code != 200:
            return pd.DataFrame()
        return pd.read_csv(io.BytesIO(r.content), low_memory=False)
    except (requests.RequestException, pd.errors.ParserError, pd.errors.EmptyDataError) as ex:
        print(f"  ! nflverse {path}: {ex}", flush=True)
        return pd.DataFrame()


def load_player_week(season, refresh=False):
    return _cached(f"pstats_{season}",
                   lambda: _nflverse_csv(f"stats_player/stats_player_week_{season}.csv"), refresh)


def load_schedule(refresh=False):
    """nflverse schedule: kickoff times (ET) + projected starting QBs, updated through the week."""
    return _cached("schedule", lambda: _nflverse_csv("schedules/games.csv"), refresh)


def load_injuries(refresh=False):
    """Official injury reports (practice participation + game status), current season."""
    return _cached(f"injuries_{CURRENT_SEASON}",
                   lambda: _nflverse_csv(f"injuries/injuries_{CURRENT_SEASON}.csv"), refresh)


def load_depth(refresh=False):
    """Latest daily depth chart snapshot (big file upstream, only the newest snapshot is kept)."""
    def fetch():
        d = _nflverse_csv(f"depth_charts/depth_charts_{CURRENT_SEASON}.csv")
        if d.empty or "dt" not in d.columns:
            return d
        d = d[d.dt == d.groupby("team").dt.transform("max")]
        return d.drop_duplicates(["team", "gsis_id", "pos_abb"])
    return _cached(f"depth_{CURRENT_SEASON}", fetch, refresh)


OUT_STATUSES = {"Out", "Doubtful"}


def injury_report(week):
    """{gsis_id: dict(name, team, pos, status, practice)} for this week's report."""
    inj = load_injuries()
    if inj.empty:
        return {}
    w = inj[(inj.season == CURRENT_SEASON) & (inj.week == week)].copy()
    w["team"] = w.team.replace(TEAM_FIX)
    return {r.gsis_id: dict(name=r.full_name, team=r.team, pos=r.position,
                            status=r.report_status if isinstance(r.report_status, str) else "",
                            practice=r.practice_status if isinstance(r.practice_status, str) else "")
            for r in w.itertuples()}


def auto_qbs(week, inj):
    """If last week's starter is ruled Out/Doubtful, next healthy QB on the depth chart starts."""
    q = load_passing(CURRENT_SEASON)
    if q.empty:
        return {}, {}
    q = q[(q.season == CURRENT_SEASON) & (q.week < week)].copy()
    q["recent_team"] = q.recent_team.replace(TEAM_FIX)
    q = q.sort_values(["week", "attempts"]).groupby("recent_team").tail(1)
    depth = load_depth()
    if not depth.empty:
        depth = depth[depth.pos_abb == "QB"].copy()
        depth["team"] = depth.team.replace(TEAM_FIX)
    out, notes = {}, {}
    for r in q.itertuples():
        st = inj.get(r.player_id, {})
        if st.get("status") in OUT_STATUSES:
            nxt = None
            if not depth.empty:
                for d in depth[depth.team == r.recent_team].sort_values("pos_rank").itertuples():
                    if d.gsis_id != r.player_id and inj.get(d.gsis_id, {}).get("status") not in OUT_STATUSES:
                        nxt = d.player_name
                        break
            if nxt:
                out[r.recent_team] = nxt
                print(f"  injury: {r.player_name} ({r.recent_team}) {st['status']} -> {nxt} starts")
        elif st.get("practice", "").startswith("Did Not"):
            notes[r.recent_team] = f"{r.player_name} DNP"
    return out, notes


ESPN_ABBR = {"WSH": "WAS", "LAR": "LA", "JAC": "JAX", "LV": "LV", "OAK": "LV"}


def load_qbr_week(season):
    """/qbr/week only holds the current season (ignores ?season), so filter here.
    Mapped onto the passing-stats columns: qb_plays ~ attempts, epa_total ~ passing_epa."""
    q = fetch_all("/qbr/week", {"season": season}, 1000)
    if q.empty or "season" not in q.columns:
        return pd.DataFrame()
    q = q[pd.to_numeric(q.season, errors="coerce") == season]
    if q.empty:
        return pd.DataFrame()
    st = q.get("season_type", pd.Series("Regular", index=q.index)).astype(str)
    return pd.DataFrame({
        "player_id": "espn_" + q.player_id.astype(str),
        "player_name": q.name_display,
        "recent_team": q.team_abb.replace(ESPN_ABBR),
        "season": season,
        "week": pd.to_numeric(q.week_num, errors="coerce"),
        "season_type": np.where(st.str.lower().str.startswith("post"), "POST", "REG"),
        "attempts": pd.to_numeric(q.qb_plays, errors="coerce"),
        "passing_epa": pd.to_numeric(q.epa_total, errors="coerce"),
    }).dropna(subset=["week", "attempts"])


def update(refresh_current=True):
    for s in range(FIRST_SEASON, CURRENT_SEASON + 1):
        refresh = refresh_current and s == CURRENT_SEASON
        print(f"season {s}{' (refresh)' if refresh else ''}", flush=True)
        load_games(s, refresh)
        load_plays(s, refresh)
        load_passing(s, refresh)
    import props
    print("player stats (props)", flush=True)
    for s in range(props.PROPS_FIRST_SEASON, CURRENT_SEASON + 1):
        load_player_week(s, refresh_current and s == CURRENT_SEASON)
    load_schedule(refresh_current)


def load_all():
    g, p, q = [], [], []
    for s in range(FIRST_SEASON, CURRENT_SEASON + 1):
        g.append(load_games(s)); p.append(load_plays(s)); q.append(load_passing(s))
    qcols = ["player_id", "player_name", "recent_team", "season", "week", "season_type",
             "attempts", "passing_epa"]
    q = [x for x in q if not x.empty] or [pd.DataFrame(columns=qcols)]
    return (pd.concat(g, ignore_index=True), pd.concat(p, ignore_index=True),
            pd.concat(q, ignore_index=True))


# ----------------------------------------------------------------------------
# FEATURES
# ----------------------------------------------------------------------------
def _boolint(s):
    return s.map({True: 1, False: 0, "True": 1, "False": 0, 1: 1, 0: 0}).fillna(0).astype(int)


def team_game_stats(plays):
    p = plays.dropna(subset=["epa", "posteam", "defteam"]).copy()
    p["is_pass"] = (p.play_type == "pass").astype(int)
    p["is_rush"] = (p.play_type == "run").astype(int)
    p["succ"] = (p.epa > 0).astype(int)
    yg = p.yards_gained.fillna(0)
    p["expl"] = (((p.is_pass == 1) & (yg >= 20)) | ((p.is_rush == 1) & (yg >= 10))).astype(int)
    p["to"] = _boolint(p.turnover)
    p["pass_epa"] = p.epa * p.is_pass
    p["rush_epa"] = p.epa * p.is_rush
    agg = dict(n=("epa", "size"), epa=("epa", "sum"), succ=("succ", "sum"),
               pass_n=("is_pass", "sum"), pass_epa=("pass_epa", "sum"),
               rush_n=("is_rush", "sum"), rush_epa=("rush_epa", "sum"),
               expl=("expl", "sum"), to=("to", "sum"))
    off = p.groupby(["game_id", "posteam"]).agg(**agg).reset_index()
    de = p.groupby(["game_id", "defteam"]).agg(**agg).reset_index()
    off.columns = ["game_id", "team"] + [f"o_{c}" for c in agg]
    de.columns = ["game_id", "team"] + [f"d_{c}" for c in agg]
    return off.merge(de, on=["game_id", "team"], how="outer")


RATE_SPECS = [  # name, numerator, denominator, K
    ("off_epa", "o_epa", "o_n", K_PLAYS), ("off_pass", "o_pass_epa", "o_pass_n", K_PASS),
    ("off_rush", "o_rush_epa", "o_rush_n", K_RUSH), ("off_succ", "o_succ", "o_n", K_PLAYS),
    ("off_expl", "o_expl", "o_n", K_PLAYS), ("off_to", "o_to", "o_n", K_PLAYS),
    ("def_epa", "d_epa", "d_n", K_PLAYS), ("def_pass", "d_pass_epa", "d_pass_n", K_PASS),
    ("def_rush", "d_rush_epa", "d_rush_n", K_RUSH), ("def_succ", "d_succ", "d_n", K_PLAYS),
    ("def_expl", "d_expl", "d_n", K_PLAYS), ("def_to", "d_to", "d_n", K_PLAYS),
]


def build_frame(games, plays, passing):
    """One row per game; every feature uses only that season's games before kickoff."""
    games = games.dropna(subset=["home_team", "away_team"]).copy()
    games["gameday"] = pd.to_datetime(games.gameday)
    tg = team_game_stats(plays)

    # --- team-game long table, ordered by date within team-season
    h = games.assign(team=games.home_team, opp=games.away_team, is_home=1,
                     pf=games.home_score, pa=games.away_score)
    a = games.assign(team=games.away_team, opp=games.home_team, is_home=0,
                     pf=games.away_score, pa=games.home_score)
    long = pd.concat([h, a], ignore_index=True)[
        ["game_id", "season", "week", "gameday", "team", "opp", "is_home", "pf", "pa"]]
    long = long.merge(tg, on=["game_id", "team"], how="left")
    long = long.sort_values(["team", "season", "gameday"]).reset_index(drop=True)

    stat_cols = [c for c in tg.columns if c not in ("game_id", "team")]
    long[stat_cols] = long[stat_cols].fillna(0.0)
    played = long.pf.notna()
    long["played"] = played.astype(int)
    long["win"] = (played & (long.pf > long.pa)).astype(int)
    long["loss"] = (played & (long.pf < long.pa)).astype(int)
    long["tie"] = (played & (long.pf == long.pa)).astype(int)
    long["pf0"], long["pa0"] = long.pf.fillna(0), long.pa.fillna(0)

    cum = stat_cols + ["played", "win", "loss", "tie", "pf0", "pa0"]
    grp = long.groupby(["team", "season"], sort=False)
    prior = grp[cum].cumsum() - long[cum]          # strictly before this game

    for name, num, den, k in RATE_SPECS:
        mu = tg[num].sum() / max(tg[den].sum(), 1)
        LEAGUE[name] = mu
        long[name] = (prior[num] + k * mu) / (prior[den] + k)

    pts_mu = pd.concat([games.home_score, games.away_score]).mean()
    pace_mu = tg.o_n.mean()
    LEAGUE["ppg"], LEAGUE["pace"] = pts_mu, pace_mu
    long["ppg"] = (prior.pf0 + K_GAMES * pts_mu) / (prior.played + K_GAMES)
    long["papg"] = (prior.pa0 + K_GAMES * pts_mu) / (prior.played + K_GAMES)
    long["pace"] = (prior.o_n + K_GAMES * pace_mu) / (prior.played + K_GAMES)
    long["gp"] = prior.played
    w, l, t = prior.win.astype(int), prior.loss.astype(int), prior.tie.astype(int)
    long["rec"] = w.astype(str) + "-" + l.astype(str) + np.where(t > 0, "-" + t.astype(str), "")
    long["rest"] = grp.gameday.diff().dt.days.fillna(7).clip(upper=21)

    long = _add_qb(long, passing)

    # --- home / away merge
    keep = ["game_id"] + TEAM_FEATS + ["rec", "qb_name"]
    hh = long[long.is_home == 1][keep].rename(columns={c: f"h_{c}" for c in keep[1:]})
    aa = long[long.is_home == 0][keep].rename(columns={c: f"a_{c}" for c in keep[1:]})
    df = games.merge(hh, on="game_id").merge(aa, on="game_id")

    roof = df.roof.fillna("outdoors").astype(str).str.lower()
    df["dome"] = roof.isin(["dome", "closed"]).astype(int)
    df["temp"] = np.where(df.dome == 1, 70, pd.to_numeric(df.temp, errors="coerce").fillna(60))
    df["wind"] = np.where(df.dome == 1, 0, pd.to_numeric(df.wind, errors="coerce").fillna(8))
    same_div = df.home_team.map(DIVISION) == df.away_team.map(DIVISION)
    df["div"] = np.maximum(_boolint(df.div_game), same_div.astype(int))  # nfldata leaves 2026 blank
    stad = df.stadium.fillna("").astype(str).str.lower()
    df["is_intl"] = stad.apply(lambda s: int(any(k in s for k in INTL)))
    df["tz_travel"] = (df.home_team.map(TZ).fillna(0) - df.away_team.map(TZ).fillna(0)).abs()

    df["h_mu"] = df.h_off_epa + df.a_def_epa
    df["a_mu"] = df.a_off_epa + df.h_def_epa
    df["net_mu"] = df.h_mu - df.a_mu
    df["tot_mu"] = df.h_mu + df.a_mu
    df["rest_diff"] = df.h_rest - df.a_rest
    df["qb_diff"] = df.h_qb_epa - df.a_qb_epa
    df["ppg_net"] = (df.h_ppg - df.h_papg) - (df.a_ppg - df.a_papg)
    df["pace_sum"] = df.h_pace + df.a_pace
    # display-only matchup strip (centered on league avg: + = offense edge)
    for s, o in (("h", "a"), ("a", "h")):
        df[f"{s}_pass_mu"] = df[f"{s}_off_pass"] + df[f"{o}_def_pass"] - 2 * LEAGUE["off_pass"]
        df[f"{s}_rush_mu"] = df[f"{s}_off_rush"] + df[f"{o}_def_rush"] - 2 * LEAGUE["off_rush"]

    df["margin"] = df.home_score - df.away_score
    df["total"] = df.home_score + df.away_score
    return df


def _add_qb(long, passing):
    q = passing.copy()
    q = q[pd.to_numeric(q.attempts, errors="coerce").fillna(0) > 0]
    if q.empty:
        print("  ! no QB passing data at all, running without QB features", flush=True)
        LEAGUE["qb_epa"] = 0.0
        long["qb_id"], long["qb_name"] = np.nan, ""
        long["qb_change"], long["qb_epa"] = 0, 0.0
        return long
    if "season_type" in q.columns:
        q = q[q.season_type.isin(["REG", "POST"]) | q.season_type.isna()]
    q["attempts"] = pd.to_numeric(q.attempts)
    q["passing_epa"] = pd.to_numeric(q.passing_epa, errors="coerce").fillna(0)

    # starter = most attempts for the team that week
    idx = q.groupby(["season", "week", "recent_team"]).attempts.idxmax()
    st = q.loc[idx, ["season", "week", "recent_team", "player_id", "player_name"]].rename(
        columns={"recent_team": "team", "player_id": "qb_id", "player_name": "qb_name"})
    long = long.merge(st, on=["season", "week", "team"], how="left")

    # upcoming games: assume last starter unless overridden
    g = long.groupby(["team", "season"], sort=False)
    long["qb_id"] = g.qb_id.ffill()
    long["qb_name"] = g.qb_name.ffill()
    unplayed = (long.played == 0) & (long.season == CURRENT_SEASON)
    no_stats = pd.Series(False, index=long.index)
    cur = q[q.season == CURRENT_SEASON]
    for team, name in QB_ACTIVE.items():
        m = unplayed & (long.team == team)
        if not m.any() or not isinstance(name, str) or not name.strip():
            continue
        now = long.loc[m, "qb_name"].dropna()
        if len(now) and now.iloc[0].lower() == name.lower():
            continue   # already the guy
        hit = cur[cur.player_name.str.contains(name, case=False, na=False, regex=False)]
        if hit.empty:   # hasn't played this year -> replacement-level QB
            print(f"  QB {team}: {name} starting (no {CURRENT_SEASON} stats, rated replacement level)")
            long.loc[m, "qb_id"], long.loc[m, "qb_name"] = f"override_{team}_{name}", name
            no_stats |= m
        else:
            row = hit.sort_values("week").iloc[-1]
            print(f"  QB {team}: {row.player_name} starting")
            long.loc[m, "qb_id"], long.loc[m, "qb_name"] = row.player_id, row.player_name

    prev = long.groupby(["team", "season"], sort=False).qb_id.shift(1)
    long["qb_change"] = (prev.notna() & long.qb_id.notna() & (prev != long.qb_id)).astype(int)

    # starter's season-to-date EPA/att before this week (any team), shrunk
    mu = q.passing_epa.sum() / q.attempts.sum()
    LEAGUE["qb_epa"] = mu
    qc = (q.groupby(["player_id", "season", "week"])[["passing_epa", "attempts"]].sum()
          .groupby(level=[0, 1]).cumsum().reset_index()
          .rename(columns={"player_id": "qb_id", "passing_epa": "cum_epa", "attempts": "cum_att"}))
    left = long[["season", "week", "qb_id"]].reset_index()
    left["week"] = left.week.astype(int)
    qc["week"] = qc.week.astype(int)
    has = left.qb_id.notna()
    lk = pd.merge_asof(left[has].sort_values("week"), qc.sort_values("week"),
                       on="week", by=["qb_id", "season"], allow_exact_matches=False)
    lk = lk.set_index("index")
    long["cum_epa"] = lk.cum_epa.reindex(long.index).fillna(0)
    long["cum_att"] = lk.cum_att.reindex(long.index).fillna(0)
    long["qb_epa"] = (long.cum_epa + K_QB * mu) / (long.cum_att + K_QB)
    long.loc[no_stats, "qb_epa"] = mu - REPLACEMENT_QB_PENALTY
    return long


def build_all():
    g, p, q = load_all()
    for col in ("home_team", "away_team"):
        g[col] = g[col].replace(TEAM_FIX)
    for col in ("posteam", "defteam"):
        p[col] = p[col].replace(TEAM_FIX)
    if "recent_team" in q.columns:
        q["recent_team"] = q.recent_team.replace(TEAM_FIX)
    global LAST_FRAME
    LAST_FRAME = build_frame(g, p, q)
    return LAST_FRAME


# nfldata / ESPN use a few different codes than nflverse -> one canonical code
TEAM_FIX = {"WSH": "WAS", "LAR": "LA", "JAC": "JAX"}
ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"


def _to_eastern(utc):
    """UTC timestamp -> US Eastern (DST: 2nd Sun Mar 2am -> 1st Sun Nov 2am). No tz packages needed."""
    y = utc.year
    mar = pd.Timestamp(y, 3, 8) + pd.Timedelta(days=(6 - pd.Timestamp(y, 3, 8).weekday()) % 7)
    nov = pd.Timestamp(y, 11, 1) + pd.Timedelta(days=(6 - pd.Timestamp(y, 11, 1).weekday()) % 7)
    dst = (mar + pd.Timedelta(hours=7)) <= utc < (nov + pd.Timedelta(hours=6))
    return utc - pd.Timedelta(hours=4 if dst else 5)


def fetch_kickoffs(season, week):
    """{(home, away): (gameday_ET, 'HH:MM' ET, venue)} from ESPN's free scoreboard."""
    try:
        r = requests.get(ESPN_SCOREBOARD, params=dict(dates=season, seasontype=2, week=week), timeout=20)
        r.raise_for_status()
        events = r.json().get("events", [])
    except (requests.RequestException, ValueError) as ex:
        print(f"  ! couldn't get kickoff times from ESPN ({ex})", flush=True)
        return {}
    out = {}
    for ev in events:
        comp = (ev.get("competitions") or [{}])[0]
        teams = {c.get("homeAway"): c.get("team", {}).get("abbreviation") for c in comp.get("competitors", [])}
        h, a = (TEAM_FIX.get(teams.get(k), teams.get(k)) for k in ("home", "away"))
        if not (h and a and ev.get("date")):
            continue
        et = _to_eastern(pd.Timestamp(ev["date"]).tz_localize(None))
        out[(h, a)] = (str(et.date()), et.strftime("%H:%M"), comp.get("venue", {}).get("fullName"))
    return out


def trainable(df):
    return df[df.home_score.notna() & (df.week >= MIN_WEEK_TRAIN)].copy()


# ----------------------------------------------------------------------------
# MODEL
# ----------------------------------------------------------------------------
def make_model():
    # conservative on purpose: early-season samples are tiny and a looser model chases noise
    # (on a 2025 holdout this cut fake 10+ pt edges and gave the best totals accuracy)
    return HistGradientBoostingRegressor(
        learning_rate=0.03, max_iter=150, max_depth=3, min_samples_leaf=100,
        l2_regularization=5.0, early_stopping=False, random_state=7)


def fit(tr):
    mm, mt = make_model(), make_model()
    mm.fit(tr[FEATURES], tr.margin)
    mt.fit(tr[FEATURES], tr.total)
    return mm, mt


def train():
    frame = build_all()
    df = trainable(frame)
    print(f"training on {len(df):,} games ({df.season.min()}-{df.season.max()})")
    mm, mt = fit(df)
    kf = KFold(5, shuffle=True, random_state=7)
    oof_m = cross_val_predict(make_model(), df[FEATURES], df.margin, cv=kf)
    oof_t = cross_val_predict(make_model(), df[FEATURES], df.total, cv=kf)
    sig_m, sig_t = float(np.std(df.margin - oof_m)), float(np.std(df.total - oof_t))
    MODELS.mkdir(exist_ok=True)
    joblib.dump(dict(margin=mm, total=mt, features=FEATURES, sigma_margin=sig_m,
                     sigma_total=sig_t, trained_through=int(df.season.max()),
                     n=len(df)), MODELS / "nfl_bundle.joblib")
    print(f"saved models/nfl_bundle.joblib | CV MAE margin {np.mean(np.abs(df.margin - oof_m)):.2f} "
          f"total {np.mean(np.abs(df.total - oof_t)):.2f} | sigma margin {sig_m:.2f}")
    import props
    try:
        props.train(frame)
    except Exception as ex:
        print(f"  ! props model not trained: {type(ex).__name__}: {ex}")


def _ats(edge, res, thr):
    sel = edge.abs() >= thr
    e, r = edge[sel], res[sel]
    win = int(((np.sign(e) == np.sign(r)) & (r != 0)).sum())
    push = int((r == 0).sum())
    loss = int(sel.sum()) - win - push
    pct = win / max(win + loss, 1)
    units = win * (100 / 110) - loss
    return win, loss, push, pct, units


def backtest(start=BACKTEST_START):
    df = trainable(build_all())
    out = []
    for s in range(start, CURRENT_SEASON):
        tr, te = df[df.season < s], df[(df.season == s) & (df.game_type == "REG")].copy()
        if te.empty or tr.empty:
            continue
        mm, mt = fit(tr)
        te["pred_margin"], te["pred_total"] = mm.predict(te[FEATURES]), mt.predict(te[FEATURES])
        out.append(te)
        print(f"  {s}: {len(te)} games", flush=True)
    bt = pd.concat(out, ignore_index=True)
    bt.to_csv(ROOT / "backtest_results.csv", index=False)

    m = bt.dropna(subset=["spread_line"])
    print(f"\nMARGIN MAE  model {np.mean(np.abs(m.margin - m.pred_margin)):.2f} | "
          f"vegas {np.mean(np.abs(m.margin - m.spread_line)):.2f}")
    t = bt.dropna(subset=["total_line"])
    print(f"TOTAL  MAE  model {np.mean(np.abs(t.total - t.pred_total)):.2f} | "
          f"vegas {np.mean(np.abs(t.total - t.total_line)):.2f}\n")

    for label, d, edge, res, thrs in (
        ("SPREAD", m, m.pred_margin - m.spread_line, m.margin - m.spread_line, [0, 1, 2, 3, 4.5]),
        ("TOTAL", t, t.pred_total - t.total_line, t.total - t.total_line, [0, 1.5, 2.5, 4, 6]),
    ):
        print(f"{label:<7}{'edge>=':>7}{'W-L-P':>14}{'win%':>8}{'units':>9}")
        for thr in thrs:
            w, l, p, pct, u = _ats(edge, res, thr)
            print(f"{'':<7}{thr:>7}{f'{w}-{l}-{p}':>14}{pct:>8.1%}{u:>+9.1f}")
        print()
    print("break-even at -110 = 52.4%. Saved backtest_results.csv")


# ----------------------------------------------------------------------------
# LINES
# ----------------------------------------------------------------------------
def am_to_dec(o):
    return 1 + (o / 100 if o > 0 else 100 / -o)


def odds_key():
    """Odds API key: env var ODDS_API_KEY, else odds_api_key.txt in this folder (gitignored)."""
    k = os.environ.get("ODDS_API_KEY", "").strip()
    f = ROOT / "odds_api_key.txt"
    if not k and f.exists():
        k = f.read_text(encoding="utf-8").strip()
    return k or None


def fetch_odds():
    key = odds_key()
    if not key:
        return {}
    r = requests.get(ODDS_URL, params=dict(apiKey=key, regions="us", oddsFormat="american",
                                            markets="h2h,spreads,totals"), timeout=30)
    if r.status_code != 200:
        print(f"  ! odds api {r.status_code}: {r.text[:120]}")
        return {}
    out = {}
    for ev in r.json():
        h, a = ODDS_NAMES.get(ev["home_team"]), ODDS_NAMES.get(ev["away_team"])
        sp, tot, mlh, mla = [], [], [], []
        for bk in ev.get("bookmakers", []):
            for mk in bk.get("markets", []):
                for oc in mk.get("outcomes", []):
                    if mk["key"] == "spreads" and oc["name"] == ev["home_team"]:
                        sp.append(-oc["point"])            # -> nflverse sign (home fav +)
                    elif mk["key"] == "totals" and oc["name"] == "Over":
                        tot.append(oc["point"])
                    elif mk["key"] == "h2h":
                        (mlh if oc["name"] == ev["home_team"] else mla).append(oc["price"])
        out[(h, a)] = dict(
            spread_line=float(np.median(sp)) if sp else np.nan,
            total_line=float(np.median(tot)) if tot else np.nan,
            ml_home=float(np.median(mlh)) if mlh else np.nan,
            ml_away=float(np.median(mla)) if mla else np.nan)
    return out


# ----------------------------------------------------------------------------
# PREDICT / GRADE
# ----------------------------------------------------------------------------
def tier(edge, tiers):
    for thr, name in tiers:
        if abs(edge) >= thr:
            return name
    return ""


def fav_line(home, away, home_margin):
    """'PIT -2.5' from home's expected margin (nflverse sign)."""
    if pd.isna(home_margin):
        return "-"
    if abs(home_margin) < 0.05:
        return "PK"
    return f"{home} {-home_margin:.1f}" if home_margin > 0 else f"{away} {home_margin:.1f}"


def side_line(team, pts):
    return f"{team} PK" if abs(pts) < 0.05 else f"{team} {pts:+.1f}"


def predict(week, season=CURRENT_SEASON, refresh=True, refresh_lines=False):
    if refresh:
        print(f"refreshing {season} data...")
        load_games(season, True); load_plays(season, True); load_passing(season, True)
        load_player_week(season, True); load_schedule(True)
        load_injuries(True); load_depth(True)
    bundle = joblib.load(MODELS / "nfl_bundle.joblib")

    # schedule: ET kickoff times + projected starting QBs (manual QB_OVERRIDES win)
    sched = load_schedule()
    sw = sched[(sched.get("season") == season) & (sched.get("week") == week)].copy() if not sched.empty else sched
    starters = {}
    if not sw.empty:
        for c in ("home_team", "away_team"):
            sw[c] = sw[c].replace(TEAM_FIX)
        for _, s in sw.iterrows():
            for side in ("home", "away"):
                n = s.get(f"{side}_qb_name")
                if isinstance(n, str) and n.strip():
                    starters[s[f"{side}_team"]] = n.strip()
    inj = injury_report(week)
    inj_qbs, qb_notes = auto_qbs(week, inj)
    global QB_ACTIVE
    QB_ACTIVE = {**(starters if QB_FROM_SCHEDULE else {}), **inj_qbs, **QB_OVERRIDES}

    df = build_all()
    wk = df[(df.season == season) & (df.week == week)].copy()
    if wk.empty:
        sys.exit(f"no games found for {season} week {week}")

    wk["gametime"] = wk.gametime.astype(object)
    wk["stadium"] = wk.stadium.astype(object)
    sk = {}
    if not sw.empty:
        for _, s in sw.iterrows():
            if isinstance(s.get("gametime"), str):
                sk[(s.home_team, s.away_team)] = (s.gameday, s.gametime[:5], s.get("stadium"))
    missing = [(r.home_team, r.away_team) for _, r in wk.iterrows() if (r.home_team, r.away_team) not in sk]
    kicks = {**fetch_kickoffs(season, week), **sk} if missing else sk
    for i, r in wk.iterrows():
        k = kicks.get((r.home_team, r.away_team))
        if k:
            wk.at[i, "gameday"], wk.at[i, "gametime"] = pd.Timestamp(k[0]), k[1]
            if isinstance(k[2], str) and k[2]:
                wk.at[i, "stadium"] = k[2]
    print(f"  kickoff times (ET) for {sum((r.home_team, r.away_team) in kicks for _, r in wk.iterrows())}/{len(wk)} games")

    X = wk[bundle["features"]]
    wk["pred_margin"] = bundle["margin"].predict(X)
    wk["pred_total"] = bundle["total"].predict(X)
    wk["home_wp"] = norm.cdf(wk.pred_margin / bundle["sigma_margin"])
    wk["proj_home"] = (wk.pred_total + wk.pred_margin) / 2
    wk["proj_away"] = (wk.pred_total - wk.pred_margin) / 2

    wk["ml_home"], wk["ml_away"], wk["line_src"] = np.nan, np.nan, "nfldata"
    odds = fetch_odds()
    for i, r in wk.iterrows():
        o = odds.get((r.home_team, r.away_team))
        if o:
            for k, v in o.items():
                if not pd.isna(v):
                    wk.at[i, k] = v
            wk.at[i, "line_src"] = "odds_api"

    use = _usage(week) if inj else {}
    rows = []
    for _, r in wk.sort_values(["gameday", "gametime"]).iterrows():
        h, a = r.home_team, r.away_team
        d = r.to_dict()
        d["model_line"] = fav_line(h, a, r.pred_margin)
        d["market_line"] = fav_line(h, a, r.spread_line)
        # spread
        if pd.notna(r.spread_line):
            e = r.pred_margin - r.spread_line
            d["sp_edge"], d["sp_tier"] = abs(e), tier(e, SPREAD_TIERS)
            d["sp_side"] = h if e > 0 else a
            d["sp_pick"] = side_line(h, -r.spread_line) if e > 0 else side_line(a, r.spread_line)
        else:
            d.update(sp_edge=np.nan, sp_tier="", sp_side="", sp_pick="")
        # total
        if pd.notna(r.total_line):
            e = r.pred_total - r.total_line
            d["tot_edge"], d["tot_tier"] = abs(e), tier(e, TOTAL_TIERS)
            d["tot_pick"] = f"{'OVER' if e > 0 else 'UNDER'} {r.total_line:g}"
        else:
            d.update(tot_edge=np.nan, tot_tier="", tot_pick="")
        # moneyline value (needs Odds API)
        d.update(ml_pick="", ml_odds=np.nan, ml_edge=np.nan)
        if pd.notna(r.ml_home) and pd.notna(r.ml_away):
            ih, ia = 1 / am_to_dec(r.ml_home), 1 / am_to_dec(r.ml_away)
            fh = ih / (ih + ia)
            eh, ea = r.home_wp - fh, (1 - r.home_wp) - (1 - fh)
            if max(eh, ea) >= ML_EDGE:
                home_side = eh >= ea
                d.update(ml_pick=h if home_side else a,
                         ml_odds=r.ml_home if home_side else r.ml_away,
                         ml_edge=eh if home_side else ea)
        d["flags"] = "|".join(_flags(r) + _injury_flags(r, inj, qb_notes, use))
        rows.append(d)

    out = pd.DataFrame(rows)
    PICKS.mkdir(exist_ok=True); CARDS.mkdir(exist_ok=True)

    import intel as intel_mod
    intel = intel_mod.load(refresh_live=refresh)
    # defense intel shown on each side: a_dline = AWAY team's defense (what the home offense faces)
    out["a_dline"] = out.away_team.map(lambda t: intel_mod.defense_line(intel, t))
    out["h_dline"] = out.home_team.map(lambda t: intel_mod.defense_line(intel, t))
    keep = ["game_id", "season", "week", "gameday", "gametime", "stadium", "away_team", "home_team",
            "a_rec", "h_rec", "a_qb_name", "h_qb_name", "proj_away", "proj_home", "pred_margin",
            "pred_total", "home_wp", "spread_line", "total_line", "ml_home", "ml_away", "line_src",
            "model_line", "market_line", "sp_edge", "sp_tier", "sp_side", "sp_pick",
            "tot_edge", "tot_tier", "tot_pick", "ml_pick", "ml_odds", "ml_edge", "flags",
            "a_off_epa", "a_def_epa", "h_off_epa", "h_def_epa",
            "a_pass_mu", "a_rush_mu", "h_pass_mu", "h_rush_mu", "a_dline", "h_dline"]
    tag = f"{season}_wk{week:02d}"
    out[keep].to_csv(PICKS / f"{tag}.csv", index=False)

    pr = None
    if (MODELS / "props_bundle.joblib").exists():
        import props
        print("props...")
        try:
            qbs = {**dict(zip(wk.home_team, wk.h_qb_name)), **dict(zip(wk.away_team, wk.a_qb_name))}
            qbs = {t: n for t, n in qbs.items() if isinstance(n, str)}
            pr = props.predict(wk, season, week, qbs, refresh_lines=refresh_lines, injuries=inj, frame=df)
            pr = intel_mod.annotate_props(pr, intel)
            pr.to_csv(PICKS / f"props_{tag}.csv", index=False)
            n_pk = int((pr.tier.fillna("") != "").sum()) if len(pr) else 0
            print(f"  {len(pr)} player props, {n_pk} plays -> picks/props_{tag}.csv")
        except Exception as ex:  # props should never take down the game card run
            print(f"  ! props skipped: {type(ex).__name__}: {ex}")
            pr = None
    else:
        print("  (no props model yet: run `python nfl_algo.py train` to build it)")

    from cards import render_cards
    html_path = CARDS / f"{tag}.html"
    html_path.write_text(render_cards(out, season, week, pr), encoding="utf-8")
    try:
        import posts
        made = posts.make_posts(out, pr, season, week, ROOT / "posts")
        print(f"  post graphics -> {', '.join('posts/' + f.name for f in made)}")
    except Exception as ex:
        print(f"  ! post graphics skipped: {type(ex).__name__}: {ex}")

    print(f"\n{season} WEEK {week}  ({len(out)} games)\n")
    for _, r in out.iterrows():
        sp = f"{r.sp_pick} [{r.sp_tier} {r.sp_edge:.1f}]" if r.sp_tier else "-"
        tt = f"{r.tot_pick} [{r.tot_tier} {r.tot_edge:.1f}]" if r.tot_tier else "-"
        print(f"{r.away_team:>3} @ {r.home_team:<3}  model {r.model_line:<10} vegas {r.market_line:<10} "
              f"| {sp:<26} | {tt}")
    print(f"\npicks -> {PICKS / (tag + '.csv')}\ncards -> {html_path}")


def _usage(week):
    """Avg (targets + carries + attempts) per game this season, by player_id."""
    ps = load_player_week(CURRENT_SEASON)
    if ps.empty:
        return {}
    ps = ps[ps.week < week]
    t = ps.targets.fillna(0) + ps.carries.fillna(0) + ps.attempts.fillna(0)
    return t.groupby(ps.player_id).mean().to_dict()


def _injury_flags(r, inj, qb_notes, use):
    """OUT / Q chips for skill players who actually get usage this season."""
    if not inj:
        return []
    out = []
    for team in (r.away_team, r.home_team):
        if team in qb_notes and team not in QB_OVERRIDES:
            out.append(f"{team} QB? {qb_notes[team]}")
        hits = [(v, use.get(k, 0)) for k, v in inj.items()
                if v["team"] == team and v["pos"] in ("QB", "RB", "WR", "TE") and use.get(k, 0) >= 4
                and v["status"] in ("Out", "Doubtful", "Questionable")]
        for v, _ in sorted(hits, key=lambda x: -x[1])[:3]:
            tag = "OUT" if v["status"] == "Out" else v["status"][0]
            out.append(f"{tag}: {v['name']}")
    return out


def _flags(r):
    f = []
    for s, t in (("a", r.away_team), ("h", r.home_team)):
        if r[f"{s}_gp"] > 0 and r[f"{s}_rest"] <= 5:
            f.append(f"{t} SHORT WEEK")
        if r[f"{s}_rest"] >= 13:
            f.append(f"{t} OFF BYE")
        if r[f"{s}_qb_change"] == 1:
            f.append(f"{t} QB CHANGE")
    if r.is_intl:
        f.append("INTL")
    elif r.dome:
        f.append("DOME")
    if r.wind >= 15:
        f.append(f"WIND {int(r.wind)}")
    if r.temp <= 32 and not r.dome:
        f.append(f"{int(r.temp)}°F")
    if r["div"]:
        f.append("DIVISION")
    if r.tz_travel >= 3 and not r.is_intl:
        f.append(f"{r.away_team} CROSS-COUNTRY")
    return f


def _units(outcome, odds=-110):
    if outcome == "W":
        return am_to_dec(odds) - 1
    return -1.0 if outcome == "L" else 0.0


def grade(week, season=CURRENT_SEASON):
    f = PICKS / f"{season}_wk{week:02d}.csv"
    if not f.exists():
        sys.exit(f"no picks file {f}")
    picks = pd.read_csv(f)
    res = load_games(season, refresh=True)[["game_id", "home_score", "away_score"]]
    picks = picks.merge(res, on="game_id", how="left")
    rows = []
    for _, r in picks.dropna(subset=["home_score"]).iterrows():
        margin, total = r.home_score - r.away_score, r.home_score + r.away_score
        base = dict(game_id=r.game_id, season=season, week=week,
                    matchup=f"{r.away_team} @ {r.home_team}",
                    score=f"{int(r.away_score)}-{int(r.home_score)}")
        if isinstance(r.sp_tier, str) and r.sp_tier:
            x = (margin - r.spread_line) * (1 if r.sp_side == r.home_team else -1)
            o = "W" if x > 0 else "L" if x < 0 else "P"
            rows.append({**base, "market": "SPREAD", "tier": r.sp_tier, "pick": r.sp_pick,
                         "edge": round(r.sp_edge, 2), "result": o, "units": _units(o)})
        if isinstance(r.tot_tier, str) and r.tot_tier:
            x = (total - r.total_line) * (1 if r.tot_pick.startswith("OVER") else -1)
            o = "W" if x > 0 else "L" if x < 0 else "P"
            rows.append({**base, "market": "TOTAL", "tier": r.tot_tier, "pick": r.tot_pick,
                         "edge": round(r.tot_edge, 2), "result": o, "units": _units(o)})
        if isinstance(r.ml_pick, str) and r.ml_pick:
            won = (margin > 0) == (r.ml_pick == r.home_team)
            o = "P" if margin == 0 else "W" if won else "L"
            rows.append({**base, "market": "ML", "tier": "VALUE",
                         "pick": f"{r.ml_pick} {int(r.ml_odds):+d}", "edge": round(r.ml_edge, 3),
                         "result": o, "units": _units(o, r.ml_odds)})
    import props
    rows += props.grade(season, week)
    if not rows:
        sys.exit("nothing to grade yet (no finished games with plays)")
    new = pd.DataFrame(rows)
    tr = pd.concat([pd.read_csv(TRACKER), new]) if TRACKER.exists() else new
    tr = tr.drop_duplicates(["game_id", "market", "pick"], keep="last")
    tr.to_csv(TRACKER, index=False)
    print(new[["matchup", "score", "market", "tier", "pick", "result", "units"]].to_string(index=False))
    tracker_summary(tr)


def tracker_summary(tr=None):
    tr = pd.read_csv(TRACKER) if tr is None else tr
    print("\nSEASON TRACKER")
    for (mkt, tr_), g in tr.groupby(["market", "tier"]):
        w, l, p = (g.result == "W").sum(), (g.result == "L").sum(), (g.result == "P").sum()
        u = g.units.sum()
        print(f"  {mkt:<20}{tr_:<7} {w}-{l}-{p}  {w / max(w + l, 1):.1%}  {u:+.2f}u  ROI {u / max(w + l, 1):+.1%}")
    u = tr.units.sum()
    print(f"  {'ALL':<27} {u:+.2f}u over {len(tr)} plays")


# ----------------------------------------------------------------------------
# AUTO: one command for the whole week (what the scheduled task runs)
# ----------------------------------------------------------------------------
def current_week(season=CURRENT_SEASON):
    g = load_games(season, refresh=True)
    open_games = g[g.home_score.isna()]
    if open_games.empty:
        return None
    return int(open_games.week.min())


def auto(push=True):
    import subprocess
    from datetime import datetime
    print(f"\n===== AUTO RUN {datetime.now():%Y-%m-%d %H:%M} =====")
    week = current_week()
    if week is None:
        print("season's over, nothing to do")
        return
    print(f"current week: {week}")

    # 1) grade last week if it's done
    if week > 1 and (PICKS / f"{CURRENT_SEASON}_wk{week - 1:02d}.csv").exists():
        try:
            grade(week - 1)
        except SystemExit as ex:
            print(f"  grade: {ex}")

    # 2) retrain once a week
    bundle = MODELS / "nfl_bundle.joblib"
    if not bundle.exists() or time.time() - bundle.stat().st_mtime > 5 * 86400:
        print("retraining (weekly)...")
        load_games(CURRENT_SEASON, True); load_plays(CURRENT_SEASON, True); load_passing(CURRENT_SEASON, True)
        load_player_week(CURRENT_SEASON, True)
        train()

    # 3) picks, props, cards, posts
    predict(week, refresh=True)

    # 4) push to GitHub -> site updates
    if push:
        def git(*a):
            r = subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True)
            return r.returncode, (r.stdout + r.stderr).strip()
        git("add", "-A")
        code, msg = git("commit", "-m", f"auto: week {week} {datetime.now():%a %H:%M}")
        if "nothing to commit" in msg:
            print("git: no changes")
            return
        code, msg = git("push")
        print("git push: " + ("done, site will update in ~1 min" if code == 0 else f"FAILED\n{msg}"))


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="AlgoHub NFL game algo")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("update")
    sub.add_parser("train")
    sub.add_parser("backtest")
    sub.add_parser("record")
    pa = sub.add_parser("auto", help="grade last week, retrain weekly, predict this week, push")
    pa.add_argument("--no-push", action="store_true")
    for name in ("predict", "grade"):
        p = sub.add_parser(name)
        p.add_argument("--week", type=int, required=True)
        p.add_argument("--season", type=int, default=CURRENT_SEASON)
        if name == "predict":
            p.add_argument("--no-refresh", action="store_true")
            p.add_argument("--refresh-lines", action="store_true",
                           help="re-pull prop lines from the Odds API (costs credits)")
    a = ap.parse_args()
    if a.cmd == "update":
        update()
    elif a.cmd == "train":
        train()
    elif a.cmd == "backtest":
        backtest()
    elif a.cmd == "record":
        tracker_summary()
    elif a.cmd == "auto":
        auto(push=not a.no_push)
    elif a.cmd == "predict":
        predict(a.week, a.season, refresh=not a.no_refresh, refresh_lines=a.refresh_lines)
    elif a.cmd == "grade":
        grade(a.week, a.season)
