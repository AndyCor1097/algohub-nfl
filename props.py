"""
AlgoHub NFL player props.
Stats:  nflverse weekly player stats (GitHub release CSVs, no key)
Lines:  The Odds API player props (needs ODDS_API_KEY), cached per week so reruns don't burn credits
Model:  HistGradientBoosting per market, features = this season's prior games only
        (usage + efficiency) + team context from the game model + what the opponent has allowed.
Markets: pass yds, rush yds, rec yds, receptions, anytime TD
"""
import json
import os
import re

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_predict

import nfl_algo as nfl

PROPS_FIRST_SEASON = 2020
ODDS_BASE = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl"

MARKETS = {
    #  key          odds api market            stat column        label         positions
    "pass_yds":   ("player_pass_yds",       "passing_yards",   "Pass Yds",   {"QB"}),
    "rush_yds":   ("player_rush_yds",       "rushing_yards",   "Rush Yds",   {"QB", "RB", "WR"}),
    "rec_yds":    ("player_reception_yds",  "receiving_yards", "Rec Yds",    {"RB", "WR", "TE"}),
    "receptions": ("player_receptions",     "receptions",      "Receptions", {"RB", "WR", "TE"}),
    "atd":        ("player_anytime_td",     "any_td",          "Anytime TD", {"QB", "RB", "WR", "TE"}),
}
API_TO_MKT = {v[0]: k for k, v in MARKETS.items()}

# edge = model prob - no-vig market prob
# tiers = model prob minus the DK price's break-even prob (percentage points).
# (EV% is shown too, but it explodes on longshots, so it doesn't set tiers)
PROP_TIERS = [(0.07, "HAMMER"), (0.05, "PLAY"), (0.03, "LEAN")]
TD_MIN_PROB = 0.12     # (legacy) no TD picks under 12%
# Anytime TD, rebuilt Oct 2026 after a 6-27 start: on weeks 4-5 (581 lines) the TD model added NOTHING
# beyond the market (its weight came out ~0 when fit next to the market price) and every TD "edge"
# was the model overrating longshots (+500 and up: model 13%, market 8.5%, actual 7.4%).
# What DID show up is the classic favorite-longshot bias: short TD prices hit MORE than the market's
# fair %, longshots hit less. So TD probability now = the market's fair % run through a calibration
# curve fit on our own graded weeks (refit every run), and TD plays only in the +100 to +250 range.
TD_MIN_PRICE = 100     # no TD plays shorter than +100
TD_MAX_PRICE = 250     # or longer than +250
TD_CAL_DEFAULT = (0.0, 1.0)     # fair is already calibrated by the log-odds de-vig (TD_SHIFT)
# Lotto TDs: just for fun, kept OUT of the main record. Top LOTTO_N longshots each week where the
# model's TD % is highest relative to the market's. Longshots lose as a group (-11% on weeks 4-5),
# so these are small-stake sprinkles with their own tracker, not plays.
LOTTO_N = 5
LOTTO_MIN_PRICE, LOTTO_MAX_PRICE = 300, 800   # (intercept, slope) on log-odds, half-strength version of the weeks 4-5 fit
TD_HOLD = 1.20         # OLD flat de-vig (kept only to convert old saved files)
# Yes-only anytime-TD prices: strip the hold in log-odds, not as a flat %. Flat /1.20 took CMC -310 (75.6%)
# down to 63%. Weeks 4-6 (893 lines vs results): log-odds shift 0.18 fit best (logloss .4197 vs .4201 flat),
# and it's right on favorites: books' 70%+ -> hit 71%, shift says 72%, flat said 63%.
TD_SHIFT = 0.18


def td_yes_fair(p):
    p = min(max(float(p), 0.005), 0.995)
    return float(1 / (1 + np.exp(-(np.log(p / (1 - p)) - TD_SHIFT))))
POS_CODE = {"QB": 0, "RB": 1, "WR": 2, "TE": 3}

STAT_MAP_V1 = {  # short name -> nflverse column
    "att": "attempts", "cmp": "completions", "pyd": "passing_yards", "ptd": "passing_tds",
    "car": "carries", "ryd": "rushing_yards", "rtd": "rushing_tds",
    "tgt": "targets", "rec": "receptions", "recyd": "receiving_yards", "rectd": "receiving_tds",
    "tsh": "target_share", "ayd": "receiving_air_yards",
}
STAT_MAP = {**STAT_MAP_V1, "ays": "air_yards_share", "wopr": "wopr"}
PLAYER_FEATS = ["gp"] + [f"{s}_pg" for s in STAT_MAP_V1] + ["tdany_pg", "l1_tgt", "l1_car", "l1_att", "pos_code"]
TEAM_FEATS = ["implied", "total_line", "team_spread", "is_home", "dome", "wind", "team_off_pass",
              "team_off_rush", "team_pace", "team_qb_epa", "opp_def_pass", "opp_def_rush"]
OPP_FEATS = ["opp_pyd_allow", "opp_ryd_allow", "opp_recyd_allow_pos", "opp_rec_allow_pos", "opp_td_allow_pos"]
FEATURES_V1 = PLAYER_FEATS + TEAM_FEATS + OPP_FEATS + ["week"]
# Props v2: usage (snap share, air yards share, WOPR, red-zone share), team pass rate, coverage
V2_USAGE = ["ays_pg", "wopr_pg", "snap_pg", "l1_snap", "rz_tgt_pg", "rz_car_pg", "rz_tgt_sh", "rz_car_sh",
            "team_pass_rate"]
V2_COV = ["cov_ypt_man", "cov_ypt_zone", "cov_tg", "opp_man_prev", "cov_ratio"]
V2_EXTRA = V2_USAGE + V2_COV
# Holdout test (2021-25, each season unseen): coverage features improved rec yds MAE by only 0.03
# on top of usage, and need ~260MB of extra downloads -> off in the model, still shown on cards.
USE_COVERAGE_FEATURES = False
FEATURES_V2 = FEATURES_V1 + V2_USAGE + (V2_COV if USE_COVERAGE_FEATURES else [])
BINNED_CAL = {"pass_yds", "rush_yds", "rec_yds"}   # receptions calibrate better pooled (holdout test)
USE_V2 = True
FEATURES = FEATURES_V2 if USE_V2 else FEATURES_V1
_V2 = None   # v2 side tables (team pass rate, coverage), set by load_stats


# ----------------------------------------------------------------------------
# DATA
# ----------------------------------------------------------------------------
def load_stats(refresh_current=False):
    parts = []
    for s in range(PROPS_FIRST_SEASON, nfl.CURRENT_SEASON + 1):
        d = nfl.load_player_week(s, refresh_current and s == nfl.CURRENT_SEASON)
        if not d.empty:
            parts.append(d)
    st = pd.concat(parts, ignore_index=True)
    st = st[st.season_type.isin(["REG", "POST"])].copy()
    st["position"] = st.position.replace({"FB": "RB", "HB": "RB"})
    st = st[st.position.isin(POS_CODE)]
    for c in ("team", "opponent_team"):
        st[c] = st[c].replace(nfl.TEAM_FIX)
    for c in STAT_MAP.values():
        st[c] = pd.to_numeric(st.get(c), errors="coerce").fillna(0.0)
    st["any_td"] = ((st.rushing_tds + st.receiving_tds) > 0).astype(int)
    st["week"] = st.week.astype(int)
    st["name_key"] = st.player_display_name.map(name_key)
    if USE_V2:
        global _V2
        import features_v2 as fv2
        seasons = list(range(PROPS_FIRST_SEASON, nfl.CURRENT_SEASON + 1))
        raw = fv2.load(seasons, refresh_current, coverage=USE_COVERAGE_FEATURES)
        st = fv2.attach_player_week(st, raw)
        cov_p, cov_d, lg_man = fv2.coverage_features(raw, seasons)
        _V2 = dict(pass_rate=fv2.team_pass_rate(raw), cov_p=cov_p, cov_d=cov_d, lg_man=lg_man)
    return st


def name_key(s):
    s = str(s).lower().replace("’", "'")
    s = re.sub(r"[.'`,]", "", s)
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()


# ----------------------------------------------------------------------------
# FEATURES
# ----------------------------------------------------------------------------
def _player_cum(st):
    """Cumulative (inclusive) season totals per player after each game they played."""
    p = st.sort_values(["player_id", "season", "week"]).copy()
    p["one"] = 1
    p["tdany"] = p.rushing_tds + p.receiving_tds
    cols = ["one", "tdany"] + list(STAT_MAP.values())
    cum = p.groupby(["player_id", "season"])[cols].cumsum()
    out = p[["player_id", "season", "week", "position", "team"]].copy()
    out["gp"] = cum["one"]
    for short, col in STAT_MAP.items():
        out[f"{short}_pg"] = cum[col] / cum["one"]
    out["tdany_pg"] = cum["tdany"] / cum["one"]
    out["l1_tgt"], out["l1_car"], out["l1_att"] = p.targets, p.carries, p.attempts
    if "snap_pct" in p.columns:   # v2 usage
        g = p.groupby(["player_id", "season"])
        n_snap = g.snap_pct.transform(lambda x: x.notna().cumsum())
        s_snap = p.snap_pct.fillna(0).groupby([p.player_id, p.season]).cumsum()
        out["snap_pg"] = (s_snap / n_snap.replace(0, np.nan)).values
        out["l1_snap"] = p.snap_pct.values
        c2 = g[["rz_tgt", "rz_car", "team_rz_tgt", "team_rz_car"]].cumsum()
        out["rz_tgt_pg"] = c2.rz_tgt / cum["one"]
        out["rz_car_pg"] = c2.rz_car / cum["one"]
        out["rz_tgt_sh"] = c2.rz_tgt / c2.team_rz_tgt.replace(0, np.nan)
        out["rz_car_sh"] = c2.rz_car / c2.team_rz_car.replace(0, np.nan)
    out = out.rename(columns={"position": "cur_pos", "team": "last_team"})
    return out


def _defense_cum(st):
    """What each defense has allowed per game, cumulative inclusive, shrunk to league avg."""
    st = st.copy()
    st["tdany"] = st.rushing_tds + st.receiving_tds
    rows = []
    g = st.groupby(["season", "week", "opponent_team"])
    base = g.agg(pyd=("passing_yards", "sum"), ryd=("rushing_yards", "sum")).reset_index()
    for pos in ("RB", "WR", "TE", "QB"):
        x = st[st.position == pos].groupby(["season", "week", "opponent_team"]).agg(
            **{f"recyd_{pos}": ("receiving_yards", "sum"), f"rec_{pos}": ("receptions", "sum"),
               f"td_{pos}": ("tdany", "sum")}).reset_index()
        base = base.merge(x, on=["season", "week", "opponent_team"], how="left")
    base = base.fillna(0).rename(columns={"opponent_team": "def_team"}).sort_values(["def_team", "season", "week"])
    stat_cols = [c for c in base.columns if c not in ("season", "week", "def_team")]
    lg = base[stat_cols].mean()
    k = 3.0
    cum = base.groupby(["def_team", "season"])[stat_cols].cumsum()
    n = base.groupby(["def_team", "season"]).cumcount() + 1
    out = base[["def_team", "season", "week"]].copy()
    for c in stat_cols:
        out[c] = (cum[c] + k * lg[c]) / (n + k)
    return out, lg


def _team_context(frame):
    """Per (season, week, team) context pulled from the game model's frame."""
    keep = {}
    for s, o in (("h", "a"), ("a", "h")):
        d = pd.DataFrame({
            "season": frame.season, "week": frame.week, "game_id": frame.game_id,
            "team": frame.home_team if s == "h" else frame.away_team,
            "opp": frame.away_team if s == "h" else frame.home_team,
            "is_home": int(s == "h"),
            "total_line": frame.total_line,
            "team_spread": frame.spread_line if s == "h" else -frame.spread_line,
            "dome": frame.dome, "wind": frame.wind,
            "team_off_pass": frame[f"{s}_off_pass"], "team_off_rush": frame[f"{s}_off_rush"],
            "team_pace": frame[f"{s}_pace"], "team_qb_epa": frame[f"{s}_qb_epa"],
            "opp_def_pass": frame[f"{o}_def_pass"], "opp_def_rush": frame[f"{o}_def_rush"],
        })
        keep[s] = d
    ctx = pd.concat(keep.values(), ignore_index=True)
    ctx["implied"] = ctx.total_line / 2 + ctx.team_spread / 2
    return ctx


def _asof(left, right, by):
    """Attach the latest right-row strictly before left.week (same season)."""
    left = left.reset_index(drop=True).copy()
    left["_i"] = np.arange(len(left))
    l = left.sort_values("week")
    r = right.sort_values("week")
    m = pd.merge_asof(l, r, on="week", by=by, allow_exact_matches=False, suffixes=("", "_r"))
    return m.sort_values("_i").drop(columns="_i").reset_index(drop=True)


def build_rows(st, frame, rows):
    """rows: player_id, season, week, team, opp, position (+ targets for training)."""
    pc = _player_cum(st)
    dc, _ = _defense_cum(st)
    out = _asof(rows, pc, ["player_id", "season"])
    dc = dc.rename(columns={"def_team": "opp"})
    out = _asof(out, dc, ["opp", "season"])
    out["opp_pyd_allow"] = out.pyd
    out["opp_ryd_allow"] = out.ryd
    pos = out.position.where(out.position.isin(["RB", "WR", "TE"]), "RB")
    for name, pre in (("opp_recyd_allow_pos", "recyd_"), ("opp_rec_allow_pos", "rec_"), ("opp_td_allow_pos", "td_")):
        out[name] = np.select([pos == p for p in ("RB", "WR", "TE")],
                              [out[f"{pre}{p}"] for p in ("RB", "WR", "TE")], np.nan)
    ctx = _team_context(frame)
    out = out.merge(ctx.drop(columns="opp"), on=["season", "week", "team"], how="left")
    out["pos_code"] = out.position.map(POS_CODE)
    out["gp"] = out.gp.fillna(0)
    if _V2 is not None:
        pr = _V2["pass_rate"]
        if len(pr):
            out = _asof(out, pr, ["team", "season"])
        else:
            out["team_pass_rate"] = np.nan
        P, D, lg_man = _V2["cov_p"], _V2["cov_d"], _V2["lg_man"]
        out = out.merge(P, on=["player_id", "season"], how="left") if len(P) else out.assign(
            cov_ypt_man=np.nan, cov_ypt_zone=np.nan, cov_tg_man=np.nan, cov_tg_zone=np.nan)
        out = out.merge(D, on=["opp", "season"], how="left") if len(D) else out.assign(opp_man_prev=np.nan)
        out["cov_tg"] = out.cov_tg_man + out.cov_tg_zone
        m = out.opp_man_prev
        out["cov_ratio"] = (m * out.cov_ypt_man + (1 - m) * out.cov_ypt_zone) / (
            lg_man * out.cov_ypt_man + (1 - lg_man) * out.cov_ypt_zone)
    for c in V2_EXTRA:
        if c not in out.columns:
            out[c] = np.nan
    return out


def training_rows(st, frame):
    rows = st[["player_id", "player_display_name", "season", "week", "team", "opponent_team", "position",
               "passing_yards", "rushing_yards", "receiving_yards", "receptions", "any_td"]].rename(
        columns={"opponent_team": "opp"})
    df = build_rows(st, frame, rows)
    return df[(df.gp >= 1) & df.implied.notna()]


def eligible(df, mkt):
    pos = MARKETS[mkt][3]
    m = df.position.isin(pos)
    if mkt == "pass_yds":
        return m & (df.att_pg >= 15)
    if mkt == "rush_yds":
        return m & (df.car_pg >= 2)
    if mkt in ("rec_yds", "receptions"):
        return m & (df.tgt_pg >= 2)
    return m & ((df.tgt_pg + df.car_pg) >= 3)


# ----------------------------------------------------------------------------
# MODEL
# ----------------------------------------------------------------------------
def _reg():
    return HistGradientBoostingRegressor(learning_rate=0.04, max_iter=400, max_leaf_nodes=20,
                                         min_samples_leaf=60, l2_regularization=1.0, random_state=7)


def _clf():
    return HistGradientBoostingClassifier(learning_rate=0.04, max_iter=300, max_leaf_nodes=20,
                                          min_samples_leaf=80, l2_regularization=1.0, random_state=7)


def train(frame=None):
    frame = nfl.build_all() if frame is None else frame
    st = load_stats()
    df = training_rows(st, frame)
    bundle = {"features": FEATURES, "models": {}, "z": {}, "version": "v2" if USE_V2 else "v1"}
    kf = KFold(5, shuffle=True, random_state=7)
    print("props training")
    for mkt, (_, stat, label, _) in MARKETS.items():
        d = df[eligible(df, mkt)]
        X, y = d[FEATURES], d[stat]
        if mkt == "atd":
            m = _clf().fit(X, y)
            oof = cross_val_predict(_clf(), X, y, cv=kf, method="predict_proba")[:, 1]
            brier = np.mean((oof - y) ** 2)
            base = np.mean((y.mean() - y) ** 2)
            print(f"  {label:<11} n={len(d):>6,}  brier {brier:.4f} (base {base:.4f})")
        else:
            m = _reg().fit(X, y)
            oof = cross_val_predict(_reg(), X, y, cv=kf)
            # normalized residuals -> empirical distribution for over/under probs (handles skew)
            z = ((y - oof) / np.sqrt(np.clip(oof, 1, None))).to_numpy()
            bundle["z"][mkt] = np.sort(np.random.default_rng(7).choice(z, min(len(z), 20000), replace=False))
            # skew differs by role (backups have lots of zeros, starters don't): one distribution per size bin
            edges = np.quantile(oof, [0.2, 0.4, 0.6, 0.8])
            bins = np.digitize(oof, edges)
            bundle.setdefault("zbins", {})[mkt] = (edges, [np.sort(z[bins == b]) for b in range(5)])
            print(f"  {label:<11} n={len(d):>6,}  MAE {np.mean(np.abs(y - oof)):.2f}  (naive {np.mean(np.abs(y - y.mean())):.2f})")
        bundle["models"][mkt] = m
    joblib.dump(bundle, nfl.MODELS / "props_bundle.joblib")
    print("saved models/props_bundle.joblib")


def p_over(bundle, mkt, proj, line):
    z = bundle["z"][mkt]
    if mkt in BINNED_CAL and "zbins" in bundle and mkt in bundle["zbins"]:
        edges, zs = bundle["zbins"][mkt]
        z = zs[int(np.digitize([proj], edges)[0])]
    thr = (line - proj) / np.sqrt(max(proj, 1))
    return 1 - np.searchsorted(z, thr, side="right") / len(z)


# ----------------------------------------------------------------------------
# LINES (Odds API)
# ----------------------------------------------------------------------------
def am_to_prob(o):
    return 100 / (o + 100) if o > 0 else -o / (-o + 100)


PRICE_BOOK = "draftkings"   # the book you bet: its line + price is what gets graded
MIN_FAIR_BOOKS = 2          # other books needed (same line, both sides) for a market fair price
MAX_VS_MARKET = 0.15        # model prob this far from the market's fair prob -> capped at LEAN
YDS_TRUST = 0.25            # yardage / receptions: final % = fair + 25% of (model - fair)
YDS_MAX_GAP = 0.25          # model 25+ points off the market = it's missing news (role, injury): no play
MAX_PROP_PLAYS = 30         # per week, best edges, one per player (TD lotto separate)


def price_market(entry):
    """DraftKings line/prices + market fair prob from other books at the same line (both sides)."""
    if "books" not in entry:  # old cache format: consensus only
        return dict(line=entry["line"], over=entry["over"], under=entry.get("under"), fair_over=None,
                    n_fair=0, book="consensus")
    books = entry["books"]
    dk = books.get(PRICE_BOOK, {})
    if "over" not in dk:
        nv = books.get("novig", {})
        if "over" not in nv:
            return None
        line, over = nv["over"]
        under = nv["under"][1] if "under" in nv and nv["under"][0] == line else None
        return dict(line=float(line), over=float(over), under=float(under) if under is not None else None,
                    fair_over=nv.get("mid"), n_fair=1 if nv.get("mid") is not None else 0, book="NOVIG")
    line, over = dk["over"]
    under = dk["under"][1] if "under" in dk and dk["under"][0] == line else None
    fair = []
    for b, v in books.items():
        if b in (PRICE_BOOK, "novig") or "over" not in v or "under" not in v:
            continue
        if v["over"][0] == line and v["under"][0] == line:
            io, iu = am_to_prob(v["over"][1]), am_to_prob(v["under"][1])
            fair.append(io / (io + iu))
    nv = books.get("novig", {})
    if nv.get("mid") is not None and "over" in nv and nv["over"][0] == line:
        fair.append(float(nv["mid"]))           # exchange price: no vig to strip
    fair_over = float(np.mean(fair)) if len(fair) >= MIN_FAIR_BOOKS else None
    if fair_over is None and line == 0.5 and under is None:
        # yes-only markets (anytime TD): other books' Yes prices with typical hold removed
        yes = [am_to_prob(v["over"][1]) for b, v in books.items()
               if b not in (PRICE_BOOK, "novig") and "over" in v and v["over"][0] == line]
        if len(yes) >= MIN_FAIR_BOOKS:
            fair_over = td_yes_fair(float(np.mean(yes)))
            if nv.get("mid") is not None and "over" in nv and nv["over"][0] == line:
                fair_over = (fair_over + float(nv["mid"])) / 2      # exchange Yes/No mid: no vig at all
    return dict(line=float(line), over=float(over), under=float(under) if under is not None else None,
                fair_over=fair_over, n_fair=len(fair), book="DK")


def _parse_event(js):
    raw = {}
    for bk in js.get("bookmakers", []):
        for mk in bk.get("markets", []):
            mkt = API_TO_MKT.get(mk["key"])
            if not mkt:
                continue
            for oc in mk.get("outcomes", []):
                player = oc.get("description") or oc.get("name")
                side = oc.get("name", "").lower()
                side = "over" if side in ("over", "yes") else "under" if side in ("under", "no") else None
                if not side:
                    continue
                pt = oc.get("point", 0.5 if mkt == "atd" else None)
                if pt is None:
                    continue
                raw.setdefault("|".join((name_key(player), mkt)), {}).setdefault(bk["key"], {})[side] = (pt, oc["price"])
    return {k: {"books": b} for k, b in raw.items()}


REFETCH_WITHIN_H = 30   # games kicking off within this many hours get fresh lines...
REFETCH_AFTER_H = 12    # ...if the saved ones are older than this


def _fetch_dk_lines(season, week, games, refresh=False):
    """{(name_key, mkt): {"books": {book: {over: (pt, price), under: (pt, price)}}}}. Cached per week.

    Books post props a few games at a time during the week, so a Tuesday pull only has some games.
    Each run only spends credits on games that (a) have no props saved yet, or (b) kick off within
    30h and were last pulled 12h+ ago. refresh=True re-pulls everything."""
    from datetime import datetime, timezone
    cache = nfl.DATA / f"props_lines_{season}_wk{week:02d}.json"
    meta_f = nfl.DATA / f"props_lines_{season}_wk{week:02d}_events.json"
    lines = json.loads(cache.read_text()) if cache.exists() and not refresh else {}
    meta = json.loads(meta_f.read_text()) if meta_f.exists() and not refresh else {}
    to_tuple = lambda d: {tuple(k.split("|")): v for k, v in d.items()}

    key = nfl.odds_key()
    if not key:
        if not lines:
            print("  (no Odds API key found: props show projections only, no picks)")
        return lines
    try:
        r = requests.get(f"{ODDS_BASE}/events", params={"apiKey": key}, timeout=30)   # free call
    except Exception as ex:
        print(f"  ! odds api events: {ex}")
        return lines
    if r.status_code != 200:
        print(f"  ! odds api events {r.status_code}: {r.text[:120]}")
        return lines
    want = {(h, a) for h, a in zip(games.home_team, games.away_team)}
    events = [e for e in r.json()
              if (nfl.ODDS_NAMES.get(e["home_team"]), nfl.ODDS_NAMES.get(e["away_team"])) in want]

    now = datetime.now(timezone.utc)
    pulled, used = 0, None
    for ev in events:
        m = meta.get(ev["id"])
        kick = datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00"))
        hrs_to_kick = (kick - now).total_seconds() / 3600
        if hrs_to_kick < -4:
            continue                                        # already played
        if m and m.get("n", 0) > 0:
            age = (now - datetime.fromisoformat(m["fetched"])).total_seconds() / 3600
            if not (0 <= hrs_to_kick <= REFETCH_WITHIN_H and age >= REFETCH_AFTER_H):
                continue                                    # have it, still fresh enough
        rr = requests.get(f"{ODDS_BASE}/events/{ev['id']}/odds", timeout=30, params=dict(
            apiKey=key, regions="us", markets=",".join(v[0] for v in MARKETS.values()), oddsFormat="american"))
        used = rr.headers.get("x-requests-remaining", used)
        if rr.status_code == 401:
            print("  odds api: out of credits this month -> props priced off Novig instead")
            break
        if rr.status_code != 200:
            print(f"  ! props {ev['away_team']} @ {ev['home_team']}: {rr.status_code}")
            continue
        got = _parse_event(rr.json())
        pulled += 1
        for k in (m or {}).get("keys", []):                 # drop this game's old lines, keep the rest
            lines.pop(k, None)
        lines.update(got)
        meta[ev["id"]] = {"fetched": now.isoformat(), "n": len(got), "keys": list(got),
                          "game": f"{ev['away_team']} @ {ev['home_team']}"}
        print(f"  props {ev['away_team']} @ {ev['home_team']}: {len(got)} player-markets"
              + ("" if got else " (books haven't posted yet)"))

    nfl.DATA.mkdir(exist_ok=True)
    cache.write_text(json.dumps(lines))
    meta_f.write_text(json.dumps(meta))
    games_with = sum(1 for v in meta.values() if v.get("n", 0) > 0)
    print(f"  prop lines: {len(lines)} player-markets | {games_with}/{len(events)} games have props"
          + f" | pulled {pulled} games this run" + (f" | odds api credits left: {used}" if used else ""))
    return lines


NOVIG_CACHE_H = 3     # re-pull Novig props at most every 3 hours


def _fetch_novig_props(season, week, refresh=False, max_seconds=600):
    """Novig exchange player props (free): {'name_key|mkt': {'line','over','under','fair_over'}}."""
    import time as _t
    f = nfl.DATA / f"props_novig_{season}_wk{week:02d}.json"
    old = json.loads(f.read_text()) if f.exists() else {}
    # a thin pull (rate limited, lines not up yet) never counts as "fresh": try again next run
    if old and len(old) >= 60 and not refresh and _t.time() - f.stat().st_mtime < NOVIG_CACHE_H * 3600:
        return old
    try:
        import sys as _s
        _s.path.insert(0, str(nfl.ROOT.parent / "novig"))
        from novig_lines import player_props
        raw = player_props("NFL", max_seconds=max_seconds)   # soonest games go first
    except Exception as ex:
        print(f"  ! novig props: {type(ex).__name__}: {ex}")
        return json.loads(f.read_text()) if f.exists() else {}
    new = {f"{name_key(p)}|{m}": v for (p, m), v in raw.items()}
    out = {**old, **new}                       # fresh prices win; lines a blocked lookup missed keep the last good price
    nfl.DATA.mkdir(exist_ok=True)
    f.write_text(json.dumps(out))
    print(f"  novig props: {len(new)} fresh player-markets ({len(out)} total with earlier pulls)")
    return out


_PREFETCH = {}


def prefetch_novig(season, week, max_seconds=240):
    """Kick off the Novig props pull on a background thread (it's all network waiting).
    fetch_prop_lines picks up the result instead of pulling again."""
    import threading

    def run():
        try:
            _PREFETCH["res"] = _fetch_novig_props(season, week, False, max_seconds=max_seconds)
        except Exception as ex:
            print(f"  ! novig props (background): {type(ex).__name__}: {ex}", flush=True)
    t = threading.Thread(target=run, daemon=True)
    _PREFETCH.clear()
    _PREFETCH.update(t=t, key=(season, week))
    t.start()


def fetch_prop_lines(season, week, games, refresh=False):
    """DraftKings (Odds API) when there are credits, plus Novig's exchange props (free) for every player.
    The DK line is what gets priced when it exists; otherwise the Novig line/price is used."""
    lines = _fetch_dk_lines(season, week, games, refresh)
    have_dk = sum(1 for v in lines.values() if "draftkings" in v.get("books", {})) > 50
    # with DraftKings lines in hand Novig is just the sharp cross-check: 3 minutes is plenty
    t = _PREFETCH.get("t")
    if t is not None and _PREFETCH.get("key") == (season, week) and not refresh:
        t.join()                                  # started at the top of the run, usually done by now
        nv = _PREFETCH.get("res")
        if nv is None:
            nv = _fetch_novig_props(season, week, False, max_seconds=120)
    else:
        nv = _fetch_novig_props(season, week, refresh, max_seconds=180 if have_dk else 300)
    for k, v in nv.items():
        e = lines.setdefault(k, {"books": {}})
        e.setdefault("books", {})
        e["books"]["novig"] = {"over": (v["line"], v["over"]) if v.get("over") is not None else None,
                               "under": (v["line"], v["under"]) if v.get("under") is not None else None,
                               "mid": v.get("fair_over")}
        for side in ("over", "under"):
            if e["books"]["novig"][side] is None:
                e["books"]["novig"].pop(side)
    n_dk = sum(1 for v in lines.values() if "draftkings" in v.get("books", {}))
    print(f"  prop lines: {n_dk} with DraftKings, {len(nv)} from Novig")
    return {tuple(k.split("|")): v for k, v in lines.items()}


# ----------------------------------------------------------------------------
# PREDICT
# ----------------------------------------------------------------------------
def candidates(st, wk, starters):
    """Players who've played for each team this season, attached to that team's upcoming game."""
    cur = st[st.season == nfl.CURRENT_SEASON].sort_values("week")
    last = cur.groupby("player_id").tail(1)[["player_id", "player_display_name", "team", "position", "name_key"]]
    rows = []
    for _, g in wk.iterrows():
        for team, opp in ((g.home_team, g.away_team), (g.away_team, g.home_team)):
            p = last[last.team == team].copy()
            p["opp"], p["season"], p["week"] = opp, g.season, g.week
            rows.append(p)
    c = pd.concat(rows, ignore_index=True)
    # passing props only for the projected starter
    is_qb = c.position == "QB"
    sk = c.team.map({t: name_key(n) for t, n in starters.items()})
    c["is_starter"] = ~is_qb | sk.isna() | (c.name_key == sk)
    return c


def weather_mult(w, mkt):
    """Multiplier on a projection from kickoff weather. Measured on 2021-25 outdoor games
    (team stats vs that team's season average):
        wind 10-15: pass yds 0.98 | 15-20: pass 0.90, rec 0.95, TDs 0.83 | 20+: pass 0.81, rec 0.82
        under 45F:  pass yds 0.93, receptions 0.95.  Rushing barely moves."""
    if not w or w.get("dome"):
        return 1.0
    over = max(0.0, min(w["wind"], 26.0) - 10.0)
    cold = w["temp"] < 45
    if mkt in ("pass_yds", "rec_yds"):
        m = (1 - 0.011 * over) * (0.95 if cold else 1.0)
    elif mkt == "receptions":
        m = (1 - 0.006 * over) * (0.97 if cold else 1.0)
    elif mkt == "atd":
        m = (1 - 0.012 * over) * (0.97 if cold else 1.0)
    else:
        m = 1.0
    return max(m, 0.78)


def td_calibration(season, week, st):
    """Fit outcome ~ market fair % (log-odds) on every past anytime-TD line we priced (all players, not just
    picks). Shrunk toward 'market is exactly right' (0, 1) until there's a lot of data."""
    from sklearn.linear_model import LogisticRegression
    rows = []
    for f in sorted(nfl.PICKS.glob(f"props_{season}_wk*.csv")):
        try:
            wk_ = int(f.stem.split("wk")[-1])
        except ValueError:
            continue
        if wk_ >= week:
            continue
        d = pd.read_csv(f)
        d = d[(d.market == "atd") & d.fair.notna()].copy()
        if "fair_v" not in d or d.fair_v.isna().all():      # saved before the log-odds de-vig: convert
            d["fair"] = d.fair.map(lambda f: td_yes_fair(min(f * TD_HOLD, 0.99)))
        d = d[["player_id", "fair"]].assign(week=wk_)
        rows.append(d)
    if not rows:
        return TD_CAL_DEFAULT
    d = pd.concat(rows).merge(st[st.season == season][["player_id", "week", "any_td"]], on=["player_id", "week"])
    if len(d) < 150 or d.any_td.nunique() < 2:
        return TD_CAL_DEFAULT
    z = np.log(d.fair.clip(.01, .99) / (1 - d.fair.clip(.01, .99))).values.reshape(-1, 1)
    m = LogisticRegression(C=1e6).fit(z, d.any_td)
    a, b = float(m.intercept_[0]), float(m.coef_[0][0])
    w = len(d) / (len(d) + 600)                 # 600 lines of data = trust the fit half way
    a, b = a * w, 1 + (b - 1) * w
    b = min(max(b, 0.8), 1.4)
    print(f"  TD calibration from {len(d)} past lines: intercept {a:+.2f}, slope {b:.2f}")
    return a, b


def td_prob(fair, cal):
    a, b = cal
    f = min(max(fair, .01), .99)
    return float(1 / (1 + np.exp(-(a + b * np.log(f / (1 - f))))))


# ----------------------------------------------------------------------------
# INJURY BOT: a ruled-out player's work goes to his healthy teammates
# ----------------------------------------------------------------------------
REALLOC = True
RUSH_COLS = ["car_pg", "ryd_pg", "rtd_pg", "rz_car_pg", "rz_car_sh"]
REC_COLS = ["tgt_pg", "rec_pg", "recyd_pg", "rectd_pg", "ayd_pg", "tsh_pg", "ays_pg", "wopr_pg", "rz_tgt_pg", "rz_tgt_sh"]


def reallocate(df, gone, st):
    """Carries -> healthy RBs, targets -> healthy WR/TE/RB (same position weighted 2x), in proportion to what each
    already gets. Only the part not already in teammates' averages moves: f = share of the team's games the missing
    player actually played (games he already missed are baked into everyone's numbers).
    Backtest (TD model, 2024-26): players handed 4+ touches scored 34.3%; without this we said 26.9%, with it 34.0%."""
    df = df.copy()
    df["inj_boost"] = 0.0
    if gone is None or gone.empty:
        return df
    for c in RUSH_COLS + REC_COLS + ["l1_car", "l1_tgt", "tdany_pg"]:
        if c in df:
            df[c] = df[c].astype(float)
    cur = st[st.season == nfl.CURRENT_SEASON]
    team_gp = cur.groupby("team").week.nunique().to_dict()
    for g in gone.itertuples():
        gp = getattr(g, "gp", 0) or 0
        if not gp or g.team not in team_gp or pd.isna(gp):
            continue
        f = min(gp / max(team_gp[g.team], 1), 1.0)
        mates = df.team == g.team
        rb = mates & (df.position == "RB")
        rec = mates & df.position.isin(["RB", "WR", "TE"])
        car = getattr(g, "car_pg", 0) or 0
        tgt = getattr(g, "tgt_pg", 0) or 0
        if rb.any() and car > 0:
            w = df.loc[rb, "car_pg"].fillna(0) + 0.3
            w = w / w.sum()
            for c in RUSH_COLS:
                v = getattr(g, c, np.nan)
                if c in df and pd.notna(v):
                    df.loc[rb, c] = df.loc[rb, c].fillna(0) + f * v * w
            if pd.notna(getattr(g, "l1_car", np.nan)):
                df.loc[rb, "l1_car"] = df.loc[rb, "l1_car"].fillna(0) + g.l1_car * w
            df.loc[rb, "tdany_pg"] = df.loc[rb, "tdany_pg"].fillna(0) + f * (getattr(g, "rtd_pg", 0) or 0) * w
            df.loc[rb, "inj_boost"] += f * car * w
        if rec.any() and tgt > 0:
            w = (df.loc[rec, "tgt_pg"].fillna(0) + 0.3) * np.where(df.loc[rec, "position"] == g.position, 2.0, 1.0)
            w = w / w.sum()
            for c in REC_COLS:
                v = getattr(g, c, np.nan)
                if c in df and pd.notna(v):
                    df.loc[rec, c] = df.loc[rec, c].fillna(0) + f * v * w
            if pd.notna(getattr(g, "l1_tgt", np.nan)):
                df.loc[rec, "l1_tgt"] = df.loc[rec, "l1_tgt"].fillna(0) + g.l1_tgt * w
            df.loc[rec, "tdany_pg"] = df.loc[rec, "tdany_pg"].fillna(0) + f * (getattr(g, "rectd_pg", 0) or 0) * w
            df.loc[rec, "inj_boost"] += f * tgt * w
    for c in ("rz_car_sh", "rz_tgt_sh", "tsh_pg", "ays_pg"):
        if c in df:
            df[c] = df[c].clip(upper=1.0)
    boosted = df[df.inj_boost >= 1.5].sort_values("inj_boost", ascending=False)
    if len(boosted):
        print("  injury bot: " + ", ".join(f"{r.player_display_name} +{r.inj_boost:.1f}" for r in boosted.head(8).itertuples()))
    return df


def predict(wk, season, week, starters, refresh_lines=False, injuries=None, frame=None):
    bundle = joblib.load(nfl.MODELS / "props_bundle.joblib")
    frame = nfl.LAST_FRAME if frame is None else frame
    st = load_stats()
    cand = candidates(st, wk, starters)
    injuries = injuries or {}
    status = cand.player_id.map(lambda k: injuries.get(k, {}).get("status", ""))
    dropped = cand[status.isin(["Out", "Doubtful"])]
    if len(dropped):
        print(f"  props: dropped {len(dropped)} Out/Doubtful players")
    cand = cand[~status.isin(["Out", "Doubtful"])].copy()
    cand["inj"] = status[cand.index].fillna("")
    df = build_rows(st, frame, cand)
    if REALLOC and len(dropped):
        try:
            df = reallocate(df, build_rows(st, frame, dropped), st)
        except Exception as ex:
            print(f"  ! injury bot skipped ({type(ex).__name__}: {ex})")
    lines = fetch_prop_lines(season, week, wk, refresh_lines)
    hist = st[st.season == season]

    # kickoff weather per game (filled by nfl_weather.apply in nfl_algo.predict)
    wx = {}
    if "game_id" in wk.columns:
        for _, g in wk.iterrows():
            wx[g.game_id] = dict(wind=float(g.get("wind", 8) or 0), temp=float(g.get("temp", 60) or 60),
                                 dome=int(g.get("dome", 0) or 0))

    try:
        td_cal = td_calibration(season, week, st)
    except Exception as ex:
        print(f"  ! TD calibration skipped ({type(ex).__name__}), using default")
        td_cal = TD_CAL_DEFAULT

    out = []
    for mkt, (_, stat, label, _) in MARKETS.items():
        has_line = df.name_key.map(lambda k: (k, mkt) in lines)
        base = eligible(df, mkt)
        if mkt == "pass_yds":
            base &= df.is_starter
        keep = base | (has_line & df.position.isin(MARKETS[mkt][3]))
        if mkt == "pass_yds":
            keep &= df.is_starter
        d = df[keep & (df.gp >= 1)].copy()
        if d.empty:
            continue
        # only bet players inside the population the model was trained on
        d["in_model"] = eligible(d, mkt)
        X = d[bundle["features"]]
        m = bundle["models"][mkt]
        d["proj"] = m.predict_proba(X)[:, 1] if mkt == "atd" else m.predict(X)
        d["wx_mult"] = d.game_id.map(lambda g: weather_mult(wx.get(g), mkt))
        d["proj"] = d.proj * d.wx_mult
        for _, r in d.iterrows():
            raw_ln = lines.get((r.name_key, mkt))
            ln = price_market(raw_ln) if raw_ln else None
            rec = dict(game_id=r.game_id, team=r.team, opp=r.opp, player=r.player_display_name, inj=r.inj,
                       player_id=r.player_id, position=r.position, market=mkt, label=label,
                       inj_boost=float(getattr(r, "inj_boost", 0) or 0),
                       proj=float(r.proj), wx_mult=float(r.wx_mult), line=np.nan, over_price=np.nan, under_price=np.nan,
                       p_over=np.nan, side="", pick="", price=np.nan, edge=np.nan, tier="", hits="",
                       book="", fair=np.nan, vs_mkt="", ev=np.nan)
            if ln:
                rec.update(line=ln["line"], over_price=ln["over"],
                           under_price=ln["under"] if ln["under"] is not None else np.nan, book=ln["book"])
                if mkt == "atd":
                    # market-first: no fair price from other books = no TD play
                    po = td_prob(ln["fair_over"], td_cal) * float(r.wx_mult) if ln["fair_over"] is not None else None
                    rec["td_pct"] = po if po is not None else np.nan     # site 'TD odds' board
                else:
                    # market-first (same fix as TDs / NBA props): sharp fair % + YDS_TRUST of the model's disagreement.
                    # Before this, raw model % vs DK's price gave 200 "plays" a week with fake 20-36% edges.
                    fair = ln["fair_over"]
                    if fair is None or (isinstance(fair, float) and np.isnan(fair)):
                        po = None
                    else:
                        pm = p_over(bundle, mkt, r.proj, ln["line"])
                        rec["p_model"] = pm
                        po = fair + YDS_TRUST * (pm - fair)
                        if abs(pm - fair) > YDS_MAX_GAP:
                            rec.update(line=ln["line"], fair=fair, vs_mkt="model way off market: news?")
                            out.append(rec)
                            continue
                if po is None:
                    rec.update(line=ln["line"], vs_mkt="no market")
                    out.append(rec)
                    continue
                # edge = model prob minus break-even prob of the DraftKings price
                e_o = po - am_to_prob(ln["over"])
                e_u = (1 - po) - am_to_prob(ln["under"]) if ln["under"] is not None else -1
                if mkt == "atd":
                    e_u = -1  # only bet the Yes side on TDs
                side = "over" if e_o >= e_u else "under"
                edge = max(e_o, e_u)
                price = ln["over"] if side == "over" else ln["under"]
                rec["ev"] = (po if side == "over" else 1 - po) * nfl.am_to_dec(price) - 1
                tier_ = next((t for thr, t in PROP_TIERS if edge >= thr), "")
                if not r.in_model:
                    tier_, rec["vs_mkt"] = "", "low usage"
                elif mkt == "atd" and not (TD_MIN_PRICE <= ln["over"] <= TD_MAX_PRICE):
                    tier_, rec["vs_mkt"] = "", "price range"
                if mkt == "atd":
                    rec["fair"] = ln["fair_over"]
                    rec["fair_v"] = 2
                    if tier_ == "HAMMER":           # new approach: no max-confidence TD plays yet
                        tier_ = "PLAY"
                elif ln["fair_over"] is not None:
                    fair_side = ln["fair_over"] if side == "over" else 1 - ln["fair_over"]
                    model_side = po if side == "over" else 1 - po
                    rec["fair"] = fair_side
                    # model fighting the whole market by a mile = usually stale info (injury, role), not edge
                    too_far = (model_side > 1.6 * fair_side + 0.03) if mkt == "atd" else \
                        (model_side - fair_side > MAX_VS_MARKET)
                    if too_far and tier_ in ("HAMMER", "PLAY"):
                        tier_, rec["vs_mkt"] = "LEAN", "capped"
                elif tier_ == "HAMMER":
                    # no market check available (thin / DK off-market line): don't go max confidence
                    tier_, rec["vs_mkt"] = "PLAY", "thin"
                rec.update(p_over=po, edge=edge, side=side, tier=tier_, price=price)
                if mkt == "atd":
                    rec["pick"] = f"{r.player_display_name} TD"
                else:
                    rec["pick"] = f"{'O' if side == 'over' else 'U'} {ln['line']:g}"
                ph = hist[hist.player_id == r.player_id][stat]
                if len(ph):
                    hit = (ph > ln["line"]).sum() if side == "over" else (ph < ln["line"]).sum()
                    rec["hits"] = f"{hit}/{len(ph)}"
            out.append(rec)
    props = pd.DataFrame(out)
    if props.empty:
        return props
    # games with real lines: only show players that have a line. Games without: top usage guys
    nol = props.line.isna()
    game_has_lines = props.line.notna().groupby(props.game_id).transform("any").astype(bool)
    rank = props[nol].groupby(["team", "market"]).proj.rank(ascending=False)
    cap = props.market.map({"pass_yds": 1, "rush_yds": 2, "rec_yds": 3, "receptions": 3, "atd": 3})
    props = props[~nol | (~game_has_lines & (rank.reindex(props.index) <= cap))]
    # ---- cap: one play per player, best MAX_PROP_PLAYS edges (a 200-play week isn't a card, it's noise)
    pl = props[props.tier.fillna("") != ""].sort_values("edge", ascending=False)
    keep_ix = pl.drop_duplicates("player_id").head(MAX_PROP_PLAYS).index
    drop_ix = pl.index.difference(keep_ix)
    props.loc[drop_ix, "tier"] = ""
    props.loc[drop_ix, "vs_mkt"] = "cut (weekly cap)"
    # ---- Lotto TDs
    props["lotto"] = 0
    props["lotto_ratio"] = np.nan
    td = props[(props.market == "atd") & props.line.notna() & props.fair.notna() & props.over_price.between(LOTTO_MIN_PRICE, LOTTO_MAX_PRICE)
               & (props.proj >= 0.10)]
    if len(td):
        ratio = td.proj / td.fair
        props.loc[td.index, "lotto_ratio"] = ratio.round(2)
        best = ratio[ratio > 1.0].sort_values(ascending=False).head(LOTTO_N).index
        props.loc[best, "lotto"] = 1
        props.loc[best, "price"] = props.loc[best, "over_price"]
        names = [f"{props.at[i, 'player']} {int(props.at[i, 'over_price']):+d}" for i in best]
        print("  lotto TDs: " + (", ".join(names) or "none"))
    props.insert(0, "season", season)
    props.insert(1, "week", week)
    return props.reset_index(drop=True)


def grade(season, week):
    f = nfl.PICKS / f"props_{season}_wk{week:02d}.csv"
    if not f.exists():
        return []
    props = pd.read_csv(f)
    lotto = props.get("lotto", pd.Series(0, index=props.index)).fillna(0).astype(int) == 1
    props = props[(props.tier.fillna("") != "") | lotto].copy()
    is_l = props.get("lotto", pd.Series(0, index=props.index)).fillna(0).astype(int) == 1
    props.loc[is_l & (props.tier.fillna("") == ""), "tier"] = "LOTTO"
    st = load_stats(refresh_current=True)
    act = st[(st.season == season) & (st.week == week)].set_index("player_id")
    rows = []
    for _, r in props.iterrows():
        stat = MARKETS[r.market][1]
        base = dict(game_id=r.game_id, season=season, week=week, matchup=f"{r.player} ({r.team})",
                    market="LOTTO TD" if r.tier == "LOTTO" else f"PROP {r.label.upper()}", tier=r.tier, pick=f"{r.player} {r.pick}" if r.market != "atd" else r.pick,
                    edge=round(r.edge, 3))
        if r.player_id not in act.index:
            o, score = "P", "DNP"  # books void props for players who don't play
        else:
            v = act.loc[r.player_id, stat]
            v = v.iloc[0] if isinstance(v, pd.Series) else v
            score = f"{v:g}"
            if r.market == "atd":
                o = "W" if v > 0 else "L"
            else:
                x = (v - r.line) * (1 if r.side == "over" else -1)
                o = "W" if x > 0 else "L" if x < 0 else "P"
        rows.append({**base, "score": score, "result": o, "units": nfl._units(o, r.price)})
    return rows
