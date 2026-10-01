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
PROP_TIERS = [(0.12, "HAMMER"), (0.08, "PLAY"), (0.05, "LEAN")]   # EV at the DraftKings price
POS_CODE = {"QB": 0, "RB": 1, "WR": 2, "TE": 3}

STAT_MAP = {  # short name -> nflverse column
    "att": "attempts", "cmp": "completions", "pyd": "passing_yards", "ptd": "passing_tds",
    "car": "carries", "ryd": "rushing_yards", "rtd": "rushing_tds",
    "tgt": "targets", "rec": "receptions", "recyd": "receiving_yards", "rectd": "receiving_tds",
    "tsh": "target_share", "ayd": "receiving_air_yards",
}
PLAYER_FEATS = ["gp"] + [f"{s}_pg" for s in STAT_MAP] + ["tdany_pg", "l1_tgt", "l1_car", "l1_att", "pos_code"]
TEAM_FEATS = ["implied", "total_line", "team_spread", "is_home", "dome", "wind", "team_off_pass",
              "team_off_rush", "team_pace", "team_qb_epa", "opp_def_pass", "opp_def_rush"]
OPP_FEATS = ["opp_pyd_allow", "opp_ryd_allow", "opp_recyd_allow_pos", "opp_rec_allow_pos", "opp_td_allow_pos"]
FEATURES = PLAYER_FEATS + TEAM_FEATS + OPP_FEATS + ["week"]


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
    bundle = {"features": FEATURES, "models": {}, "z": {}}
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
            print(f"  {label:<11} n={len(d):>6,}  MAE {np.mean(np.abs(y - oof)):.2f}  (naive {np.mean(np.abs(y - y.mean())):.2f})")
        bundle["models"][mkt] = m
    joblib.dump(bundle, nfl.MODELS / "props_bundle.joblib")
    print("saved models/props_bundle.joblib")


def p_over(bundle, mkt, proj, line):
    z = bundle["z"][mkt]
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


def price_market(entry):
    """DraftKings line/prices + market fair prob from other books at the same line (both sides)."""
    if "books" not in entry:  # old cache format: consensus only
        return dict(line=entry["line"], over=entry["over"], under=entry.get("under"), fair_over=None,
                    n_fair=0, book="consensus")
    books = entry["books"]
    dk = books.get(PRICE_BOOK, {})
    if "over" not in dk:
        return None
    line, over = dk["over"]
    under = dk["under"][1] if "under" in dk and dk["under"][0] == line else None
    fair = []
    for b, v in books.items():
        if b == PRICE_BOOK or "over" not in v or "under" not in v:
            continue
        if v["over"][0] == line and v["under"][0] == line:
            io, iu = am_to_prob(v["over"][1]), am_to_prob(v["under"][1])
            fair.append(io / (io + iu))
    fair_over = float(np.mean(fair)) if len(fair) >= MIN_FAIR_BOOKS else None
    return dict(line=float(line), over=float(over), under=float(under) if under is not None else None,
                fair_over=fair_over, n_fair=len(fair), book="DK")


def fetch_prop_lines(season, week, games, refresh=False):
    """{(name_key, mkt): {"books": {book: {over: (pt, price), under: (pt, price)}}}}. Cached per week."""
    cache = nfl.DATA / f"props_lines_{season}_wk{week:02d}.json"
    if cache.exists() and not refresh:
        return {tuple(k.split("|")): v for k, v in json.loads(cache.read_text()).items()}
    key = nfl.odds_key()
    if not key:
        print("  (no Odds API key found: props show projections only, no picks)")
        return {}
    r = requests.get(f"{ODDS_BASE}/events", params={"apiKey": key}, timeout=30)
    if r.status_code != 200:
        print(f"  ! odds api events {r.status_code}: {r.text[:120]}")
        return {}
    want = {(h, a) for h, a in zip(games.home_team, games.away_team)}
    events = [e for e in r.json()
              if (nfl.ODDS_NAMES.get(e["home_team"]), nfl.ODDS_NAMES.get(e["away_team"])) in want]
    markets = ",".join(v[0] for v in MARKETS.values())
    raw, used = {}, None
    for ev in events:
        rr = requests.get(f"{ODDS_BASE}/events/{ev['id']}/odds", timeout=30, params=dict(
            apiKey=key, regions="us", markets=markets, oddsFormat="american"))
        used = rr.headers.get("x-requests-remaining", used)
        if rr.status_code != 200:
            print(f"  ! props {ev['away_team']} @ {ev['home_team']}: {rr.status_code}")
            continue
        for bk in rr.json().get("bookmakers", []):
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
                    raw.setdefault((name_key(player), mkt), {}).setdefault(bk["key"], {})[side] = (pt, oc["price"])
    lines = {k: {"books": books} for k, books in raw.items()}
    nfl.DATA.mkdir(exist_ok=True)
    cache.write_text(json.dumps({"|".join(k): v for k, v in lines.items()}))
    print(f"  prop lines: {len(lines)} player-markets from {len(events)} games"
          + (f" | odds api credits left: {used}" if used else ""))
    return lines


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
    lines = fetch_prop_lines(season, week, wk, refresh_lines)
    hist = st[st.season == season]

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
        X = d[bundle["features"]]
        m = bundle["models"][mkt]
        d["proj"] = m.predict_proba(X)[:, 1] if mkt == "atd" else m.predict(X)
        for _, r in d.iterrows():
            raw_ln = lines.get((r.name_key, mkt))
            ln = price_market(raw_ln) if raw_ln else None
            rec = dict(game_id=r.game_id, team=r.team, opp=r.opp, player=r.player_display_name, inj=r.inj,
                       player_id=r.player_id, position=r.position, market=mkt, label=label,
                       proj=float(r.proj), line=np.nan, over_price=np.nan, under_price=np.nan,
                       p_over=np.nan, side="", pick="", price=np.nan, edge=np.nan, tier="", hits="",
                       book="", fair=np.nan, vs_mkt="")
            if ln:
                rec.update(line=ln["line"], over_price=ln["over"],
                           under_price=ln["under"] if ln["under"] is not None else np.nan, book=ln["book"])
                po = float(r.proj) if mkt == "atd" else p_over(bundle, mkt, r.proj, ln["line"])
                # edge = expected value of the bet at the DraftKings price
                ev_o = po * nfl.am_to_dec(ln["over"]) - 1
                ev_u = (1 - po) * nfl.am_to_dec(ln["under"]) - 1 if ln["under"] is not None else -1
                if mkt == "atd":
                    ev_u = -1  # only bet the Yes side on TDs
                side = "over" if ev_o >= ev_u else "under"
                edge = max(ev_o, ev_u)
                tier_ = next((t for thr, t in PROP_TIERS if edge >= thr), "")
                if ln["fair_over"] is not None:
                    fair_side = ln["fair_over"] if side == "over" else 1 - ln["fair_over"]
                    model_side = po if side == "over" else 1 - po
                    rec["fair"] = fair_side
                    # model fighting the whole market by a mile = usually stale info (injury, role), not edge
                    if model_side - fair_side > MAX_VS_MARKET and tier_ in ("HAMMER", "PLAY"):
                        tier_, rec["vs_mkt"] = "LEAN", "capped"
                elif mkt != "atd" and tier_ == "HAMMER":
                    # no market check available (thin / DK off-market line): don't go max confidence
                    tier_, rec["vs_mkt"] = "PLAY", "thin"
                rec.update(p_over=po, edge=edge, side=side, tier=tier_,
                           price=ln["over"] if side == "over" else ln["under"])
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
    props.insert(0, "season", season)
    props.insert(1, "week", week)
    return props.reset_index(drop=True)


def grade(season, week):
    f = nfl.PICKS / f"props_{season}_wk{week:02d}.csv"
    if not f.exists():
        return []
    props = pd.read_csv(f)
    props = props[props.tier.fillna("") != ""]
    st = load_stats(refresh_current=True)
    act = st[(st.season == season) & (st.week == week)].set_index("player_id")
    rows = []
    for _, r in props.iterrows():
        stat = MARKETS[r.market][1]
        base = dict(game_id=r.game_id, season=season, week=week, matchup=f"{r.player} ({r.team})",
                    market=f"PROP {r.label.upper()}", tier=r.tier, pick=f"{r.player} {r.pick}" if r.market != "atd" else r.pick,
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
