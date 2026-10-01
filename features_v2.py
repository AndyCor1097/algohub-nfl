"""
Props v2 features (all free nflverse data):
  snap share         - snap_counts (weekly)
  red-zone share     - targets / carries inside the 20, from play-by-play
  team pass rate     - dropbacks / plays, from play-by-play
  air yards share, WOPR - from weekly player stats (handled in props.STAT_MAP)
  coverage           - receiver yds/target vs MAN and vs ZONE from PRIOR seasons,
                       x opponent's man rate from the PRIOR season (no peeking at the future)

Each season's raw pull is reduced to small tables and cached in data/, so the big downloads
happen once. The current season is refreshed on demand.
"""
import io
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import nfl_algo as nfl

COV_FIRST = 2018          # first season with man/zone tags
K_TGT = 25                # coverage yds/target shrinkage (targets)
LOCAL = os.environ.get("NFLVERSE_LOCAL")   # optional folder of already-downloaded nflverse files


def _src(path, usecols=None):
    """nflverse release file -> DataFrame (from LOCAL folder if present, else download)."""
    name = Path(path).name
    if LOCAL and (Path(LOCAL) / name).exists():
        f = Path(LOCAL) / name
        return pd.read_csv(f, usecols=(lambda c: c in usecols) if usecols else None, low_memory=False)
    try:
        r = requests.get(f"{nfl.NFLVERSE}/{path}", timeout=400)
        if r.status_code != 200:
            return pd.DataFrame()
        return pd.read_csv(io.BytesIO(r.content), usecols=(lambda c: c in usecols) if usecols else None,
                           low_memory=False, compression="gzip" if path.endswith(".gz") else None)
    except (requests.RequestException, ValueError, pd.errors.ParserError) as ex:
        print(f"  ! v2 data {path}: {ex}")
        return pd.DataFrame()


def _cache(name, build, refresh=False):
    f = nfl.DATA / f"{name}.csv.gz"
    if f.exists() and not refresh:
        return pd.read_csv(f, low_memory=False)
    df = build()
    if not df.empty:
        nfl.DATA.mkdir(exist_ok=True)
        df.to_csv(f, index=False)
    return df


# ----------------------------------------------------------------------------
# per-season reduced tables
# ----------------------------------------------------------------------------
_PLAYERS = None


def _pfr_to_gsis():
    global _PLAYERS
    if _PLAYERS is None:
        p = _cache("v2_players", lambda: _src("players/players.csv", ["gsis_id", "pfr_id"]))
        _PLAYERS = dict(zip(p.pfr_id, p.gsis_id)) if not p.empty else {}
    return _PLAYERS


def snaps(season, refresh=False):
    def build():
        s = _src(f"snap_counts/snap_counts_{season}.csv",
                 ["season", "week", "game_type", "pfr_player_id", "team", "offense_pct", "offense_snaps"])
        if s.empty:
            return s
        s["player_id"] = s.pfr_player_id.map(_pfr_to_gsis())
        s = s[s.player_id.notna() & (s.offense_snaps.fillna(0) > 0)]
        return s[["player_id", "season", "week", "offense_pct"]].rename(columns={"offense_pct": "snap_pct"})
    return _cache(f"v2_snaps_{season}", build, refresh)


PBP_COLS = ["game_id", "play_id", "season", "week", "posteam", "defteam", "yardline_100", "qb_dropback",
            "pass_attempt", "rush_attempt", "receiver_player_id", "rusher_player_id", "yards_gained", "epa"]


def _pbp(season):
    return _src(f"pbp/play_by_play_{season}.csv.gz", PBP_COLS)


def redzone(season, pbp=None, refresh=False):
    """player-week rz targets/carries + team-week rz totals, dropbacks, plays."""
    def build():
        pb = _pbp(season) if pbp is None else pbp
        if pb.empty:
            return pb
        pb = pb[pb.posteam.notna()].copy()
        pb["posteam"] = pb.posteam.replace(nfl.TEAM_FIX)
        rz = pb[pb.yardline_100 <= 20]
        tg = rz[rz.receiver_player_id.notna()].groupby(["season", "week", "posteam", "receiver_player_id"]).size()
        ca = rz[(rz.rush_attempt == 1) & rz.rusher_player_id.notna()].groupby(
            ["season", "week", "posteam", "rusher_player_id"]).size()
        pl = pd.concat([tg.rename("rz_tgt"), ca.rename("rz_car")], axis=1).fillna(0).reset_index()
        pl.columns = ["season", "week", "team", "player_id", "rz_tgt", "rz_car"]
        pl["kind"] = "player"
        plays = pb[(pb.qb_dropback == 1) | (pb.rush_attempt == 1)]
        team = plays.groupby(["season", "week", "posteam"]).agg(
            dropbacks=("qb_dropback", "sum"), plays=("play_id", "size")).join(
            rz[rz.receiver_player_id.notna()].groupby(["season", "week", "posteam"]).size().rename("team_rz_tgt")).join(
            rz[rz.rush_attempt == 1].groupby(["season", "week", "posteam"]).size().rename("team_rz_car")).fillna(0).reset_index()
        team = team.rename(columns={"posteam": "team"})
        team["kind"] = "team"
        return pd.concat([pl, team], ignore_index=True)
    return _cache(f"v2_rz_{season}", build, refresh)


def coverage_season(season, pbp=None, refresh=False):
    """targets with man/zone tag (receiver), and defense man rate, for one season."""
    def build():
        pa = _src(f"pbp_participation/pbp_participation_{season}.csv",
                  ["nflverse_game_id", "play_id", "defense_man_zone_type"])
        pb = _pbp(season) if pbp is None else pbp
        if pa.empty or pb.empty or "defense_man_zone_type" not in pa.columns:
            return pd.DataFrame()
        m = pb[pb.qb_dropback == 1].merge(pa, left_on=["game_id", "play_id"],
                                          right_on=["nflverse_game_id", "play_id"])
        m = m[m.defense_man_zone_type.isin(["MAN_COVERAGE", "ZONE_COVERAGE"])]
        m["man"] = (m.defense_man_zone_type == "MAN_COVERAGE").astype(int)
        m["defteam"] = m.defteam.replace(nfl.TEAM_FIX)
        t = m[m.receiver_player_id.notna()].copy()
        t["yds"] = t.yards_gained.fillna(0)
        rec = t.groupby(["receiver_player_id", "man"]).agg(n=("yds", "size"), yds=("yds", "sum")).reset_index()
        rec = rec.rename(columns={"receiver_player_id": "id"})
        rec["kind"] = "recv"
        d = m.groupby("defteam").agg(n=("man", "size"), man=("man", "sum")).reset_index().rename(
            columns={"defteam": "id"})
        d["kind"] = "def"
        out = pd.concat([rec, d], ignore_index=True)
        out["season"] = season
        return out
    return _cache(f"v2_cov_{season}", build, refresh)


# ----------------------------------------------------------------------------
# assemble
# ----------------------------------------------------------------------------
def load(seasons, refresh_current=False, coverage=True):
    """Returns dict of player-week, team-week, coverage tables across seasons."""
    sn, rz, cov = [], [], []
    import time
    for s in seasons:
        refresh = refresh_current and s == nfl.CURRENT_SEASON
        if s == nfl.CURRENT_SEASON and not refresh:   # current season: refresh if older than 6h
            f = nfl.DATA / f"v2_rz_{s}.csv.gz"
            refresh = not f.exists() or time.time() - f.stat().st_mtime > 6 * 3600
        print(f"  v2 features {s}...", flush=True)
        pbp = None
        need_pbp = refresh or not (nfl.DATA / f"v2_rz_{s}.csv.gz").exists() or \
            (coverage and s < nfl.CURRENT_SEASON and not (nfl.DATA / f"v2_cov_{s}.csv.gz").exists())
        if need_pbp:
            pbp = _pbp(s)
        sn.append(snaps(s, refresh))
        rz.append(redzone(s, pbp, refresh))
    for s in (range(COV_FIRST, nfl.CURRENT_SEASON) if coverage else []):   # finished seasons only
        c = coverage_season(s)
        if not c.empty:
            cov.append(c)
    sn = pd.concat([x for x in sn if not x.empty], ignore_index=True) if any(not x.empty for x in sn) else pd.DataFrame()
    rz = pd.concat([x for x in rz if not x.empty], ignore_index=True) if any(not x.empty for x in rz) else pd.DataFrame()
    cov = pd.concat(cov, ignore_index=True) if cov else pd.DataFrame()
    return dict(snaps=sn, rz=rz, cov=cov)


def attach_player_week(st, v2):
    """Add snap_pct, rz_tgt, rz_car, team_rz_tgt, team_rz_car to the weekly player stats."""
    st = st.copy()
    if not v2["snaps"].empty:
        sn = v2["snaps"].drop_duplicates(["player_id", "season", "week"])
        st = st.merge(sn, on=["player_id", "season", "week"], how="left")
    else:
        st["snap_pct"] = np.nan
    rz = v2["rz"]
    if not rz.empty:
        pl = rz[rz.kind == "player"][["player_id", "season", "week", "rz_tgt", "rz_car"]]
        tm = rz[rz.kind == "team"][["team", "season", "week", "team_rz_tgt", "team_rz_car"]]
        st = st.merge(pl.drop_duplicates(["player_id", "season", "week"]), on=["player_id", "season", "week"], how="left")
        st = st.merge(tm.drop_duplicates(["team", "season", "week"]), on=["team", "season", "week"], how="left")
        for c in ("rz_tgt", "rz_car", "team_rz_tgt", "team_rz_car"):
            st[c] = st[c].fillna(0)
    else:
        for c in ("rz_tgt", "rz_car", "team_rz_tgt", "team_rz_car"):
            st[c] = 0.0
    st["snap_pct"] = pd.to_numeric(st.snap_pct, errors="coerce")
    return st


def team_pass_rate(v2):
    """Cumulative (inclusive) dropback rate per team-season, shrunk to league avg; for asof lookup."""
    rz = v2["rz"]
    if rz.empty:
        return pd.DataFrame(columns=["team", "season", "week", "team_pass_rate"])
    t = rz[rz.kind == "team"].sort_values(["team", "season", "week"]).copy()
    lg = t.dropbacks.sum() / max(t.plays.sum(), 1)
    k = 120.0  # plays
    g = t.groupby(["team", "season"])
    t["team_pass_rate"] = (g.dropbacks.cumsum() + k * lg) / (g.plays.cumsum() + k)
    return t[["team", "season", "week", "team_pass_rate"]]


def coverage_features(v2, seasons):
    """(player_id, season) -> prior-seasons yds/target vs man & zone; (team, season) -> prior-season man rate."""
    cov = v2["cov"]
    if cov.empty:
        return pd.DataFrame(), pd.DataFrame(), np.nan
    rec, d = cov[cov.kind == "recv"], cov[cov.kind == "def"]
    lg_ypt = rec.groupby("man").yds.sum() / rec.groupby("man").n.sum()
    lg_man = d.man.sum() / d.n.sum()
    prow, drow = [], []
    for s in seasons:
        prev = rec[(rec.season < s) & (rec.season >= s - 2)]
        if not prev.empty:
            g = prev.groupby(["id", "man"])[["n", "yds"]].sum().unstack("man").fillna(0)
            for flag, tag in ((1, "man"), (0, "zone")):
                n = g[("n", flag)] if ("n", flag) in g else 0
                y = g[("yds", flag)] if ("yds", flag) in g else 0
                g[f"ypt_{tag}"] = (y + K_TGT * lg_ypt.get(flag, 7.0)) / (n + K_TGT)
                g[f"tg_{tag}"] = n
            x = g[["ypt_man", "ypt_zone", "tg_man", "tg_zone"]].copy()
            x.columns = ["cov_ypt_man", "cov_ypt_zone", "cov_tg_man", "cov_tg_zone"]
            x = x.reset_index().rename(columns={"id": "player_id"})
            x["season"] = s
            prow.append(x)
        pd_ = d[d.season == s - 1]
        if not pd_.empty:
            drow.append(pd.DataFrame({"opp": pd_.id, "season": s, "opp_man_prev": pd_.man / pd_.n}))
    P = pd.concat(prow, ignore_index=True) if prow else pd.DataFrame()
    D = pd.concat(drow, ignore_index=True) if drow else pd.DataFrame()
    if not P.empty:
        P = P.drop_duplicates(["player_id", "season"])
    return P, D, lg_man
