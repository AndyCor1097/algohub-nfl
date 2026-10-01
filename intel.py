"""
Matchup Intel: who wins vs man/zone, and what each defense actually does.

Receivers (2024-2025 seasons, nflverse participation + pbp):
    yards/target and EPA/target vs MAN and vs ZONE, shrunk toward league average
Defenses:
    man coverage rate, pressure rate        -> 2025 (2026 coverage isn't published yet)
    blitz rate, avg pass rushers, box count -> 2026 live (FTN charting)
Matchup:
    each WR/TE/RB's expected yards/target vs THIS defense's man rate,
    compared to his yards/target vs an average defense = coverage edge %

Shown on cards + props. Not a model input yet: there's only 1-2 seasons of coverage
history, so it can't be validated. It's there to confirm (COV ✓) or warn (COV ✗).
"""
import io

import numpy as np
import pandas as pd
import requests

import nfl_algo as nfl

COV_SEASONS = (2024, 2025)
K_TGT = 25                # targets of shrinkage toward league avg
COV_CONFIRM = 0.03        # |coverage edge| that counts as agreeing / disagreeing with a pick


def _get_csv(path, usecols=None):
    try:
        r = requests.get(f"{nfl.NFLVERSE}/{path}", timeout=300)
        if r.status_code != 200:
            print(f"  ! intel: {path} -> {r.status_code}")
            return pd.DataFrame()
        cols = (lambda c: c in usecols) if usecols else None
        return pd.read_csv(io.BytesIO(r.content), usecols=cols, low_memory=False,
                           compression="gzip" if path.endswith(".gz") else None)
    except (requests.RequestException, ValueError, pd.errors.ParserError) as ex:
        print(f"  ! intel: {path}: {ex}")
        return pd.DataFrame()


PBP_COLS = ["game_id", "play_id", "posteam", "defteam", "qb_dropback", "play_type",
            "receiver_player_id", "yards_gained", "epa", "complete_pass"]
PART_COLS = ["nflverse_game_id", "play_id", "defense_man_zone_type", "defense_coverage_type",
             "was_pressure", "number_of_pass_rushers"]


def _season_plays(season):
    print(f"  intel: coverage {season} (one-time download, ~70MB)...", flush=True)
    pb = _get_csv(f"pbp/play_by_play_{season}.csv.gz", PBP_COLS)
    pa = _get_csv(f"pbp_participation/pbp_participation_{season}.csv", PART_COLS)
    if pb.empty or pa.empty:
        return pd.DataFrame()
    m = pb[pb.qb_dropback == 1].merge(pa, left_on=["game_id", "play_id"],
                                      right_on=["nflverse_game_id", "play_id"], how="left")
    m["season"] = season
    for c in ("posteam", "defteam"):
        m[c] = m[c].replace(nfl.TEAM_FIX)
    return m


def build_static(refresh=False):
    """Receiver man/zone splits + 2025 defense coverage. Downloaded once, then cached small."""
    fr, fd = nfl.DATA / "intel_receivers.csv.gz", nfl.DATA / "intel_def_cov.csv.gz"
    if fr.exists() and fd.exists() and not refresh:
        return pd.read_csv(fr), pd.read_csv(fd)
    plays = pd.concat([_season_plays(s) for s in COV_SEASONS], ignore_index=True)
    if plays.empty:
        return pd.DataFrame(), pd.DataFrame()
    cov = plays[plays.defense_man_zone_type.isin(["MAN_COVERAGE", "ZONE_COVERAGE"])].copy()
    cov["man"] = (cov.defense_man_zone_type == "MAN_COVERAGE").astype(int)

    # receivers
    t = cov[cov.receiver_player_id.notna()].copy()
    t["yds"] = t.yards_gained.fillna(0)
    lg = t.groupby("man").agg(ypt=("yds", "mean"), epa=("epa", "mean"))
    g = t.groupby(["receiver_player_id", "man"]).agg(n=("yds", "size"), yds=("yds", "sum"),
                                                     epa_s=("epa", "sum")).reset_index()
    rows = []
    for pid, x in g.groupby("receiver_player_id"):
        rec = {"player_id": pid}
        for flag, tag in ((1, "man"), (0, "zone")):
            y = x[x.man == flag]
            n = float(y.n.sum()) if len(y) else 0.0
            rec[f"tg_{tag}"] = n
            rec[f"ypt_{tag}"] = ((y.yds.sum() if len(y) else 0) + K_TGT * lg.loc[flag, "ypt"]) / (n + K_TGT)
            rec[f"epa_{tag}"] = ((y.epa_s.sum() if len(y) else 0) + K_TGT * lg.loc[flag, "epa"]) / (n + K_TGT)
        rows.append(rec)
    recv = pd.DataFrame(rows)
    recv = recv[(recv.tg_man + recv.tg_zone) >= 20]

    # defenses, most recent season with coverage data
    last = cov[cov.season == max(COV_SEASONS)]
    d = plays[plays.season == max(COV_SEASONS)]
    dcov = last.groupby("defteam").agg(man_rate=("man", "mean")).join(
        d.groupby("defteam").agg(press_rate=("was_pressure", "mean"))).reset_index().rename(
        columns={"defteam": "team"})
    top = (last[last.defense_coverage_type.notna()].groupby(["defteam", "defense_coverage_type"]).size()
           .rename("n").reset_index().sort_values("n").groupby("defteam").tail(1))
    dcov = dcov.merge(top.rename(columns={"defteam": "team", "defense_coverage_type": "top_shell"})
                      [["team", "top_shell"]], on="team", how="left")
    dcov["cov_season"] = max(COV_SEASONS)
    dcov.attrs["lg_man"] = float(cov[cov.season == max(COV_SEASONS)].man.mean())
    nfl.DATA.mkdir(exist_ok=True)
    recv.to_csv(fr, index=False)
    dcov.to_csv(fd, index=False)
    print(f"  intel: {len(recv)} receivers, {len(dcov)} defenses")
    return recv, dcov


def build_live(refresh=False):
    """2026 defensive tendencies from FTN charting joined to 2026 pbp."""
    f = nfl.DATA / f"intel_def_{nfl.CURRENT_SEASON}.csv.gz"
    if f.exists() and not refresh:
        return pd.read_csv(f)
    s = nfl.CURRENT_SEASON
    ftn = _get_csv(f"ftn_charting/ftn_charting_{s}.csv",
                   ["nflverse_game_id", "nflverse_play_id", "n_blitzers", "n_pass_rushers", "n_defense_box",
                    "is_play_action", "is_motion"])
    pb = _get_csv(f"pbp/play_by_play_{s}.csv.gz", ["game_id", "play_id", "defteam", "qb_dropback", "play_type"])
    if ftn.empty or pb.empty:
        return pd.DataFrame()
    m = ftn.merge(pb, left_on=["nflverse_game_id", "nflverse_play_id"], right_on=["game_id", "play_id"])
    m["defteam"] = m.defteam.replace(nfl.TEAM_FIX)
    db = m[m.qb_dropback == 1]
    runs = m[m.play_type == "run"]
    out = db.groupby("defteam").agg(blitz_rate=("n_blitzers", lambda x: (x.fillna(0) > 0).mean()),
                                    rushers=("n_pass_rushers", "mean")).join(
        runs.groupby("defteam").agg(box=("n_defense_box", "mean"))).reset_index().rename(columns={"defteam": "team"})
    out.to_csv(f, index=False)
    return out


def load(refresh_live=False):
    try:
        recv, dcov = build_static()
        live = build_live(refresh_live)
    except Exception as ex:  # intel is a bonus, never block picks
        print(f"  ! intel skipped: {type(ex).__name__}: {ex}")
        return None
    if dcov.empty:
        return None
    defs = dcov.merge(live, on="team", how="outer") if not live.empty else dcov
    lg_man = float(dcov.man_rate.mean())
    return dict(recv=recv.set_index("player_id") if not recv.empty else recv,
                defs=defs.set_index("team"), lg_man=lg_man)


def defense_line(intel, team):
    """'MAN 43% · C1 · PRESS 33% · BLITZ 22%' for a defense."""
    if not intel or team not in intel["defs"].index:
        return ""
    d = intel["defs"].loc[team]
    bits = []
    if pd.notna(d.get("man_rate")):
        bits.append(f"MAN {d.man_rate:.0%}")
    if isinstance(d.get("top_shell"), str):
        bits.append(d.top_shell.replace("COVER_", "C").replace("2_MAN", "2-MAN"))
    if pd.notna(d.get("press_rate")):
        bits.append(f"PRESS {d.press_rate:.0%}")
    if pd.notna(d.get("blitz_rate")):
        bits.append(f"BLITZ {d.blitz_rate:.0%}")
    return " · ".join(bits)


def coverage_edge(intel, player_id, opp):
    """Receiver's expected yds/target vs this defense's man rate, relative to vs an average D."""
    if not intel or isinstance(intel["recv"], pd.DataFrame) and intel["recv"].empty:
        return None
    if player_id not in intel["recv"].index or opp not in intel["defs"].index:
        return None
    r = intel["recv"].loc[player_id]
    man = intel["defs"].loc[opp].get("man_rate")
    if pd.isna(man):
        return None
    lg = intel["lg_man"]
    exp = man * r.ypt_man + (1 - man) * r.ypt_zone
    base = lg * r.ypt_man + (1 - lg) * r.ypt_zone
    return dict(edge=exp / base - 1, ypt_man=r.ypt_man, ypt_zone=r.ypt_zone,
                tg=int(r.tg_man + r.tg_zone), opp_man=man)


def annotate_props(props, intel):
    """Add coverage edge + COV ✓/✗ to receiving props."""
    if props is None or props.empty or not intel:
        return props
    props = props.copy()
    for c in ("cov_edge", "ypt_man", "ypt_zone", "opp_man"):
        props[c] = np.nan
    props["cov_mark"] = ""
    for i, r in props.iterrows():
        if r.market not in ("rec_yds", "receptions"):
            continue
        ce = coverage_edge(intel, r.player_id, r.opp)
        if not ce:
            continue
        props.loc[i, ["cov_edge", "ypt_man", "ypt_zone", "opp_man"]] = [ce["edge"], ce["ypt_man"],
                                                                      ce["ypt_zone"], ce["opp_man"]]
        if isinstance(r.side, str) and r.side and abs(ce["edge"]) >= COV_CONFIRM:
            agrees = (ce["edge"] > 0) == (r.side == "over")
            props.loc[i, "cov_mark"] = "✓" if agrees else "✗"
    return props
