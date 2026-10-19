"""
restore_pregame.py: puts back the PRE-KICKOFF lines and picks for games that a late rerun overwrote.

Every auto run pushes picks/ to GitHub, so git has a copy of each run. For every game in the week,
this takes the version saved by the last run BEFORE that game kicked off, and writes it back into
picks/<season>_wkNN.csv and picks/props_<season>_wkNN.csv. Games that haven't started are left alone.

  python restore_pregame.py          (current season, week 6)
  python restore_pregame.py 7        (another week)
"""
import io
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
PICKS = ROOT / "picks"
SEASON = 2026


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")


def versions(rel):
    """[(commit time as UTC Timestamp, hash)] newest first."""
    r = git("log", "--format=%H|%cI", "--", rel)
    out = []
    for ln in r.stdout.splitlines():
        h, t = ln.split("|", 1)
        out.append((pd.Timestamp(t).tz_convert("UTC"), h))
    return out


def at(h, rel):
    r = git("show", f"{h}:{rel}")
    return pd.read_csv(io.StringIO(r.stdout)) if r.returncode == 0 and r.stdout.strip() else None


def kickoff_utc(gameday, gametime):
    ko = pd.Timestamp(f"{str(gameday)[:10]} {str(gametime)[:5]}").tz_localize("America/New_York")
    return ko.tz_convert("UTC")


def main(week):
    tag = f"{SEASON}_wk{week:02d}"
    cur_path = PICKS / f"{tag}.csv"
    cur = pd.read_csv(cur_path)
    kick = {r.game_id: kickoff_utc(r.gameday, r.gametime) for r in cur.itertuples()}
    now = pd.Timestamp.now(tz="UTC")
    started = {g: k for g, k in kick.items() if k <= now}
    if not started:
        print("no games have started yet: nothing to restore")
        return
    for rel, label in ((f"picks/{tag}.csv", "game picks"), (f"picks/props_{tag}.csv", "props")):
        path = ROOT / rel
        if not path.exists():
            continue
        vs = versions(rel)
        now_df = pd.read_csv(path)
        fixed, n = [], 0
        cache = {}
        for g, ko in started.items():
            pre = next((h for t, h in vs if t < ko), None)          # last saved run before kickoff
            if pre is None:
                continue
            if pre not in cache:
                cache[pre] = at(pre, rel)
            old = cache[pre]
            if old is None or "game_id" not in old:
                continue
            rows = old[old.game_id == g]
            if len(rows):
                fixed.append(rows)
                n += 1
        if not fixed:
            print(f"  {label}: no earlier saved versions found")
            continue
        keep = now_df[~now_df.game_id.isin({f.game_id.iloc[0] for f in fixed})]
        out = pd.concat(fixed + [keep], ignore_index=True)
        out = out[[c for c in now_df.columns if c in out.columns]]
        out.to_csv(path, index=False)
        print(f"  {label}: restored pre-kickoff rows for {n} game(s) -> {path.name}")
    print("done. Push it with: git add picks && git commit -m \"restore pre-kickoff picks\" && git push")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
