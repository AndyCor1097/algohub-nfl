"""Post graphics: ready-to-share images of the week's plays (1080x1350, IG/X portrait)."""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from cards import COLORS

W, H = 1080, 1350
BG, CARD, LINE, TX, MU = "#0b0e13", "#12161d", "#222a35", "#e6edf3", "#7d8590"
TIER = {"HAMMER": "#f2b134", "PLAY": "#3fb950", "LEAN": "#58a6ff", "VALUE": "#3fb950"}
RANK = {"HAMMER": 3, "PLAY": 2, "LEAN": 1, "VALUE": 2}
HANDLE = "@TheAlgoHub"

FONT_BOLD = ["C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/segoeuib.ttf",
             "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "DejaVuSans-Bold.ttf"]
FONT_COND = ["C:/Windows/Fonts/impact.ttf", "C:/Windows/Fonts/arialnb.ttf",
             "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf"] + FONT_BOLD
FONT_REG = ["C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "DejaVuSans.ttf"]


def _font(paths, size):
    for p in paths:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def _canvas(title, sub, season, week):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.text((64, 64), f"THE ALGOHUB  ·  NFL  ·  {season}", font=_font(FONT_BOLD, 26), fill=MU)
    d.text((60, 100), f"WEEK {week}", font=_font(FONT_COND, 120), fill=TX)
    d.text((64, 250), title, font=_font(FONT_BOLD, 40), fill=TIER["HAMMER"])
    if sub:
        d.text((64, 302), sub, font=_font(FONT_REG, 24), fill=MU)
    d.line((64, 350, W - 64, 350), fill=LINE, width=2)
    d.text((64, H - 70), HANDLE, font=_font(FONT_BOLD, 28), fill=TX)
    note = "model picks · not betting advice"
    f = _font(FONT_REG, 22)
    d.text((W - 64 - d.textlength(note, font=f), H - 64), note, font=f, fill=MU)
    return im, d


def _badge(d, x, y, team, size=56):
    bg, fg = COLORS.get(team, ("#30363d", "#e6edf3"))
    d.rounded_rectangle((x, y, x + size, y + size), radius=12, fill=bg)
    f = _font(FONT_BOLD, 20 if len(team) > 2 else 22)
    tw = d.textlength(team, font=f)
    d.text((x + (size - tw) / 2, y + size / 2 - 12), team, font=f, fill=fg)


def _chip(d, x, y, tier):
    f = _font(FONT_BOLD, 18)
    tw = d.textlength(tier, font=f)
    d.rounded_rectangle((x, y, x + tw + 24, y + 32), radius=7, fill=TIER.get(tier, MU))
    d.text((x + 12, y + 5), tier, font=f, fill="#0b0e13")
    return tw + 24


def _rows(d, items, top=380, max_rows=9):
    """items: dict(team, tier, pick, sub, right)"""
    if not items:
        d.text((64, top + 20), "No plays clear the bar this week.", font=_font(FONT_REG, 32), fill=MU)
        return
    n = min(len(items), max_rows)
    rh = min(96, (H - 120 - top) // n)
    for i, it in enumerate(items[:n]):
        y = top + i * rh
        d.rounded_rectangle((52, y, W - 52, y + rh - 12), radius=16, fill=CARD)
        bs = min(56, rh - 28)
        _badge(d, 72, y + (rh - 12 - bs) / 2, it["team"], bs)
        x = 72 + bs + 22
        cw = _chip(d, x, y + 12, it["tier"])
        d.text((x + cw + 14, y + 13), it["sub"], font=_font(FONT_REG, 22), fill=MU)
        d.text((x, y + 44), it["pick"], font=_font(FONT_COND, max(24, min(34, rh - 60))), fill=TX)
        f = _font(FONT_COND, 34)
        rt = it["right"]
        d.text((W - 76 - d.textlength(rt, font=f), y + (rh - 12) / 2 - 20), rt, font=f,
               fill=TIER.get(it["tier"], TX))


def game_items(picks):
    out = []
    for _, r in picks.iterrows():
        mu = f"{r.away_team} @ {r.home_team}"
        day = pd.to_datetime(r.gameday).strftime("%a").upper()
        if isinstance(r.sp_tier, str) and r.sp_tier:
            out.append(dict(team=r.sp_side, tier=r.sp_tier, pick=r.sp_pick, sub=f"{mu} · {day} · SPREAD",
                            right=f"{r.sp_edge:.1f} PTS", k=(RANK[r.sp_tier], r.sp_edge / 7)))
        if isinstance(r.tot_tier, str) and r.tot_tier:
            out.append(dict(team="O/U", tier=r.tot_tier, pick=f"{mu}  {r.tot_pick}",
                            sub=f"{day} · TOTAL", right=f"{r.tot_edge:.1f} PTS",
                            k=(RANK[r.tot_tier], r.tot_edge / 10)))
        if isinstance(r.get("ml_pick"), str) and r.ml_pick:
            out.append(dict(team=r.ml_pick, tier="VALUE", pick=f"{r.ml_pick} ML {int(r.ml_odds):+d}",
                            sub=f"{mu} · {day} · MONEYLINE", right=f"+{r.ml_edge * 100:.0f}%",
                            k=(RANK["VALUE"], r.ml_edge * 10)))
    return [x for x in sorted(out, key=lambda x: x["k"], reverse=True)]


def prop_items(props):
    if props is None or props.empty:
        return []
    p = props[props.tier.fillna("") != ""].copy()
    if p.empty:
        return []
    p["_r"] = p.tier.map(RANK)
    p = p.sort_values(["_r", "edge"], ascending=False)
    out = []
    for _, r in p.iterrows():
        td = r.market == "atd"
        pick = f"{r.player}  TD" if td else f"{r.player}  {r.pick}"
        sub = f"{r.label.upper()} · {int(r.price):+d}" + (f" · COV {r.cov_mark}" if isinstance(r.get("cov_mark"), str) and r.cov_mark else "")
        out.append(dict(team=r.team, tier=r.tier, pick=pick, sub=sub, right=f"+{r.edge * 100:.0f}%"))
    return out


def make_posts(picks, props, season, week, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    tag = f"{season}_wk{week:02d}"
    files = []
    gi = game_items(picks)
    shown = f" · top 9 shown" if len(gi) > 9 else ""
    im, d = _canvas("MODEL PLAYS", f"{len(gi)} plays · spreads, totals, moneyline{shown}", season, week)
    _rows(d, gi)
    f = out_dir / f"{tag}_plays.png"; im.save(f); files.append(f)

    pi = prop_items(props)
    if pi:
        im, d = _canvas("TOP PLAYER PROPS", f"{len(pi)} prop plays · top {min(len(pi), 9)} shown", season, week)
        _rows(d, pi)
        f = out_dir / f"{tag}_props.png"; im.save(f); files.append(f)
        td = [x for x in pi if x["pick"].endswith("TD")]
        if td:
            im, d = _canvas("ANYTIME TD CALLS", f"{len(td)} TD plays", season, week)
            _rows(d, td)
            f = out_dir / f"{tag}_tds.png"; im.save(f); files.append(f)
    return files
