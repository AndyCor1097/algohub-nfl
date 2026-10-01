"""AlgoHub NFL cards - dark, edge-first layout. render_cards(df, season, week) -> html str."""
import html
import math
from datetime import datetime

import pandas as pd

COLORS = {  # primary, text-on-primary
    "ARI": ("#97233F", "#fff"), "ATL": ("#A71930", "#fff"), "BAL": ("#241773", "#fff"),
    "BUF": ("#00338D", "#fff"), "CAR": ("#0085CA", "#fff"), "CHI": ("#0B162A", "#fff"),
    "CIN": ("#FB4F14", "#000"), "CLE": ("#FF3C00", "#000"), "DAL": ("#041E42", "#fff"),
    "DEN": ("#FB4F14", "#000"), "DET": ("#0076B6", "#fff"), "GB": ("#203731", "#FFB612"),
    "HOU": ("#03202F", "#fff"), "IND": ("#002C5F", "#fff"), "JAX": ("#006778", "#fff"),
    "KC": ("#E31837", "#fff"), "LA": ("#003594", "#FFD100"), "LAC": ("#0080C6", "#fff"),
    "LV": ("#000000", "#A5ACAF"), "MIA": ("#008E97", "#fff"), "MIN": ("#4F2683", "#FFC62F"),
    "NE": ("#002244", "#fff"), "NO": ("#D3BC8D", "#000"), "NYG": ("#0B2265", "#fff"),
    "NYJ": ("#125740", "#fff"), "PHI": ("#004C54", "#fff"), "PIT": ("#FFB612", "#000"),
    "SEA": ("#002244", "#69BE28"), "SF": ("#AA0000", "#fff"), "TB": ("#D50A0A", "#fff"),
    "TEN": ("#0C2340", "#4B92DB"), "WAS": ("#5A1414", "#FFB612"),
}
NAMES = {
    "ARI": "Cardinals", "ATL": "Falcons", "BAL": "Ravens", "BUF": "Bills", "CAR": "Panthers",
    "CHI": "Bears", "CIN": "Bengals", "CLE": "Browns", "DAL": "Cowboys", "DEN": "Broncos",
    "DET": "Lions", "GB": "Packers", "HOU": "Texans", "IND": "Colts", "JAX": "Jaguars",
    "KC": "Chiefs", "LA": "Rams", "LAC": "Chargers", "LV": "Raiders", "MIA": "Dolphins",
    "MIN": "Vikings", "NE": "Patriots", "NO": "Saints", "NYG": "Giants", "NYJ": "Jets",
    "PHI": "Eagles", "PIT": "Steelers", "SEA": "Seahawks", "SF": "49ers", "TB": "Buccaneers",
    "TEN": "Titans", "WAS": "Commanders",
}
TIER_RANK = {"HAMMER": 3, "PLAY": 2, "LEAN": 1, "": 0}
e = lambda s: html.escape(str(s)) if s is not None and not (isinstance(s, float) and math.isnan(s)) else ""


def _s(v):
    return "" if v is None or (isinstance(v, float) and math.isnan(v)) else str(v)


def _kick(r):
    d = pd.to_datetime(r["gameday"])
    t = _s(r.get("gametime"))
    try:
        t = datetime.strptime(t[:5], "%H:%M").strftime("%I:%M %p").lstrip("0")  # Windows-safe
    except ValueError:
        t = "TBD"
    return d.strftime("%a").upper(), f"{d.strftime('%A, %B').upper()} {d.day}", f"{t} ET"


def _badge(t):
    bg, fg = COLORS.get(t, ("#30363d", "#fff"))
    return f'<span class="badge" style="background:{bg};color:{fg}">{e(t)}</span>'


def _team_row(r, s, win_side):
    t = r[f"{'away' if s == 'a' else 'home'}_team"]
    proj = r[f"proj_{'away' if s == 'a' else 'home'}"]
    qb = _s(r.get(f"{s}_qb_name"))
    sub = " · ".join(x for x in (_s(r.get(f"{s}_rec")), qb) if x)
    return f"""
      <div class="team {'fav' if win_side == s else ''}">
        {_badge(t)}
        <div class="tname"><b>{e(NAMES.get(t, t))}</b><small>{e(sub)}</small></div>
        <div class="proj">{proj:.1f}</div>
      </div>"""


def _wp(r):
    a, h = r["away_team"], r["home_team"]
    hw = float(r["home_wp"])
    ca, ch = COLORS.get(a, ("#555",))[0], COLORS.get(h, ("#777",))[0]
    return f"""
      <div class="wp">
        <div class="wplab"><span>{e(a)} {1 - hw:.0%}</span><span class="cap">WIN PROB</span><span>{e(h)} {hw:.0%}</span></div>
        <div class="wpbar"><i style="width:{(1 - hw) * 100:.1f}%;background:{ca}"></i><i style="width:{hw * 100:.1f}%;background:{ch}"></i></div>
      </div>"""


def _market(label, vegas, model, edge, tier_, pick):
    has = isinstance(tier_, str) and tier_
    edge_txt = "-" if edge is None or (isinstance(edge, float) and math.isnan(edge)) else f"{edge:.1f}"
    pick_html = (f'<div class="pick t-{tier_.lower()}"><span class="tier">{e(tier_)}</span>{e(pick)}</div>'
                 if has else '<div class="pick none">NO PLAY</div>')
    return f"""
      <div class="mkt">
        <div class="mlabel">{label}</div>
        <div class="cols">
          <div><small>VEGAS</small><span>{e(vegas)}</span></div>
          <div><small>MODEL</small><span>{e(model)}</span></div>
          <div class="edge {'hot' if has else ''}"><small>EDGE</small><span>{edge_txt}</span></div>
        </div>
        {pick_html}
      </div>"""


def _strip(r):
    rows = []
    for s, o in (("a", "h"), ("h", "a")):
        t = r["away_team" if s == "a" else "home_team"]
        ot = r["home_team" if s == "a" else "away_team"]
        for kind in ("pass", "rush"):
            v = float(r[f"{s}_{kind}_mu"])
            w = min(abs(v) / 0.30, 1) * 50
            cls = "pos" if v > 0 else "neg"
            rows.append(f"""
          <div class="srow"><span class="slab">{e(t)} {kind.upper()} O <em>vs</em> {e(ot)} D</span>
            <span class="sbar"><i class="{cls}" style="width:{w:.1f}%"></i></span>
            <span class="sval {cls}">{v:+.2f}</span></div>""")
    return f'<div class="strip"><div class="mlabel">EPA MATCHUP <em>(+ = offense edge)</em></div>{"".join(rows)}</div>'


def _card(r):
    sp_t, tot_t = _s(r.get("sp_tier")), _s(r.get("tot_tier"))
    best = max((sp_t, tot_t), key=lambda x: TIER_RANK.get(x, 0))
    dkey, _, kick = _kick(r)
    win_side = "h" if r["pred_margin"] > 0 else "a"
    flags = [f for f in _s(r.get("flags")).split("|") if f]
    chips = "".join(f'<span class="chip">{e(f)}</span>' for f in flags)
    tot_model = f'{r["pred_total"]:.1f}'
    tot_vegas = "-" if pd.isna(r.get("total_line")) else f'{r["total_line"]:g}'
    ml = ""
    if _s(r.get("ml_pick")):
        ml = (f'<div class="ml">ML VALUE <b>{e(r["ml_pick"])} {int(r["ml_odds"]):+d}</b>'
              f'<span>+{r["ml_edge"]:.1%} vs no-vig</span></div>')
    return f"""
    <article class="card b-{best.lower() or 'none'}" data-day="{dkey}" data-play="{1 if best else 0}" data-rank="{TIER_RANK[best]}">
      <header class="meta"><span>{dkey} · {e(kick)}</span><span>{e(_s(r.get('stadium')))}</span></header>
      {_team_row(r, 'a', win_side)}
      {_team_row(r, 'h', win_side)}
      {_wp(r)}
      {_market("SPREAD", r["market_line"], r["model_line"], r.get("sp_edge"), sp_t, r.get("sp_pick"))}
      {_market("TOTAL", tot_vegas, tot_model, r.get("tot_edge"), tot_t, r.get("tot_pick"))}
      {ml}
      {_strip(r)}
      {f'<div class="chips">{chips}</div>' if chips else ''}
    </article>"""


def render_cards(df, season, week):
    df = df.copy()
    df["_order"] = pd.to_datetime(df.gameday).astype("int64")
    df = df.sort_values(["_order", "gametime"])
    sections, days = [], []
    for gd, g in df.groupby("gameday", sort=False):
        dkey, dlong, _ = _kick(g.iloc[0])
        days.append(dkey)
        cards = "".join(_card(r) for r in g.to_dict("records"))
        sections.append(f"""
  <section class="day" data-day="{dkey}">
    <div class="dayhead"><h2>{e(dlong)}</h2><span>{len(g)} GAME{'S' if len(g) != 1 else ''}</span></div>
    <div class="grid">{cards}</div>
  </section>""")
    tiers = list(df.sp_tier.fillna("")) + list(df.tot_tier.fillna(""))
    n_play = sum(1 for t in tiers if t)
    n_ham = sum(1 for t in tiers if t == "HAMMER")
    uniq = list(dict.fromkeys(days))
    tabs = "".join(f'<button data-f="{d}">{d}</button>' for d in uniq)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AlgoHub NFL Week {week}</title>
<link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;700;800&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{{--bg:#0b0e13;--card:#12161d;--card2:#171c25;--line:#222a35;--tx:#e6edf3;--mu:#7d8590;
--lean:#58a6ff;--play:#3fb950;--ham:#f2b134;--neg:#f85149}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--tx);font:14px/1.4 Inter,system-ui,sans-serif;padding:28px 16px 60px}}
.wrap{{max-width:1180px;margin:0 auto}}
.num,.proj,.cols span,.top h1,.stats b,.wplab,.sval,.pick{{font-family:"Barlow Condensed",sans-serif}}
.top{{display:flex;justify-content:space-between;align-items:flex-end;gap:16px;flex-wrap:wrap;border-bottom:1px solid var(--line);padding-bottom:18px}}
.top .k{{color:var(--mu);font-size:12px;letter-spacing:.18em;font-weight:600}}
.top h1{{font-size:44px;font-weight:800;letter-spacing:.02em;line-height:1}}
.stats{{display:flex;gap:22px}} .stats div{{text-align:right}} .stats small{{display:block;color:var(--mu);font-size:11px;letter-spacing:.14em}}
.stats b{{font-size:30px}} .stats .h b{{color:var(--ham)}} .stats .p b{{color:var(--play)}}
.tabs{{display:flex;gap:6px;margin:18px 0 6px;flex-wrap:wrap}}
.tabs button{{background:transparent;border:1px solid var(--line);color:var(--mu);padding:7px 14px;border-radius:999px;font:600 12px Inter,system-ui,sans-serif;letter-spacing:.08em;cursor:pointer}}
.tabs button.on{{background:var(--tx);color:var(--bg);border-color:var(--tx)}}
.dayhead{{display:flex;justify-content:space-between;align-items:baseline;margin:26px 0 12px}}
.dayhead h2{{font:700 13px Inter,system-ui,sans-serif;letter-spacing:.16em}} .dayhead span{{color:var(--mu);font-size:11px;letter-spacing:.14em}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:14px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:14px;overflow:hidden;border-top:3px solid var(--line)}}
.card.b-lean{{border-top-color:var(--lean)}} .card.b-play{{border-top-color:var(--play)}}
.card.b-hammer{{border-top-color:var(--ham);box-shadow:0 0 0 1px rgba(242,177,52,.25),0 8px 30px rgba(242,177,52,.08)}}
.meta{{display:flex;justify-content:space-between;gap:10px;padding:11px 16px;color:var(--mu);font-size:11px;letter-spacing:.08em;border-bottom:1px solid var(--line)}}
.meta span:last-child{{text-align:right;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.team{{display:flex;align-items:center;gap:12px;padding:10px 16px}}
.badge{{width:40px;height:40px;border-radius:10px;display:grid;place-items:center;font:800 13px "Barlow Condensed";letter-spacing:.04em;flex:none}}
.tname{{flex:1;min-width:0}} .tname b{{display:block;font-size:16px;font-weight:600}} .tname small{{color:var(--mu);font-size:12px}}
.proj{{font-size:32px;font-weight:700;color:var(--mu)}} .team.fav .proj{{color:var(--tx)}}
.wp{{padding:6px 16px 14px}} .wplab{{display:flex;justify-content:space-between;font-size:15px;font-weight:700;margin-bottom:6px}}
.wplab .cap{{color:var(--mu);font:600 10px Inter,system-ui,sans-serif;letter-spacing:.16em;align-self:center}}
.wpbar{{display:flex;height:8px;border-radius:99px;overflow:hidden;gap:2px;background:var(--line)}} .wpbar i{{display:block;height:100%}}
.mkt{{border-top:1px solid var(--line);padding:12px 16px;background:var(--card2)}}
.mlabel{{color:var(--mu);font:600 10px Inter,system-ui,sans-serif;letter-spacing:.18em;margin-bottom:8px}} .mlabel em{{font-style:normal;letter-spacing:.04em;opacity:.7}}
.cols{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px}} .cols small{{display:block;color:var(--mu);font-size:10px;letter-spacing:.14em}}
.cols span{{font-size:20px;font-weight:700}} .edge span{{color:var(--mu)}} .edge.hot span{{color:var(--tx);font-size:26px}}
.pick{{margin-top:10px;display:flex;align-items:center;gap:10px;font-size:20px;font-weight:700;letter-spacing:.02em}}
.pick .tier{{font:800 11px Inter,system-ui,sans-serif;letter-spacing:.14em;padding:4px 8px;border-radius:6px;color:#0b0e13}}
.t-lean .tier{{background:var(--lean)}} .t-play .tier{{background:var(--play)}} .t-hammer .tier{{background:var(--ham)}}
.t-hammer{{color:var(--ham)}} .pick.none{{color:#4b535d;font:600 12px Inter,system-ui,sans-serif;letter-spacing:.16em}}
.ml{{border-top:1px solid var(--line);padding:10px 16px;font-size:12px;color:var(--mu);letter-spacing:.06em;display:flex;gap:10px;align-items:baseline;flex-wrap:wrap}}
.ml b{{color:var(--play);font-size:14px}}
.strip{{border-top:1px solid var(--line);padding:12px 16px}}
.srow{{display:grid;grid-template-columns:1fr 90px 46px;align-items:center;gap:10px;font-size:12px;padding:3px 0}}
.slab em{{color:var(--mu);font-style:normal}}
.sbar{{position:relative;height:6px;background:var(--line);border-radius:99px}}
.sbar::after{{content:"";position:absolute;left:50%;top:-3px;width:1px;height:12px;background:var(--mu)}}
.sbar i{{position:absolute;top:0;height:100%;border-radius:99px}} .sbar i.pos{{left:50%;background:var(--play)}} .sbar i.neg{{right:50%;background:var(--neg)}}
.sval{{text-align:right;font-size:15px;font-weight:700}} .sval.pos{{color:var(--play)}} .sval.neg{{color:var(--neg)}}
.chips{{display:flex;flex-wrap:wrap;gap:6px;padding:0 16px 14px}}
.chip{{font:600 10px Inter,system-ui,sans-serif;letter-spacing:.1em;color:#c9d1d9;border:1px solid var(--line);background:#0f1319;padding:4px 8px;border-radius:6px}}
.foot{{color:var(--mu);font-size:11px;margin-top:30px;letter-spacing:.06em}}
.hide{{display:none}}
</style></head><body><div class="wrap">
  <div class="top">
    <div><div class="k">THE ALGOHUB · NFL GAME ALGO · {season}</div><h1>WEEK {week}</h1></div>
    <div class="stats">
      <div><small>GAMES</small><b>{len(df)}</b></div>
      <div class="p"><small>PLAYS</small><b>{n_play}</b></div>
      <div class="h"><small>HAMMERS</small><b>{n_ham}</b></div>
    </div>
  </div>
  <div class="tabs"><button class="on" data-f="all">ALL</button><button data-f="plays">PLAYS ONLY</button>{tabs}</div>
  {''.join(sections)}
  <p class="foot">Model trained on in-season team EPA only, shrunk toward league average. Edges in points vs. market. LEAN / PLAY / HAMMER tiers. Not betting advice.</p>
</div>
<script>
document.querySelectorAll('.tabs button').forEach(b=>b.onclick=()=>{{
  document.querySelectorAll('.tabs button').forEach(x=>x.classList.toggle('on',x===b));
  const f=b.dataset.f;
  document.querySelectorAll('.card').forEach(c=>c.classList.toggle('hide',
    f==='plays'?c.dataset.play!=='1':(f!=='all'&&c.dataset.day!==f)));
  document.querySelectorAll('.day').forEach(s=>s.classList.toggle('hide',
    !s.querySelector('.card:not(.hide)')));
}});
</script></body></html>"""
