"""AlgoHub NFL - Streamlit site.  Run locally:  streamlit run app.py"""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from cards import render_cards

ROOT = Path(__file__).resolve().parent
PICKS, TRACKER, BACKTEST = ROOT / "picks", ROOT / "tracker.csv", ROOT / "backtest_results.csv"
TIER_RANK = {"HAMMER": 3, "PLAY": 2, "LEAN": 1}

st.set_page_config(page_title="AlgoHub NFL", page_icon="🏈", layout="wide")
st.markdown("""<style>
.block-container{padding-top:1.4rem;max-width:1240px}
[data-testid="stMetricValue"]{font-size:1.9rem}
</style>""", unsafe_allow_html=True)


@st.cache_data(ttl=300)
def load_csv(path, mtime):
    return pd.read_csv(path)


def read(path):
    return load_csv(str(path), path.stat().st_mtime) if path.exists() else None


# ----------------------------------------------------------------------------
files = sorted((f for f in PICKS.glob("*_wk*.csv") if not f.name.startswith("props_")), reverse=True)
st.markdown("#### THE ALGOHUB · NFL GAME ALGO")
if not files:
    st.info("No picks yet. Run `python nfl_algo.py predict --week N` and push the picks folder.")
    st.stop()

opts = {f"{f.stem[:4]} · Week {int(f.stem.split('wk')[1])}": f for f in files}
c1, _ = st.columns([1, 3])
choice = c1.selectbox("Week", list(opts), label_visibility="collapsed")
picks = read(opts[choice])
season, week = int(picks.season.iloc[0]), int(picks.week.iloc[0])
props = read(PICKS / f"props_{season}_wk{week:02d}.csv")

t_cards, t_bets, t_posts, t_record, t_bt = st.tabs(["Cards", "Best Bets", "Posts", "Record", "Backtest"])

# --- cards ------------------------------------------------------------------
with t_cards:
    n_days = picks.gameday.nunique()
    height = 330 + n_days * 70 + math.ceil(len(picks) / 3) * 960
    st.caption("Tap **PLAYER PROPS** on any game to open its props.")
    components.html(render_cards(picks, season, week, props), height=height, scrolling=True)

# --- best bets --------------------------------------------------------------
with t_bets:
    rows = []
    for _, r in picks.iterrows():
        mu = f"{r.away_team} @ {r.home_team}"
        if isinstance(r.sp_tier, str) and r.sp_tier:
            rows.append(dict(Matchup=mu, Day=r.gameday, Market="Spread", Pick=r.sp_pick, Tier=r.sp_tier,
                             Edge=round(r.sp_edge, 1), Vegas=r.market_line, Model=r.model_line))
        if isinstance(r.tot_tier, str) and r.tot_tier:
            rows.append(dict(Matchup=mu, Day=r.gameday, Market="Total", Pick=r.tot_pick, Tier=r.tot_tier,
                             Edge=round(r.tot_edge, 1), Vegas=f"{r.total_line:g}", Model=f"{r.pred_total:.1f}"))
        if isinstance(r.get("ml_pick"), str) and r.ml_pick:
            rows.append(dict(Matchup=mu, Day=r.gameday, Market="ML", Pick=f"{r.ml_pick} {int(r.ml_odds):+d}",
                             Tier="VALUE", Edge=round(r.ml_edge * 100, 1), Vegas="", Model=f"{r.home_wp:.0%} home"))
    if props is not None and not props.empty:
        for _, p in props[props.tier.fillna("") != ""].iterrows():
            g = picks[picks.game_id == p.game_id]
            mu = f"{g.away_team.iloc[0]} @ {g.home_team.iloc[0]}" if len(g) else p.team
            is_td = p.market == "atd"
            rows.append(dict(Matchup=mu, Day=g.gameday.iloc[0] if len(g) else "", Market=f"Prop · {p.label}",
                             Pick=f"{p.player} {'TD' if is_td else p.pick} ({int(p.price):+d})",
                             Tier=p.tier, Edge=round(p.edge * 100, 1),
                             Vegas=f"{int(p.over_price):+d}" if is_td else f"{p.line:g}",
                             Model=f"{p.proj:.0%}" if is_td else f"{p.proj:.1f}"))
    if rows:
        bets = pd.DataFrame(rows)
        bets["_r"] = bets.Tier.map(TIER_RANK).fillna(0)
        bets = bets.sort_values(["_r", "Edge"], ascending=False).drop(columns="_r")
        m1, m2, m3 = st.columns(3)
        m1.metric("Hammers", int((bets.Tier == "HAMMER").sum()))
        m2.metric("Plays", int((bets.Tier == "PLAY").sum()))
        m3.metric("Leans", int((bets.Tier == "LEAN").sum()))
        st.dataframe(bets, hide_index=True, use_container_width=True)
        st.caption("Edge: points vs. Vegas for spreads/totals · % over no-vig for ML · probability points over the DraftKings break-even for props.")
    else:
        st.info("No plays clear the thresholds this week.")

# --- posts ------------------------------------------------------------------
with t_posts:
    imgs = sorted((ROOT / "posts").glob(f"{season}_wk{week:02d}_*.png"))
    if not imgs:
        st.info("No post graphics for this week yet. They're made automatically by `predict`.")
    else:
        st.caption("Ready to post. Tap download, or long-press the image on your phone to save.")
        cols = st.columns(min(len(imgs), 3))
        for i, f in enumerate(imgs):
            with cols[i % len(cols)]:
                st.image(str(f), use_container_width=True)
                st.download_button(f"Download {f.stem.split('_')[-1]}", f.read_bytes(), file_name=f.name,
                                   mime="image/png", key=f.name, use_container_width=True)

# --- record -----------------------------------------------------------------
with t_record:
    tr = read(TRACKER)
    if tr is None or tr.empty:
        st.info("Nothing graded yet. Run `python nfl_algo.py grade --week N` after games finish.")
    else:
        seasons = sorted(tr.season.unique(), reverse=True)
        s = st.selectbox("Season", seasons, key="rec_season") if len(seasons) > 1 else seasons[0]
        t = tr[tr.season == s]
        w, l, p = (t.result == "W").sum(), (t.result == "L").sum(), (t.result == "P").sum()
        u = t.units.sum()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Record", f"{w}-{l}-{p}")
        m2.metric("Win %", f"{w / max(w + l, 1):.1%}")
        m3.metric("Units", f"{u:+.2f}")
        m4.metric("ROI", f"{u / max(w + l, 1):+.1%}")

        curve = t.groupby("week").units.sum().cumsum().rename("Cumulative units")
        st.line_chart(curve)

        agg = (t.groupby(["market", "tier"])
               .agg(W=("result", lambda x: (x == "W").sum()), L=("result", lambda x: (x == "L").sum()),
                    P=("result", lambda x: (x == "P").sum()), Units=("units", "sum")).reset_index())
        agg["Win %"] = (agg.W / (agg.W + agg.L).clip(lower=1)).map("{:.1%}".format)
        agg["Units"] = agg.Units.round(2)
        st.dataframe(agg.rename(columns={"market": "Market", "tier": "Tier"}), hide_index=True,
                     use_container_width=True)
        with st.expander("Every graded pick"):
            st.dataframe(t.sort_values(["week", "matchup"], ascending=[False, True]), hide_index=True,
                         use_container_width=True)

# --- backtest ---------------------------------------------------------------
with t_bt:
    bt = read(BACKTEST)
    if bt is None:
        st.info("Run `python nfl_algo.py backtest` and push backtest_results.csv.")
    else:
        def ats(edge, res, thrs):
            out = []
            for thr in thrs:
                sel = edge.abs() >= thr
                e, r = edge[sel], res[sel]
                w = int(((np.sign(e) == np.sign(r)) & (r != 0)).sum())
                p = int((r == 0).sum())
                l = int(sel.sum()) - w - p
                out.append({"Edge ≥": thr, "Record": f"{w}-{l}-{p}", "Win %": f"{w / max(w + l, 1):.1%}",
                            "Units": round(w * 100 / 110 - l, 1)})
            return pd.DataFrame(out)

        sp, to = bt.dropna(subset=["spread_line"]), bt.dropna(subset=["total_line"])
        st.caption(f"Walk-forward: each season predicted by a model trained only on earlier seasons "
                   f"({int(bt.season.min())}–{int(bt.season.max())}, {len(bt):,} games). Break-even at -110 = 52.4%.")
        a, b = st.columns(2)
        a.markdown("**Spread**")
        a.dataframe(ats(sp.pred_margin - sp.spread_line, sp.margin - sp.spread_line, [0, 1, 2, 3, 4.5]),
                    hide_index=True, use_container_width=True)
        b.markdown("**Total**")
        b.dataframe(ats(to.pred_total - to.total_line, to.total - to.total_line, [0, 1.5, 2.5, 4, 6]),
                    hide_index=True, use_container_width=True)
        a.metric("Margin MAE · model / Vegas",
                 f"{(sp.margin - sp.pred_margin).abs().mean():.2f} / {(sp.margin - sp.spread_line).abs().mean():.2f}")
        b.metric("Total MAE · model / Vegas",
                 f"{(to.total - to.pred_total).abs().mean():.2f} / {(to.total - to.total_line).abs().mean():.2f}")

st.caption("For entertainment. Not betting advice.")
