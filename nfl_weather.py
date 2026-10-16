"""
nfl_weather.py: kickoff-time forecast for every outdoor NFL game (Open-Meteo, free, no key).

The game model was trained WITH temp + wind (nflverse records them for past games), but for
upcoming games those columns are blank, so the model was quietly using 60F / 8 mph for every game.
This fills in the real forecast at kickoff so the model's totals/spreads actually see the weather.

History (2012-2025, outdoor games): totals by wind
   0-5 mph 46.5 pts | 5-10 45.6 | 10-15 43.5 | 15-20 42.2
and books UNDER-adjust: 10-20 mph games land 1-2 pts under the closing total.
"""
from datetime import datetime
from zoneinfo import ZoneInfo

import requests

ET = ZoneInfo("America/New_York")

# stadium name keyword -> (lat, lon)
STADIUMS = {
    "m&t bank": (39.2780, -76.6227), "highmark": (42.7738, -78.7870), "bank of america": (35.2258, -80.8528),
    "soldier": (41.8623, -87.6167), "paycor": (39.0954, -84.5160), "huntington bank": (41.5061, -81.6995),
    "empower": (39.7439, -105.0201), "lambeau": (44.5013, -88.0622), "everbank": (30.3239, -81.6373),
    "arrowhead": (39.0489, -94.4839), "hard rock": (25.9580, -80.2389), "gillette": (42.0909, -71.2643),
    "metlife": (40.8135, -74.0745), "lincoln financial": (39.9008, -75.1675), "acrisure": (40.4468, -80.0158),
    "lumen": (47.5952, -122.3316), "levi": (37.4030, -121.9700), "raymond james": (27.9759, -82.5033),
    "nissan": (36.1665, -86.7713), "northwest": (38.9076, -76.8645), "fedex": (38.9076, -76.8645),
    "state farm": (33.5276, -112.2626), "mercedes-benz": (33.7554, -84.4008), "at&t": (32.7473, -97.0945),
    "reliant": (29.6847, -95.4107), "nrg": (29.6847, -95.4107), "lucas oil": (39.7601, -86.1639),
    "wembley": (51.5560, -0.2796), "tottenham": (51.6043, -0.0664), "bernabeu": (40.4531, -3.6883),
    "maracana": (-22.9122, -43.2302), "allianz": (48.2188, 11.6247), "bayern": (48.2188, 11.6247),
    "deutsche bank": (50.0686, 8.6455), "frankfurt": (50.0686, 8.6455), "croke": (53.3607, -6.2512),
    "stade de france": (48.9245, 2.3602), "banorte": (19.3029, -99.1505), "azteca": (19.3029, -99.1505),
    "melbourne": (-37.8200, 144.9834),
}
RETRACTABLE = {"ARI", "ATL", "DAL", "HOU", "IND"}   # roof blank in the schedule -> assume closed
DOME_WORDS = ("dome", "closed")


def _coords(stadium):
    s = (stadium or "").lower()
    for k, v in STADIUMS.items():
        if k in s:
            return v
    return None


def is_indoors(roof, home_team, stadium):
    r = str(roof or "").lower()
    if r in DOME_WORDS:
        return True
    if r in ("outdoors", "open"):
        return False
    s = (stadium or "").lower()
    return home_team in RETRACTABLE and not any(k in s for k in ("bernabeu", "maracana", "wembley", "tottenham"))


def kickoff_forecast(stadium, gameday, gametime_et):
    """{'temp','wind','gust','rain_pct','precip_in'} at kickoff, or None."""
    ll = _coords(stadium)
    if not ll:
        return None
    try:
        day = str(gameday)[:10]
        hh, mm = (str(gametime_et) or "13:00")[:5].split(":")
        ko = datetime.fromisoformat(f"{day}T{int(hh):02d}:{int(mm):02d}").replace(tzinfo=ET)
        r = requests.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": ll[0], "longitude": ll[1], "timezone": "GMT", "forecast_days": 16,
            "hourly": "temperature_2m,wind_speed_10m,wind_gusts_10m,precipitation_probability,precipitation",
            "temperature_unit": "fahrenheit", "wind_speed_unit": "mph", "precipitation_unit": "inch"}, timeout=15)
        h = r.json()["hourly"]
        ts = [datetime.fromisoformat(t).replace(tzinfo=ZoneInfo("UTC")) for t in h["time"]]
        # average the 3 game hours (kickoff +0h, +1h, +2h)
        idx = [i for i, t in enumerate(ts) if 0 <= (t - ko).total_seconds() / 3600 < 3]
        if not idx:
            return None
        avg = lambda k: sum((h[k][i] or 0) for i in idx) / len(idx)
        return {"temp": round(avg("temperature_2m")), "wind": round(avg("wind_speed_10m")),
                "gust": round(max(h["wind_gusts_10m"][i] or 0 for i in idx)),
                "rain_pct": round(max(h["precipitation_probability"][i] or 0 for i in idx)),
                "precip_in": round(sum(h["precipitation"][i] or 0 for i in idx), 2)}
    except Exception as ex:
        print(f"  ! weather {stadium}: {type(ex).__name__}")
        return None


def apply(wk):
    """Fill temp/wind/dome on this week's games with kickoff forecasts. Adds wx_* columns."""
    for c in ("wx_label", "rain_pct", "precip_in", "gust"):
        wk[c] = "" if c == "wx_label" else 0.0
    n = 0
    for i, r in wk.iterrows():
        if is_indoors(r.get("roof"), r.home_team, r.get("stadium")):
            wk.at[i, "dome"], wk.at[i, "temp"], wk.at[i, "wind"] = 1, 70, 0
            continue
        f = kickoff_forecast(r.get("stadium"), r.gameday, r.gametime)
        if not f:
            continue
        n += 1
        wk.at[i, "dome"], wk.at[i, "temp"], wk.at[i, "wind"] = 0, f["temp"], f["wind"]
        wk.at[i, "rain_pct"], wk.at[i, "precip_in"], wk.at[i, "gust"] = f["rain_pct"], f["precip_in"], f["gust"]
        lab = f"{f['temp']}°F · {f['wind']} MPH"
        if f["rain_pct"] >= 40:
            lab += f" · {'SNOW' if f['temp'] <= 33 else 'RAIN'} {f['rain_pct']}%"
        wk.at[i, "wx_label"] = lab
        print(f"  weather {r.away_team} @ {r.home_team}: {lab} (gusts {f['gust']})")
    print(f"  kickoff forecasts for {n} outdoor games")
    return wk
