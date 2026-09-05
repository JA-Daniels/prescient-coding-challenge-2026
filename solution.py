"""
Prescient Coding Challenge 2026 -- your submission.

THIS IS THE ONLY FILE YOU MAY CHANGE.

You implement one function. The harness calls it once per trading day and hands
you a `hist` object holding every observation STRICTLY BEFORE that day. You
return the weights you want to hold for that day.

    generate_weights(hist, prev_weights, params) -> weights

What you get
------------
hist.date                 the day you are allocating for (no data for it yet)
hist.returns              DataFrame [date x asset] of daily returns, decimals
hist.prices               DataFrame [date x asset] of total-return index levels
hist.macro                DataFrame [date x macro feature]
hist.assets               list of the six asset codes, in order
hist.benchmark            Series of benchmark weights
hist.active_weight(w)     total active weight of w -- the number rule 3 tests

prev_weights              what you held yesterday. Trading away from it costs
                          money, so look at it.
params                    the PARAMS dict below, passed straight through

Optional extras, in case you want them: hist.cov() gives an EWMA covariance
matrix and hist.te(w) an ex-ante tracking error. No rule depends on either.

What you must return
--------------------
Six weights (dict, Series or array in hist.assets order) that sum to 1, are all
non-negative, sit within 10% of their benchmark weight, have a total active
weight of no more than 40%, keep total equity at or below 75% and gold at or
below 10%. `make_legal()` below
already does all of that -- you can leave it alone.

Declare every tuneable number in PARAMS. Parameter count is part of the score.

Run `python harness.py` to test on the practice window (calendar 2025), then
`python validate.py` before you submit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Every tuneable number lives here. Fewer is better.
# --------------------------------------------------------------------------- #

PARAMS = {
    "trend_fast": 63,
    "trend_mid": 126,
    "trend_slow": 252,
    "vol_days": 126,
    "active_target": 0.18,
    "trade_speed": 0.25,
    "no_trade": 0.002,
}

# The rules, restated locally so this file reads on its own.
ACTIVE_BAND = 0.10       # per asset, distance from benchmark
ACTIVE_BUDGET = 0.40     # total, summed over assets
EQUITY = ["SA_EQUITY", "GLOBAL_EQUITY"]
EQUITY_CAP = 0.75        # total equity, whatever the bands allow
GOLD_CAP = 0.10


# --------------------------------------------------------------------------- #
# <<--------------------- YOUR CODE GOES BELOW THIS LINE --------------------->>
#
# This is your playground. Delete or rewrite anything here. What follows is a
# deliberately naive starting point so you can see the shape of a working
# answer. It is NOT a good answer -- on the practice window it loses to the
# benchmark. Your job is to do better.
#
# Three steps:
#   1. build a signal (here: a plain inverse-volatility tilt, which knows
#      nothing at all about expected return),
#   2. make the weights legal,
#   3. move only part of the way from yesterday, so you do not pay the full
#      trading cost every day.
#
# Steps 2 and 3 are plumbing. Keep them. Step 1 is the actual question, and
# inverse volatility is a poor answer to it: it will always prefer cash and
# bonds, whatever is happening in the world.
#
# Things worth thinking about. Which of these six assets actually diversifies
# the other five? Gold and global equity are both priced in rands -- what does
# that mean when the currency moves? The macro file has a term spread and a
# policy rate in it; what should a steepening curve do to your bond weight? And
# look at the cost table in the README before you trade property daily.
# --------------------------------------------------------------------------- #


def _clip(x, lo=-3.0, hi=3.0):
    return float(np.clip(x, lo, hi))


def _price_momentum(series, horizon, daily_vol):
    if len(series) <= horizon or not np.isfinite(daily_vol) or daily_vol <= 1e-12:
        return 0.0
    r = float(series.iloc[-1] / series.iloc[-1-horizon] - 1.0)
    return _clip(r / (daily_vol * np.sqrt(horizon)))


def _pct_momentum(series, horizon, lookback=252):
    if len(series) <= horizon + 5:
        return 0.0
    r = float(series.iloc[-1] / series.iloc[-1-horizon] - 1.0)
    d = series.astype(float).pct_change(fill_method=None).tail(lookback).dropna()
    sd = float(d.std()) if len(d) else 0.0
    return _clip(r / (sd * np.sqrt(horizon))) if sd > 1e-12 else 0.0


def _change_score(series, horizon, lookback=252):
    if len(series) <= horizon + 5:
        return 0.0
    delta = float(series.iloc[-1] - series.iloc[-1-horizon])
    d = series.astype(float).diff().tail(lookback).dropna()
    sd = float(d.std()) if len(d) else 0.0
    return _clip(delta / (sd * np.sqrt(horizon))) if sd > 1e-12 else 0.0


def _level_z(series, lookback=756):
    x = series.astype(float).tail(lookback).dropna()
    if len(x) < 60:
        return 0.0
    sd = float(x.std())
    return _clip((float(x.iloc[-1]) - float(x.mean())) / sd) if sd > 1e-12 else 0.0


def build_signal(hist, params) -> pd.Series:
    """Asset-specific tactical attractiveness score."""
    assets = hist.assets
    prices = hist.prices.reindex(columns=assets)
    returns = hist.returns.reindex(columns=assets)
    macro = hist.macro

    fast = int(params["trend_fast"])
    mid = int(params["trend_mid"])
    slow = int(params["trend_slow"])
    vol_days = int(params["vol_days"])

    daily_vol = returns.tail(vol_days).std().replace(0.0, np.nan)

    trend = pd.Series(0.0, index=assets)
    for a in assets:
        if a == "SA_CASH":
            continue
        trend[a] = np.mean([
            _price_momentum(prices[a], fast, float(daily_vol.get(a, np.nan))),
            _price_momentum(prices[a], mid, float(daily_vol.get(a, np.nan))),
            _price_momentum(prices[a], slow, float(daily_vol.get(a, np.nan))),
        ])

    property_slow = np.mean([
        _price_momentum(prices["SA_PROPERTY"], mid, float(daily_vol["SA_PROPERTY"])),
        _price_momentum(prices["SA_PROPERTY"], slow, float(daily_vol["SA_PROPERTY"])),
    ])

    em = np.mean([_pct_momentum(macro["em_equity"], fast),
                  _pct_momentum(macro["em_equity"], mid)])
    fx = np.mean([_pct_momentum(macro["usdzar"], fast),
                  _pct_momentum(macro["usdzar"], mid)])
    dxy = np.mean([_pct_momentum(macro["dxy"], fast),
                   _pct_momentum(macro["dxy"], mid)])
    vix = _level_z(macro["vix"], 504)

    # Positive = supportive for South African / EM risk assets.
    risk_on = _clip((em - dxy - fx - vix) / 4.0)
    global_risk = _clip((-dxy - vix) / 2.0)

    sa_yield_move = np.mean([_change_score(macro["sa_10y"], fast),
                             _change_score(macro["sa_10y"], mid)])
    local_curve = _level_z(macro["sa_10y"] - macro["jibar_3m"], 756)
    rate_score = _clip((-sa_yield_move + local_curve) / 2.0)

    us_yield_move = np.mean([_change_score(macro["us_10y"], fast),
                             _change_score(macro["us_10y"], mid)])
    short_rate = _level_z(macro["jibar_3m"], 756)

    score = pd.Series(0.0, index=assets)
    score["SA_EQUITY"] = (trend["SA_EQUITY"] + risk_on) / 2.0
    score["GLOBAL_EQUITY"] = (trend["GLOBAL_EQUITY"] + fx + global_risk) / 3.0
    score["SA_BONDS"] = (trend["SA_BONDS"] + rate_score) / 2.0
    score["SA_CASH"] = (-risk_on + short_rate) / 2.0
    score["SA_PROPERTY"] = (property_slow + rate_score + trend["SA_EQUITY"]) / 3.0
    score["GOLD"] = (trend["GOLD"] + fx - us_yield_move) / 3.0

    return score.clip(-3.0, 3.0)


def _active_from_signal(signal: pd.Series, hist, params) -> pd.Series:
    """Convert scores to zero-sum active weights, with volatility used only for sizing."""
    assets = hist.assets
    score = signal.reindex(assets).astype(float).fillna(0.0)

    vol = hist.returns.tail(int(params["vol_days"])).std() * np.sqrt(252)
    risk_vol = vol.drop(labels=["SA_CASH"], errors="ignore").replace(0.0, np.nan)
    ref_vol = float(risk_vol.median()) if len(risk_vol.dropna()) else 1.0

    scale = pd.Series(1.0, index=assets)
    for a in assets:
        if a != "SA_CASH" and np.isfinite(vol.get(a, np.nan)) and vol[a] > 1e-12:
            scale[a] = np.sqrt(ref_vol / float(vol[a]))

    q = score * scale
    q = q - q.mean()
    total = float(q.abs().sum())
    if total <= 1e-12:
        return pd.Series(0.0, index=assets)

    return float(params["active_target"]) * q / total


def make_legal(weights: pd.Series, hist) -> pd.Series:
    """Force `weights` to satisfy every rule. You can leave this alone.

    Everything happens in active space -- how far each asset sits from its
    benchmark weight -- because that is how the rules are written.

    The loop is there because the steps interfere: forcing the active weights
    to net to zero (so the portfolio sums to 1) can push an asset back outside
    its band. A few passes settles it. The budget scaling goes last and is safe
    there: shrinking every active weight toward zero cannot breach a band, a
    cap, or non-negativity.
    """
    bm = hist.benchmark
    active = weights.reindex(hist.assets).astype(float) - bm

    for _ in range(50):
        active = active.clip(lower=-ACTIVE_BAND, upper=ACTIVE_BAND)  # rule 2
        active = active.clip(lower=-bm)                              # keeps weights >= 0
        # rule 4: total equity cap. Trim the equity block back, sharing the
        # cut over whichever equity assets still have room to come down.
        eq_excess = (bm[EQUITY] + active[EQUITY]).sum() - EQUITY_CAP
        eq_full = eq_excess > -1e-12
        if eq_excess > 0:
            floor = np.maximum(-ACTIVE_BAND, -bm[EQUITY])
            down = (active[EQUITY] - floor).clip(lower=0)
            if down.sum() > 1e-15:
                active[EQUITY] = active[EQUITY] - eq_excess * down / down.sum()

        active["GOLD"] = min(active["GOLD"], GOLD_CAP - bm["GOLD"])  # rule 5

        excess = active.sum()          # must be zero for weights to sum to 1
        if abs(excess) < 1e-12:
            break
        # give the correction to the assets that have room to absorb it
        room = (ACTIVE_BAND - active) if excess < 0 else (active + bm).clip(lower=0)
        room = room.clip(lower=0)
        if excess < 0 and eq_full:
            room[EQUITY] = 0.0   # equity is at its cap -- top up elsewhere
        if room.sum() <= 1e-15:
            break
        active = active - excess * room / room.sum()

    total = active.abs().sum()                                       # rule 3
    if total > ACTIVE_BUDGET:
        active = active * (ACTIVE_BUDGET / total)

    return bm + active


def generate_weights(hist, prev_weights, params):
    """Return the six portfolio weights to hold on hist.date."""
    bm = hist.benchmark

    if len(hist.returns) < int(params["trend_slow"]) + 10:
        return bm.to_dict()

    signal = build_signal(hist, params)
    target = make_legal(bm + _active_from_signal(signal, hist, params), hist)

    prev = prev_weights.reindex(hist.assets).astype(float)
    gap = target - prev

    costs = pd.Series({
        "SA_EQUITY": 15.0,
        "GLOBAL_EQUITY": 20.0,
        "SA_BONDS": 8.0,
        "SA_CASH": 1.0,
        "SA_PROPERTY": 35.0,
        "GOLD": 25.0,
    }).reindex(hist.assets)

    # Wider no-trade bands and slower trading for expensive assets.
    threshold = float(params["no_trade"]) * (0.5 + costs / costs.max())
    gap = gap.where(gap.abs() >= threshold, 0.0)

    speed = float(params["trade_speed"]) * np.sqrt(15.0 / costs)
    speed = speed.clip(lower=0.10, upper=0.60)

    weights = prev + speed * gap
    return make_legal(weights, hist).to_dict()


# <<--------------------- YOUR CODE GOES ABOVE THIS LINE --------------------->>
