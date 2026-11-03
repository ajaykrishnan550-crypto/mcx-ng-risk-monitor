"""Rule-based MCX Natural Gas option-seller risk engine.

Input option CSV columns: expiry,strike,option_type,ltp,oi,oi_change,volume,iv,delta,gamma,theta,vega
Input futures CSV columns: timestamp,open,high,low,close,volume,oi
All IV values may be supplied either as percentages (e.g. 42.5) or decimals (0.425).
This is a screening tool, not a forecast or a guarantee of safety.
"""
from __future__ import annotations
import pandas as pd
import numpy as np


def _num(s, default=0.0):
    return pd.to_numeric(s, errors="coerce").fillna(default)


def _normalise_iv(x):
    x = float(x) if pd.notna(x) else np.nan
    if pd.isna(x): return np.nan
    return x * 100 if 0 < x < 3 else x


def _wilder(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(alpha=1/period, adjust=False, min_periods=period).mean()


def add_indicators(fut: pd.DataFrame) -> pd.DataFrame:
    f = fut.copy()
    for c in ["open", "high", "low", "close", "volume", "oi"]:
        if c in f: f[c] = _num(f[c])
    f = f.sort_values("timestamp") if "timestamp" in f else f
    if not {"high", "low", "close"}.issubset(f.columns):
        return f
    prev = f.close.shift(1)
    tr = pd.concat([(f.high-f.low).abs(), (f.high-prev).abs(), (f.low-prev).abs()], axis=1).max(axis=1)
    f["atr14"] = _wilder(tr, 14)
    f["ema20"] = f.close.ewm(span=20, adjust=False).mean()
    f["ema50"] = f.close.ewm(span=50, adjust=False).mean()
    up = f.high.diff()
    down = -f.low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=f.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=f.index)
    atr = _wilder(tr, 14).replace(0, np.nan)
    plus_di = 100 * _wilder(plus_dm, 14) / atr
    minus_di = 100 * _wilder(minus_dm, 14) / atr
    dx = 100 * (plus_di-minus_di).abs() / (plus_di+minus_di).replace(0, np.nan)
    f["adx14"] = _wilder(dx.fillna(0), 14)
    f["plus_di"] = plus_di
    f["minus_di"] = minus_di
    f["volume_avg20"] = f.volume.rolling(20, min_periods=5).mean() if "volume" in f else np.nan
    f["oi_change_pct"] = f.oi.pct_change() * 100 if "oi" in f else np.nan
    f["atr_pct"] = f.atr14 / f.close * 100
    return f


def analyse(options: pd.DataFrame, futures: pd.DataFrame, prior: dict | None = None) -> dict:
    prior = prior or {}
    o = options.copy()
    required = {"strike", "option_type", "ltp", "oi"}
    missing = required - set(o.columns)
    if missing: raise ValueError("Options CSV missing columns: " + ", ".join(sorted(missing)))
    for c in ["strike", "ltp", "oi", "oi_change", "volume", "iv", "delta", "gamma", "theta", "vega"]:
        if c not in o: o[c] = np.nan
        o[c] = pd.to_numeric(o[c], errors="coerce")
    o["option_type"] = o.option_type.astype(str).str.upper().str.strip().replace({"CALL":"CE", "PUT":"PE"})
    if "expiry" in o and o.expiry.notna().any():
        exp = o.expiry.dropna().astype(str).iloc[0]
        o = o[o.expiry.astype(str) == exp]
    ce = o[o.option_type.eq("CE")]
    pe = o[o.option_type.eq("PE")]
    pcr = pe.oi.sum() / ce.oi.sum() if ce.oi.sum() else np.nan
    f = add_indicators(futures) if len(futures) else futures.copy()
    last = f.iloc[-1] if len(f) else pd.Series(dtype=float)
    price = float(last.get("close", np.nan))
    strike = float(o.loc[(o.strike-price).abs().idxmin(), "strike"]) if pd.notna(price) and len(o) else np.nan
    atm = o[o.strike.eq(strike)] if pd.notna(strike) else o.iloc[0:0]
    call_atm = atm[atm.option_type.eq("CE")].ltp.sum()
    put_atm = atm[atm.option_type.eq("PE")].ltp.sum()
    straddle = float(call_atm + put_atm) if len(atm) else np.nan
    atm_iv = float(atm.iv.dropna().mean()) if atm.iv.notna().any() else np.nan
    atm_iv = _normalise_iv(atm_iv)
    prev_iv = prior.get("atm_iv")
    prev_straddle = prior.get("straddle")
    iv_change = ((atm_iv / prev_iv)-1)*100 if prev_iv and atm_iv else np.nan
    straddle_change = ((straddle / prev_straddle)-1)*100 if prev_straddle and straddle else np.nan
    adx = float(last.get("adx14", np.nan))
    atr_pct = float(last.get("atr_pct", np.nan))
    vol_ratio = float(last.get("volume", np.nan) / last.get("volume_avg20", np.nan)) if pd.notna(last.get("volume_avg20", np.nan)) and last.get("volume_avg20", 0) else np.nan
    oi_change = float(last.get("oi_change_pct", np.nan))
    prev_close = float(f.close.iloc[-2]) if len(f) > 1 else np.nan
    price_change = ((price/prev_close)-1)*100 if pd.notna(prev_close) and prev_close else np.nan
    ema20 = float(last.get("ema20", np.nan)); ema50=float(last.get("ema50",np.nan))
    trend = "UP" if pd.notna(ema20) and pd.notna(ema50) and price > ema20 > ema50 else "DOWN" if pd.notna(ema20) and pd.notna(ema50) and price < ema20 < ema50 else "MIXED"

    # Overall warning score: transparent heuristic, capped at 100.
    components = {}
    components["IV expansion"] = 20 if pd.notna(iv_change) and iv_change >= 10 else 12 if pd.notna(iv_change) and iv_change >= 5 else 0
    components["Straddle expansion"] = 20 if pd.notna(straddle_change) and straddle_change >= 15 else 12 if pd.notna(straddle_change) and straddle_change >= 8 else 0
    components["Futures momentum + OI"] = 20 if pd.notna(price_change) and pd.notna(oi_change) and abs(price_change) >= 0.8 and oi_change >= 5 else 10 if pd.notna(price_change) and abs(price_change) >= 0.8 else 0
    components["Volume surge"] = 15 if pd.notna(vol_ratio) and vol_ratio >= 2 else 8 if pd.notna(vol_ratio) and vol_ratio >= 1.5 else 0
    components["PCR shift"] = 0  # Requires at least two time-stamped chain snapshots; never infer from a single PCR.
    components["ADX trend"] = 10 if pd.notna(adx) and adx >= 25 else 5 if pd.notna(adx) and adx >= 20 else 0
    components["Volatility regime"] = 5 if pd.notna(atr_pct) and atr_pct >= 2.5 else 0
    overall = min(100, int(sum(components.values())))

    # Side scores are directional/volatility warnings for short option exposure.
    put = 0; call = 0
    if trend == "DOWN": put += 20
    if trend == "UP": call += 20
    if pd.notna(price_change):
        if price_change <= -0.8: put += 15
        if price_change >= 0.8: call += 15
    if pd.notna(last.get("minus_di", np.nan)) and pd.notna(last.get("plus_di", np.nan)):
        if last.minus_di > last.plus_di: put += 10
        if last.plus_di > last.minus_di: call += 10
    if pd.notna(iv_change) and iv_change >= 5: put += 10; call += 10
    if pd.notna(straddle_change) and straddle_change >= 8: put += 10; call += 10
    if pd.notna(adx) and adx >= 25:
        if trend == "DOWN": put += 10
        elif trend == "UP": call += 10
    if pd.notna(vol_ratio) and vol_ratio >= 1.5 and pd.notna(price_change):
        if price_change < 0: put += 10
        elif price_change > 0: call += 10
    put = min(100, put); call = min(100, call)
    def label(s): return "EXTREME" if s >= 75 else "HIGH" if s >= 55 else "CAUTION" if s >= 30 else "LOW"
    warnings = []
    if pd.notna(iv_change) and iv_change >= 5: warnings.append("ATM IV is expanding versus the previous snapshot")
    if pd.notna(straddle_change) and straddle_change >= 8: warnings.append("ATM straddle premium is expanding")
    if trend == "DOWN": warnings.append("Futures trend is bearish; short puts face directional risk")
    if trend == "UP": warnings.append("Futures trend is bullish; short calls face directional risk")
    if pd.notna(vol_ratio) and vol_ratio >= 1.5: warnings.append("Volume is elevated versus the 20-bar average")
    if pd.notna(adx) and adx >= 25: warnings.append("ADX indicates a stronger directional trend")
    if not warnings: warnings.append("No configured warning trigger fired; this does not mean the trade is safe")
    return {
        "overall_score": overall, "overall_label": label(overall),
        "put_score": put, "put_label": label(put), "call_score": call, "call_label": label(call),
        "pcr": float(pcr) if pd.notna(pcr) else None, "futures_price": price,
        "price_change_pct": price_change, "atm_strike": strike, "atm_straddle": straddle,
        "atm_iv": atm_iv, "iv_change_pct": iv_change, "straddle_change_pct": straddle_change,
        "adx14": adx, "atr_pct": atr_pct, "volume_ratio": vol_ratio, "futures_oi_change_pct": oi_change,
        "trend": trend, "components": components, "warnings": warnings,
        "snapshot": {"atm_iv": atm_iv, "straddle": straddle}
    }
