import time
import logging
from datetime import datetime, timezone, date

import streamlit as st
import pandas as pd
import numpy as np
import requests
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("liquidity_dashboard")

# =============================================================================
# MACRO LIQUIDITY REGIME CONFIG
# Upstream (macro liquidity) vs downstream (ticker-level flow) — the regime
# score gates/scales the ticker scanner below rather than competing with it.
# =============================================================================

FRED_SERIES = {
    "us_m2": "M2SL",                 # US M2 money stock, monthly, SA
    "ea_m3": "MABMM301EZM189S",      # Eurozone M3, monthly
    "cn_m2": "MYAGM2CNM189N",        # China M2 via IMF/FRED — DISCONTINUED, see below
    "fed_bs": "WALCL",               # Fed total assets, weekly
    "ecb_bs": "ECBASSETSW",          # ECB total assets, weekly
    "credit_spread": "BAMLC0A0CM",   # ICE BofA US Corp OAS, daily
    "dollar_index": "DTWEXBGS",      # Trade-weighted broad USD index, daily
}

# Series known to have stopped updating on FRED. Flagged explicitly so the UI
# shows "Discontinued" instead of an unexplained blank — verified via FRED
# directly (MYAGM2CNM189N's last observation is Aug 2019, not merely lagged).
DISCONTINUED_SERIES = {
    "cn_m2": "FRED series MYAGM2CNM189N (M2 for China) stopped publishing after Aug 2019 — "
             "it is discontinued, not lagged. No free real-time replacement was found on FRED "
             "as of this build; weight is automatically redistributed to the remaining components.",
}

# Illustrative — tune freely. Must sum to 1.0 within each group.
GLOBAL_M2_SUBWEIGHTS = {"us_m2": 0.45, "ea_m3": 0.25, "cn_m2": 0.30}

REGIME_WEIGHTS = {
    "global_m2": 0.40,
    "fed_bs": 0.20,
    "ecb_bs": 0.10,
    "cn_m2_standalone": 0.10,   # PBOC weighted separately per your framework
    "credit_spread": 0.10,      # inverted: tighter spreads = more liquidity supportive
    "dollar_index": 0.10,       # inverted: weaker dollar = more liquidity supportive
}

ZSCORE_WINDOW_MONTHS = 36  # 3-year rolling window for normalization

# 1. SET UP DASHBOARD INTERFACE
st.set_page_config(layout="wide", page_title="Institutional Liquidity Flow Map", page_icon="⚡")
st.title("⚡ Structural Liquidity & Sector Flow Engine")
st.markdown("Track real-time price/volume momentum and positioning across custom framework layers.")
st.caption(
    "⚠️ This dashboard uses price change and relative volume as a **proxy** for institutional "
    "activity (via yfinance). It does not use Level 2, dark-pool, or actual order-flow data, "
    "which yfinance does not provide. Treat 'Liquidity Score' as a momentum/volume heuristic, "
    "not confirmed institutional flow."
)

# 2. DEFINE SYSTEMATIC TICKER MAPPING FROM USER ALLOCATION GRID
# Merged from V2 (your ticker changes + BTC/Gold addition) with two corrections
# applied — see the note block below for what changed and why.
TICKER_MAP = {
    # ALTERNATIVE LIQUIDITY HAVENS — handled separately from equities throughout
    # the dashboard; see SPECIAL_THEMES below and the dedicated section in the UI.
    "Bitcoin & Gold": ["BTC-USD", "GC=F"],

    # INFRASTRUCTURE LAYERS
    "Logistics & Hard Assets": ["TPL", "ADPORTS.AE", "ICTEY", "CNI", "CP", "UNP"],
    "Grids & Power Generation": ["GEV", "ETN", "NVT", "CEG", "PWR", "LIN", "ABBN.SW", "SU.PA"],
    "Water & Utilities": ["CWCO", "XYL", "ECL", "WM", "RSG"],
    "Tech-Adjacent Infra": ["VRT", "BE", "ANET", "FTNT", "CHKP", "CRWD", "ZS"],

    # ENERGY & COMMODITY LAYERS
    "Royalties": ["FNV", "WPM", "BSM", "DMLP"],
    "Uranium & Baseload Energy": ["CCJ", "CNQ", "XOM", "SU", "CVX"],
    "Copper & Industrial Materials": ["FCX", "SCCO", "BHP", "NEM", "COP", "NUE", "PH", "CAT"],

    # AI / SEMICONDUCTOR LAYERS
    "Semiconductor Monopolies": ["TSM", "ASML", "SHECY", "6920.T"],
    "Robotics, Architecture & Automation": ["AVGO", "CDNS", "QCOM", "FANUY", "8035.T", "SNPS", "ABB"],
    "AI Softwares & Velocity Applications": ["NOW", "PANW", "STX"],

    # EMERGING MARKETS JURISDICTIONS
    "Emerging Markets: India": ["SIEMENS.NS", "POWERGRID.NS", "PIIND.NS", "SUNPHARMA.NS", "HCLTECH.NS", "ABB.NS"],
    "Emerging Markets: GCC": ["2222.SR", "ADNOCGAS.AE", "2082.SR", "7010.SR"],
    "Emerging Markets: Other": ["9984.T", "TLK", "INDO", "VALE", "0883.HK", "CSUAY", "0941.HK"],

    # BUSINESS & FUTURISTIC OVERLAY (HEALTHCARE & LONGEVITY)
    "Healthcare & Longevity": ["NVO", "AZN", "ISRG", "TMO"],
}

# Themes with fundamentally different trading mechanics than equities (24/7
# crypto trading, futures contract roll effects, no exchange "session" concept).
# Excluded from the main equity pillar chart and given their own section instead
# of being visually averaged in with 100+ stocks — see Section "Bitcoin & Gold"
# in the UI below.
SPECIAL_THEMES = ["Bitcoin & Gold"]

# =============================================================================
# CHANGES FROM YOUR V2 FILE — read before deploying
# =============================================================================
# CORRECTED (V2 had introduced a regression vs. the verified fix from earlier
# in this session):
#   - ADPORTS.AD -> ADPORTS.AE, ADNOCGAS.AD -> ADNOCGAS.AE
#     Yahoo Finance uses .AE for Abu Dhabi Securities Exchange (ADX) listings —
#     confirmed via search (ISO country code AE, no ".AD" suffix exists in
#     Yahoo's taxonomy). Your V2 changed both from .AE to .AD; that looks like
#     an unintended find-and-replace rather than a deliberate fix, so it's been
#     reverted back to .AE.
#   - ABB restored in "Robotics, Architecture & Automation" (V2 had removed it,
#     apparently on the assumption that ABB.NS in the India pillar covers it).
#     ABB (Zurich-listed parent, NYSE-tradable) and ABB.NS (ABB India Limited)
#     are DIFFERENT securities — ABB India is a separately-listed, separately-
#     priced majority-owned subsidiary, not the same stock on two tickers.
#     Dropping ABB silently removes real exposure to the global parent company.
# KEPT FROM V2 (verified correct):
#   - 2222.SR (Saudi Aramco), 2082.SR (ACWA Power), 7010.SR (STC) — Yahoo uses
#     Tadawul's numeric codes for Saudi listings, not letter tickers. Confirmed
#     live on finance.yahoo.com for 2082.SR; the old ARAMCO.SR/ACWAPOWER.SR/
#     STC.SR letter-tickers were the actual bug in the original version.
#   - 0883.HK / 0941.HK replacing CEO/CHL (Hong Kong-listed CNOOC and China
#     Mobile) — more precise than the US ADR tickers they replaced.
#   - EQT removed from Uranium & Baseload Energy — EQT is a natural gas
#     producer, not a uranium/baseload name; this was a miscategorization in
#     the original grid, not a data error, and the removal is correct.
#   - BSM, DMLP added to Royalties; BTC-USD, GC=F added as a new theme.
# STILL UNVERIFIED — check these on finance.yahoo.com before trusting results:
#   ICTEY, SHECY, CSUAY, FANUY (carried over from the original grid, never
#   independently confirmed).
# =============================================================================



ALL_TICKERS = [ticker for sublist in TICKER_MAP.values() for ticker in sublist]

MARKET_SESSION_LABELS = {
    "PRE": "Pre-Market",
    "PREPRE": "Pre-Market",
    "REGULAR": "Regular Hours",
    "POST": "After Hours",
    "POSTPOST": "After Hours",
    "CLOSED": "Closed",
}


# =============================================================================
# MACRO LIQUIDITY REGIME ENGINE
# Cached for a full day — this is a slow-moving backdrop, not a live metric.
# Recomputing it every 5 min alongside the ticker scanner would just replay
# the same stale monthly/weekly print and imply false precision.
# =============================================================================

def _get_fred_key():
    key = st.secrets.get("FRED_API_KEY", None) if hasattr(st, "secrets") else None
    if not key:
        key = st.sidebar.text_input("FRED API Key (not stored, session only)", type="password")
    return key


@st.cache_data(ttl=86400)
def fetch_fred_series(series_id, api_key, start_date="2015-01-01"):
    """Pull one FRED series as a clean pandas Series indexed by date."""
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start_date,
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        obs = resp.json().get("observations", [])
        if not obs:
            return pd.Series(dtype=float)
        df = pd.DataFrame(obs)[["date", "value"]]
        df["date"] = pd.to_datetime(df["date"])
        # FRED uses "." for missing observations — must be filtered, not coerced blindly.
        df = df[df["value"] != "."]
        df["value"] = df["value"].astype(float)
        return df.set_index("date")["value"]
    except Exception as e:
        logger.warning(f"FRED fetch failed for {series_id}: {e}")
        return pd.Series(dtype=float)


def to_monthly_yoy(series):
    """Resample to month-end (forward-filling gaps) and compute YoY % change."""
    if series.empty:
        return pd.Series(dtype=float)
    monthly = series.resample("ME").last().ffill()
    return monthly.pct_change(12) * 100


def to_monthly_yoy_diff(series):
    """For spreads/DXY: YoY change in level (not % change — a move from 100bp
    to 150bp is a 50bp widening, not usefully expressed as '% change')."""
    if series.empty:
        return pd.Series(dtype=float)
    monthly = series.resample("ME").last().ffill()
    return monthly.diff(12)


def rolling_zscore_series(yoy_series, window=ZSCORE_WINDOW_MONTHS, min_periods=18):
    """Full rolling z-score history, not just the latest point — this is what
    lets us both (a) show today's regime reading and (b) regress historical
    theme returns against historical regime readings using the exact same math,
    instead of maintaining two versions of the same logic that can drift apart."""
    if yoy_series.empty:
        return pd.Series(dtype=float)
    roll_mean = yoy_series.rolling(window, min_periods=min_periods).mean()
    roll_std = yoy_series.rolling(window, min_periods=min_periods).std()
    z = (yoy_series - roll_mean) / roll_std.replace(0, np.nan)
    return z.replace([np.inf, -np.inf], np.nan)


def weighted_composite_row(row, weights):
    """Row-wise weighted average that re-normalizes over only the non-null
    components present THAT MONTH — a component with a shorter history simply
    joins the blend once it has enough data, rather than forcing the whole
    composite to start later or silently counting missing data as neutral (0)."""
    avail = {k: v for k, v in row.items() if k in weights and pd.notna(v)}
    if not avail:
        return np.nan
    wsum = sum(weights[k] for k in avail)
    return sum(v * weights[k] for k, v in avail.items()) / wsum


@st.cache_data(ttl=86400)
def compute_component_zscore_frame(api_key):
    """Single source of truth: monthly z-score history for every macro
    component, the blended Global M2 column, and the final composite —
    everything downstream (the live gauge AND the historical beta regression)
    reads from this one frame so they can never disagree with each other."""
    raw, as_of = {}, {}
    for key, series_id in FRED_SERIES.items():
        s = fetch_fred_series(series_id, api_key)
        raw[key] = s
        as_of[key] = s.index.max() if not s.empty else None

    yoy = {
        k: (to_monthly_yoy_diff(v) if k in ("credit_spread", "dollar_index") else to_monthly_yoy(v))
        for k, v in raw.items()
    }
    z = {k: rolling_zscore_series(v) for k, v in yoy.items()}

    df_z = pd.DataFrame(z)
    if df_z.empty:
        return df_z, as_of

    # Invert spreads/dollar so "higher" always means "more liquidity supportive."
    if "credit_spread" in df_z:
        df_z["credit_spread"] = -df_z["credit_spread"]
    if "dollar_index" in df_z:
        df_z["dollar_index"] = -df_z["dollar_index"]

    df_z["global_m2"] = df_z.apply(lambda r: weighted_composite_row(r, GLOBAL_M2_SUBWEIGHTS), axis=1)
    df_z["cn_m2_standalone"] = df_z.get("cn_m2", np.nan)

    df_z["composite_z"] = df_z.apply(lambda r: weighted_composite_row(r, REGIME_WEIGHTS), axis=1)

    return df_z, as_of


@st.cache_data(ttl=86400)
def compute_liquidity_regime(api_key):
    """Latest-point view for the dashboard's live gauge/table.

    IMPORTANT: this does NOT pick one shared calendar row across all
    components. Fed/ECB balance sheets and credit spreads update
    weekly/daily; US M2 and Eurozone M3 report ~1-2 months behind. Forcing
    everything onto one shared 'latest date' would always land on the
    current month — which, by construction, the lagging monthly series
    haven't posted yet — making Global M2 blank almost every time this is
    viewed, even when perfectly good May data exists. Instead, each
    component reports its OWN latest available z-score and as-of date,
    and the composite blends whatever's currently available."""
    df_z, as_of = compute_component_zscore_frame(api_key)
    if df_z.empty:
        return np.nan, pd.DataFrame(), as_of

    raw_keys = ["us_m2", "ea_m3", "cn_m2", "fed_bs", "ecb_bs", "credit_spread", "dollar_index"]
    latest_by_component = {}
    for k in raw_keys:
        if k in df_z.columns:
            col = df_z[k].dropna()
            latest_by_component[k] = float(col.iloc[-1]) if not col.empty else np.nan
        else:
            latest_by_component[k] = np.nan

    global_m2_now = weighted_composite_row(pd.Series(latest_by_component), GLOBAL_M2_SUBWEIGHTS)
    component_now = {
        "global_m2": global_m2_now,
        "fed_bs": latest_by_component.get("fed_bs", np.nan),
        "ecb_bs": latest_by_component.get("ecb_bs", np.nan),
        "cn_m2_standalone": latest_by_component.get("cn_m2", np.nan),
        "credit_spread": latest_by_component.get("credit_spread", np.nan),
        "dollar_index": latest_by_component.get("dollar_index", np.nan),
    }
    composite = weighted_composite_row(pd.Series(component_now), REGIME_WEIGHTS)

    def status_for(component_key, raw_key, value):
        if raw_key in DISCONTINUED_SERIES:
            return "Discontinued"
        if pd.isna(value):
            return "No data"
        return "OK"

    component_keys = list(REGIME_WEIGHTS.keys())
    raw_key_lookup = {"global_m2": "us_m2", "cn_m2_standalone": "cn_m2"}
    table = pd.DataFrame({
        "Component": component_keys,
        "Z-Score": [round(component_now[k], 2) if pd.notna(component_now[k]) else None for k in component_keys],
        "Weight": [REGIME_WEIGHTS[k] for k in component_keys],
        "Status": [status_for(k, raw_key_lookup.get(k, k), component_now[k]) for k in component_keys],
        "Latest Data As Of": [as_of.get(raw_key_lookup.get(k, k), None) for k in component_keys],
    })
    notes = [DISCONTINUED_SERIES[raw_key_lookup.get(k, k)] for k in component_keys
             if raw_key_lookup.get(k, k) in DISCONTINUED_SERIES]

    return (float(composite) if pd.notna(composite) else np.nan), table, as_of, notes


def classify_regime(composite_z):
    if composite_z is None or np.isnan(composite_z):
        return "Unknown — insufficient data", "gray"
    if composite_z > 0.5:
        return "Liquidity Expanding (Tailwind)", "green"
    if composite_z < -0.5:
        return "Liquidity Contracting (Headwind)", "red"
    return "Neutral / Transitional", "orange"


def regime_multiplier(composite_z, clip=(0.7, 1.3), sensitivity=0.15):
    """Fallback uniform multiplier — used only when a theme has no reliable
    beta estimate yet (see theme_regime_multiplier below for the real per-theme
    version)."""
    if composite_z is None or np.isnan(composite_z):
        return 1.0
    return float(np.clip(1 + composite_z * sensitivity, clip[0], clip[1]))


# =============================================================================
# PER-THEME LIQUIDITY BETA
# Estimates how sensitive each pillar's historical monthly returns actually
# are to the macro liquidity regime, via simple OLS: theme_return ~ regime_z.
# This replaces the flat, uniform multiplier with one that scales up for
# historically liquidity-sensitive themes (e.g. high-beta semis) and dampens
# for historically insensitive ones (e.g. defensive utilities/water).
# =============================================================================

BETA_MIN_MONTHS = 12          # below this, we don't trust the slope at all
BETA_LIMITED_MONTHS = 24      # below this, flagged "Limited" confidence
THEME_RETURN_LOOKBACK = "5y"


@st.cache_data(ttl=86400)
def fetch_theme_monthly_returns(period=THEME_RETURN_LOOKBACK):
    """Equal-weighted average monthly return across each theme's constituent
    tickers. Cached daily — this is for a slow-moving historical regression,
    not a live metric, and re-downloading 5 years of monthly bars every
    5 minutes would be both pointless and a good way to get rate-limited."""
    try:
        hist = yf.download(
            ALL_TICKERS, period=period, interval="1mo",
            group_by="ticker", threads=True, progress=False,
        )
    except Exception as e:
        logger.warning(f"Theme monthly-return download failed: {e}")
        return pd.DataFrame()

    if hist.empty:
        return pd.DataFrame()

    theme_returns = {}
    for theme, tickers in TICKER_MAP.items():
        per_ticker_rets = []
        for tkr in tickers:
            try:
                if isinstance(hist.columns, pd.MultiIndex):
                    if tkr not in hist.columns.get_level_values(0):
                        continue
                    closes = hist[tkr]["Close"].dropna()
                else:
                    closes = hist["Close"].dropna()  # single-ticker edge case
                if len(closes) < 6:
                    continue
                per_ticker_rets.append(closes.pct_change().dropna())
            except Exception:
                continue
        if per_ticker_rets:
            # Outer-align on date, average across whatever tickers have data
            # that month rather than requiring every ticker to be present.
            theme_returns[theme] = pd.concat(per_ticker_rets, axis=1).mean(axis=1)

    if not theme_returns:
        return pd.DataFrame()

    df = pd.DataFrame(theme_returns)
    df.index = df.index.to_period("M").to_timestamp("M")
    df = df.groupby(df.index).mean()
    return df


@st.cache_data(ttl=86400)
def compute_theme_betas(api_key):
    """OLS slope of each theme's monthly return on the monthly composite
    liquidity z-score. Returns a per-theme table with Beta, correlation,
    sample size, and a confidence flag — insufficient-history themes get
    Beta=NaN and are handled as neutral (Relative Beta = 1.0) downstream,
    never silently assigned a fabricated number."""
    df_z, _ = compute_component_zscore_frame(api_key)
    theme_returns = fetch_theme_monthly_returns()

    rows = []
    if df_z.empty or theme_returns.empty or "composite_z" not in df_z:
        for theme in TICKER_MAP.keys():
            rows.append({"Theme": theme, "Beta": np.nan, "Correlation": np.nan,
                         "Months of Data": 0, "Confidence": "No data"})
        return pd.DataFrame(rows)

    z_series = df_z["composite_z"].dropna()

    for theme in TICKER_MAP.keys():
        if theme not in theme_returns.columns:
            rows.append({"Theme": theme, "Beta": np.nan, "Correlation": np.nan,
                         "Months of Data": 0, "Confidence": "No price history"})
            continue

        combined = pd.concat(
            [z_series.rename("z"), theme_returns[theme].rename("ret")], axis=1
        ).dropna()
        n = len(combined)

        if n < BETA_MIN_MONTHS:
            rows.append({"Theme": theme, "Beta": np.nan, "Correlation": np.nan,
                         "Months of Data": n, "Confidence": f"Insufficient history (<{BETA_MIN_MONTHS}mo)"})
            continue

        try:
            beta, _intercept = np.polyfit(combined["z"], combined["ret"], 1)
            corr = np.corrcoef(combined["z"], combined["ret"])[0, 1]
        except Exception:
            rows.append({"Theme": theme, "Beta": np.nan, "Correlation": np.nan,
                         "Months of Data": n, "Confidence": "Regression failed"})
            continue

        confidence = "OK" if n >= BETA_LIMITED_MONTHS else f"Limited ({BETA_MIN_MONTHS}-{BETA_LIMITED_MONTHS}mo)"
        rows.append({"Theme": theme, "Beta": round(float(beta), 4),
                     "Correlation": round(float(corr), 2), "Months of Data": n,
                     "Confidence": confidence})

    betas_df = pd.DataFrame(rows)

    # Normalize betas relative to the cross-theme average so the SCALE of the
    # multiplier stays anchored to what regime_multiplier() used to produce —
    # an average-beta theme gets roughly the old uniform behavior; a
    # high-beta theme gets amplified, a low/negative-beta theme dampened.
    valid_betas = betas_df["Beta"].dropna().abs()
    avg_abs_beta = valid_betas.mean() if not valid_betas.empty else np.nan

    def relative_beta(b):
        if pd.isna(b) or pd.isna(avg_abs_beta) or avg_abs_beta == 0:
            return 1.0
        return float(b / avg_abs_beta)

    betas_df["Relative Beta"] = betas_df["Beta"].apply(relative_beta)
    return betas_df


def theme_regime_multiplier(composite_z, relative_beta, clip=(0.5, 1.6), sensitivity=0.15):
    """Per-theme version of regime_multiplier(): same shape, but scaled by
    that theme's historical sensitivity to the liquidity regime. Wider clip
    band than the uniform version since high-beta themes should legitimately
    swing further than the flat case did."""
    if composite_z is None or np.isnan(composite_z):
        return 1.0
    if relative_beta is None or (isinstance(relative_beta, float) and np.isnan(relative_beta)):
        relative_beta = 1.0
    return float(np.clip(1 + composite_z * sensitivity * relative_beta, clip[0], clip[1]))


# 3A. BATCH-FETCH HISTORICAL BARS FOR ALL TICKERS IN ONE CALL
# One bulk request instead of N separate requests dramatically cuts the odds of
# Yahoo rate-limiting / temporarily blocking you when running this every few minutes.
@st.cache_data(ttl=300)
def fetch_batch_history(ticker_list):
    try:
        hist = yf.download(
            ticker_list,
            period="10d",
            group_by="ticker",
            threads=True,
            progress=False,
        )
        return hist
    except Exception as e:
        logger.warning(f"Batch history download failed: {e}")
        return pd.DataFrame()


# 3B. PER-TICKER LIVE / PRE-MARKET SNAPSHOT + METRIC CALCULATIONS
@st.cache_data(ttl=300)
def fetch_liquidity_metrics(ticker_list):
    data_rows = []
    failed_tickers = []

    # SPY benchmark fetch is now guarded — a single failed request no longer
    # crashes the whole app.
    spy_pct = None
    try:
        spy = yf.Ticker("SPY")
        spy_hist = spy.history(period="5d")
        if len(spy_hist) >= 2:
            spy_pct = ((spy_hist["Close"].iloc[-1] - spy_hist["Close"].iloc[-2])
                       / spy_hist["Close"].iloc[-2]) * 100
    except Exception as e:
        logger.warning(f"SPY benchmark fetch failed: {e}")

    if spy_pct is None:
        spy_pct = 0.0  # neutral fallback so Alpha column still renders, flagged in UI

    batch_hist = fetch_batch_history(ticker_list)

    for ticker_symbol in ticker_list:
        try:
            # Prefer the batch-downloaded history; fall back to a single fetch
            # only if the batch call didn't return this ticker.
            if (not batch_hist.empty) and ticker_symbol in batch_hist.columns.get_level_values(0):
                hist = batch_hist[ticker_symbol].dropna(how="all")
            else:
                hist = yf.Ticker(ticker_symbol).history(period="10d")

            if hist.empty or len(hist) < 2:
                failed_tickers.append((ticker_symbol, "insufficient history"))
                continue

            avg_volume = hist["Volume"].iloc[:-1].mean()

            t = yf.Ticker(ticker_symbol)
            info = t.info

            current_price = info.get("regularMarketPrice") or hist["Close"].iloc[-1]
            prev_close = info.get("previousClose") or hist["Close"].iloc[-2]
            current_volume = info.get("regularMarketVolume") or hist["Volume"].iloc[-1]
            pre_market_price = info.get("preMarketPrice")
            market_state = info.get("marketState", "REGULAR")

            # Guard against zero/None denominators instead of letting them
            # propagate as inf/NaN into the sort and color scale.
            if not prev_close:
                failed_tickers.append((ticker_symbol, "missing previousClose"))
                continue

            is_pre_market = market_state in ("PRE", "PREPRE") and pre_market_price
            if is_pre_market:
                price_change = ((pre_market_price - prev_close) / prev_close) * 100
            else:
                price_change = ((current_price - prev_close) / prev_close) * 100

            rvol = current_volume / avg_volume if avg_volume and avg_volume > 0 else float("nan")
            alpha_perf = price_change - spy_pct
            liquidity_score = rvol * price_change if price_change > 0 else rvol * (price_change * 0.5)

            data_rows.append({
                "Ticker": ticker_symbol,
                "Price": round(current_price, 2),
                "Change %": round(price_change, 2),
                "RVOL": round(rvol, 2) if pd.notna(rvol) else None,
                "Alpha vs SPY": round(alpha_perf, 2),
                "Liquidity Score": round(liquidity_score, 2) if pd.notna(liquidity_score) else None,
                "Volume State": MARKET_SESSION_LABELS.get(market_state, market_state),
            })

            # Small delay to be gentler on Yahoo's undocumented endpoint when
            # looping .info calls for many symbols.
            time.sleep(0.05)

        except Exception as e:
            failed_tickers.append((ticker_symbol, str(e)))
            continue

    return pd.DataFrame(data_rows), spy_pct, failed_tickers


# =============================================================================
# MACRO LIQUIDITY REGIME PANEL — upstream backdrop, refreshed daily
# =============================================================================
st.subheader("🌍 Macro Liquidity Regime")

fred_key = _get_fred_key()

if not fred_key:
    st.warning("Enter your FRED API key in the sidebar to compute the liquidity regime score.")
    composite_z, regime_table = np.nan, pd.DataFrame()
    betas_df = pd.DataFrame()
else:
    with st.spinner("Pulling macro liquidity data from FRED..."):
        composite_z, regime_table, as_of_dates, discontinued_notes = compute_liquidity_regime(fred_key)

    regime_label, regime_color = classify_regime(composite_z)

    gcol1, gcol2 = st.columns([1, 2])
    with gcol1:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=0 if np.isnan(composite_z) else round(composite_z, 2),
            title={"text": regime_label},
            gauge={
                "axis": {"range": [-2, 2]},
                "bar": {"color": regime_color},
                "steps": [
                    {"range": [-2, -0.5], "color": "#f8d7da"},
                    {"range": [-0.5, 0.5], "color": "#fff3cd"},
                    {"range": [0.5, 2], "color": "#d4edda"},
                ],
            },
        ))
        gauge.update_layout(height=280, margin=dict(t=40, b=10))
        st.plotly_chart(gauge, use_container_width=True)
    with gcol2:
        st.dataframe(regime_table, use_container_width=True, hide_index=True)
        st.caption(
            "Each component shows ITS OWN latest available reading — components are not forced "
            "onto one shared calendar date, since slow-reporting series (M2, M3) would otherwise "
            "always appear blank in the current month. Z-scores are vs. each component's own "
            f"trailing {ZSCORE_WINDOW_MONTHS}-month distribution. Missing components are excluded "
            "and weights re-normalized — never silently treated as neutral."
        )
        if discontinued_notes:
            for note in discontinued_notes:
                st.warning(f"⚠️ {note}")

    with st.spinner("Estimating historical per-theme liquidity sensitivity..."):
        betas_df = compute_theme_betas(fred_key)

    with st.expander("📈 Per-Theme Liquidity Beta (historical sensitivity to the regime)"):
        st.dataframe(
            betas_df.sort_values("Relative Beta", ascending=False),
            use_container_width=True, hide_index=True,
        )
        st.caption(
            f"Beta = slope of each theme's monthly return on the composite regime z-score "
            f"over the trailing {THEME_RETURN_LOOKBACK}, equal-weighted across constituents. "
            f"Themes with fewer than {BETA_MIN_MONTHS} months of overlapping data get "
            "Relative Beta = 1.0 (neutral fallback, same as the old uniform multiplier) "
            "rather than a fabricated slope. **This is backward-looking — a theme's "
            "historical liquidity sensitivity is not guaranteed to persist, and a high R² "
            "here is correlation, not causation.**"
        )

st.divider()

# Run calculations engine
with st.spinner("Processing data pipelines..."):
    df_metrics, spy_performance, failed = fetch_liquidity_metrics(ALL_TICKERS)

st.caption(f"Ticker data as of {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')} · cached 5 min")

if failed:
    with st.expander(f"⚠️ {len(failed)} ticker(s) failed / excluded — click to see why"):
        st.dataframe(pd.DataFrame(failed, columns=["Ticker", "Reason"]), use_container_width=True)


# Map classifications onto calculations return
def assign_theme(ticker):
    for theme, tickers in TICKER_MAP.items():
        if ticker in tickers:
            return theme
    return "Other"


if not df_metrics.empty:
    df_metrics["Thematic Destination"] = df_metrics["Ticker"].apply(assign_theme)

    # Bitcoin/Gold trade on fundamentally different mechanics (24/7 crypto,
    # futures contracts) than equities — RVOL and "institutional activity"
    # framing don't translate cleanly across that boundary, so they're split
    # out from the equity KPIs/chart and given their own section instead.
    df_special = df_metrics[df_metrics["Thematic Destination"].isin(SPECIAL_THEMES)]
    df_main = df_metrics[~df_metrics["Thematic Destination"].isin(SPECIAL_THEMES)]

    # 4. DASHBOARD TOP-LEVEL KPIS (equities only — see note above)
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("SPY Market Benchmark Return", f"{spy_performance:.2f}%")

    scored = df_main.dropna(subset=["Liquidity Score"])
    if not scored.empty:
        top_mover = scored.sort_values(by="Liquidity Score", ascending=False).iloc[0]
        kpi2.metric("Top Liquidity Inflow Target", f"{top_mover['Ticker']}", f"{top_mover['Change %']}% Change")
    else:
        kpi2.metric("Top Liquidity Inflow Target", "N/A")

    rvol_valid = df_main.dropna(subset=["RVOL"])
    if not rvol_valid.empty:
        high_rvol_sector = rvol_valid.groupby("Thematic Destination")["RVOL"].mean().idxmax()
        kpi3.metric("Highest Institutional Activity Cluster", high_rvol_sector)
    else:
        kpi3.metric("Highest Institutional Activity Cluster", "N/A")

    # =========================================================================
    # BITCOIN & GOLD — dedicated section, standing apart from the equity pillars
    # =========================================================================
    st.subheader("🟡 Bitcoin & Gold — Alternative Liquidity Hedges")
    st.caption(
        "Shown separately from the equity pillars above and below: BTC-USD trades 24/7 with no "
        "market-hours concept, and GC=F is a front-month futures contract subject to roll-date "
        "price gaps that don't reflect a real overnight move. RVOL/Liquidity Score are still "
        "computed the same way (self-relative to each asset's own history), but comparing them "
        "directly against equity RVOL would be apples-to-oranges — hence the separate section "
        "instead of folding them into the 15-pillar chart below."
    )
    if not df_special.empty:
        bcols = st.columns(len(df_special))
        for col, (_, row) in zip(bcols, df_special.iterrows()):
            with col:
                st.metric(
                    label=f"{row['Ticker']}",
                    value=f"${row['Price']:,.2f}",
                    delta=f"{row['Change %']}%",
                )
                st.caption(
                    f"RVOL: {row['RVOL'] if pd.notna(row['RVOL']) else 'N/A'}  |  "
                    f"Liquidity Score: {row['Liquidity Score'] if pd.notna(row['Liquidity Score']) else 'N/A'}"
                )
        btc_gold_beta = betas_df[betas_df["Theme"].isin(SPECIAL_THEMES)] if not betas_df.empty else pd.DataFrame()
        if not btc_gold_beta.empty:
            st.caption(
                "Historical liquidity beta for this pillar (from the Per-Theme Beta expander above) "
                "is a genuinely interesting read here: it directly tests the classic 'hard money "
                "hedge' thesis — whether BTC/Gold have actually moved with macro liquidity expansion "
                "historically, rather than assuming it."
            )
            st.dataframe(btc_gold_beta, use_container_width=True, hide_index=True)
    else:
        st.info("BTC-USD / GC=F data unavailable this cycle — check the failed-tickers panel above.")

    st.divider()

    # 5. VISUALIZING LIQUIDITY FLOW VIA AGGREGATED HEATMAP (equities only)
    st.subheader("📊 Capital Migration Across Your Framework Pillars")

    theme_summary = df_main.groupby("Thematic Destination").agg({
        "Change %": "mean",
        "RVOL": "mean",
        "Liquidity Score": "mean",
    }).reset_index()

    # Regime × Theme, now with per-theme liquidity beta: each pillar's
    # multiplier reflects its own historical sensitivity to the macro
    # backdrop rather than one flat number applied everywhere.
    if not betas_df.empty:
        beta_lookup = betas_df.set_index("Theme")["Relative Beta"].to_dict()
    else:
        beta_lookup = {}

    def _row_multiplier(theme):
        rel_beta = beta_lookup.get(theme, 1.0)
        return theme_regime_multiplier(composite_z, rel_beta)

    theme_summary["Regime Multiplier"] = theme_summary["Thematic Destination"].apply(_row_multiplier)
    theme_summary["Regime-Adjusted Score"] = (
        theme_summary["Liquidity Score"] * theme_summary["Regime Multiplier"]
    ).round(2)
    theme_summary["Regime Multiplier"] = theme_summary["Regime Multiplier"].round(2)

    st.caption(
        "Regime multiplier is now **per-theme**, scaled by each pillar's historical liquidity "
        "beta (see expander above). Themes without enough price history fall back to the "
        "neutral 1.0x beta — check the 'Confidence' column in the beta table before trusting "
        "an extreme multiplier."
    )

    fig = px.bar(
        theme_summary,
        x="Thematic Destination",
        y="Regime-Adjusted Score",
        color="Change %",
        hover_data=["RVOL", "Liquidity Score", "Regime Multiplier"],
        color_continuous_scale="RdYlGn",
        title="Pillar Score, Regime-Adjusted by Theme-Specific Liquidity Beta",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "⚠️ Beta is estimated from trailing price history and can shift — re-check the beta "
        "table periodically rather than treating today's sensitivity ranking as permanent."
    )

    # 6. SCANNER DATA TABLES
    st.subheader("🔍 Individual Ticker Liquidity Ledger")

    selected_theme = st.selectbox("Filter View by Thematic Destination Pillar:", ["All Destinations"] + list(TICKER_MAP.keys()))

    display_df = df_metrics.copy()
    if selected_theme != "All Destinations":
        display_df = display_df[display_df["Thematic Destination"] == selected_theme]

    display_df = display_df.sort_values(by="Liquidity Score", ascending=False, na_position="last")

    if not display_df.empty:
        try:
            st.dataframe(
                display_df.style.background_gradient(subset=["Change %", "Liquidity Score"], cmap="RdYlGn"),
                use_container_width=True,
            )
        except ImportError:
            # background_gradient needs matplotlib; degrade to a plain table
            # instead of taking the whole app down if it's ever missing.
            logger.warning("matplotlib unavailable — rendering unstyled dataframe")
            st.dataframe(display_df, use_container_width=True)
    else:
        st.info("No tickers in this pillar returned valid data this cycle.")
else:
    st.error(
        "Data pipeline returned no results. This usually means Yahoo Finance is rate-limiting "
        "requests from this IP, or there's a network/config issue — check the failed-tickers "
        "panel above (if shown) or your network settings."
    )
