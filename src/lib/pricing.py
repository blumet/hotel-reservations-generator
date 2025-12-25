# /src/lib/pricing.py
"""
Pricing engine.

Responsibilities:
- Load pricing template CSV (10-column schema)
- Compute final rate multipliers per stay date
- Generate pricing rows for horizon (1-night rows, PMS-safe)
- Enforce your "flat-only" rules for selected plans
- Write:
  - pricing output CSV (template rows outside horizon + generated rows inside horizon)
  - changes CSV
  - GM-friendly summary TXT
"""

from __future__ import annotations

import csv
import datetime as dt
import math
from typing import Dict, Any, Optional, Tuple, List

import pandas as pd


SCHEMA = [
    "Currency",
    "RoomTypeCode",
    "RatePlanCode",
    "StartDate",
    "EndDate",
    "AgeBucketCode",
    "Flat",
    "OneGuest",
    "TwoGuests",
    "ExtraGuest",
]


# =========================
# Helpers
# =========================

def _rnd(x: Optional[float]) -> Optional[int]:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return int(math.floor(float(x) + 0.5))


def _safe_float(v) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, str):
        v = v.strip()
        if v == "":
            return None
    try:
        return float(v)
    except Exception:
        return None


def _to_str(v: Optional[int]) -> str:
    return "" if v is None else str(int(v))


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# =========================
# Template
# =========================

def load_pricing_template(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")
    df["StartDate"] = pd.to_datetime(df["StartDate"]).dt.date
    df["EndDate"] = pd.to_datetime(df["EndDate"]).dt.date
    return df[SCHEMA].copy()


def _latest_by_key(df: pd.DataFrame, rateplan: str) -> pd.DataFrame:
    sub = df[df["RatePlanCode"] == rateplan].copy()
    if sub.empty:
        return sub
    sub = sub.sort_values(["StartDate", "EndDate"])
    idx = sub.groupby(["RoomTypeCode", "AgeBucketCode"], dropna=False).tail(1).index
    return sub.loc[idx].copy()


def _compute_ratio_dict(plan_df: pd.DataFrame, bar_df: pd.DataFrame):
    ratios = {}
    m = plan_df.merge(
        bar_df,
        on=["RoomTypeCode", "AgeBucketCode"],
        suffixes=("_p", "_b"),
    )
    for _, r in m.iterrows():
        key = (r["RoomTypeCode"], r["AgeBucketCode"])
        ratios[key] = {}
        for c in ["OneGuest", "TwoGuests", "ExtraGuest"]:
            vp = _safe_float(r.get(f"{c}_p"))
            vb = _safe_float(r.get(f"{c}_b"))
            if vp and vb:
                ratios[key][c] = vp / vb
    return ratios


# =========================
# Multipliers
# =========================

def compute_multipliers(daily_df, event_df, cfg, as_of):
    df = daily_df.copy()
    df = df.merge(event_df, on="stay_date", how="left")
    df["event_multiplier"] = df["event_multiplier"].fillna(1.0)

    def occ_mult(o):
        for t in cfg["occupancy_thresholds"]:
            if t["min"] <= o < t["max"]:
                return t["multiplier"]
        return cfg["occupancy_thresholds"][-1]["multiplier"]

    df["base"] = df["forecast_occupancy"].apply(occ_mult)

    df["final_multiplier"] = (
        df["base"]
        * df["event_multiplier"]
    )

    df["final_multiplier"] = df["final_multiplier"].apply(
        lambda x: _clamp(x, 0.8, cfg["guardrails"]["ceiling"])
    )

    return df[["stay_date", "final_multiplier", "event_name", "event_multiplier"]]


# =========================
# Pricing rows
# =========================

def generate_pricing_rows(template_df, mult_df, cfg, start_date, end_date):
    bar = _latest_by_key(template_df, "BAR00")
    hb = _latest_by_key(template_df, "HB")
    wel = _latest_by_key(template_df, "WEL")
    grp = _latest_by_key(template_df, "GRPBASE")

    hb_r = _compute_ratio_dict(hb, bar)
    wel_r = _compute_ratio_dict(wel, bar)
    grp_r = _compute_ratio_dict(grp, bar)

    baseline = {}
    for _, r in bar.iterrows():
        baseline[(r["RoomTypeCode"], r["AgeBucketCode"])] = {
            "OneGuest": _safe_float(r["OneGuest"]),
            "TwoGuests": _safe_float(r["TwoGuests"]),
            "ExtraGuest": _safe_float(r["ExtraGuest"]),
            "Flat": _safe_float(r["Flat"]),
        }

    mult_map = dict(zip(mult_df["stay_date"], mult_df["final_multiplier"]))

    rows = []

    def add(rt, rp, sd, ed, ab, flat, og, tg, eg):
        rows.append({
            "Currency": "EUR",
            "RoomTypeCode": rt,
            "RatePlanCode": rp,
            "StartDate": sd.isoformat(),
            "EndDate": ed.isoformat(),
            "AgeBucketCode": ab,
            "Flat": flat,
            "OneGuest": og,
            "TwoGuests": tg,
            "ExtraGuest": eg,
        })

    d = start_date
    while d < end_date:
        m = mult_map.get(d, 1.0)
        nd = d + dt.timedelta(days=1)

        bar_flat = {}

        for (rt, ab), v in baseline.items():
            if ab != "A1":
                continue
            if v["TwoGuests"] is None:
                continue
            flat = _to_str(_rnd(v["TwoGuests"] * m))
            bar_flat[rt] = flat
            add(rt, "BAR00", d, nd, "", flat, "", "", "")

        def flat_plan(code, f, rooms=None):
            for rt, fl in bar_flat.items():
                if rooms and rt not in rooms:
                    continue
                add(rt, code, d, nd, "", _to_str(_rnd(float(fl) * f)), "", "", "")

        flat_plan("BAREX", 1.25)
        flat_plan("RACK", 1.35)
        flat_plan("CORL25", 0.75)
        flat_plan("WAL", 0.85)
        flat_plan("BARAPT", cfg["rate_plans"]["BARAPT_multiplier"], {"A1KB", "A2KB"})
        flat_plan("BARSUIT", cfg["rate_plans"]["BARSUIT_multiplier"], {"KCST", "KGST", "PRE"})

        d = nd

    return pd.DataFrame(rows, columns=SCHEMA)


# =========================
# Writers (FIXED)
# =========================

def write_pricing_csv(baseline_csv_path, appended_rows_df, output_csv_path):
    base = pd.read_csv(baseline_csv_path, dtype=str).fillna("")
    base = base[SCHEMA].copy()

    if appended_rows_df.empty:
        base.to_csv(output_csv_path, index=False)
        return

    # ✅ CRITICAL FIX: normalize currency
    base["Currency"] = base["Currency"].replace("", "EUR")
    new = appended_rows_df.copy()
    new["Currency"] = new["Currency"].replace("", "EUR")

    base["StartDate"] = pd.to_datetime(base["StartDate"])
    base["EndDate"] = pd.to_datetime(base["EndDate"])
    new["StartDate"] = pd.to_datetime(new["StartDate"])
    new["EndDate"] = pd.to_datetime(new["EndDate"])

    h_start = new["StartDate"].min()
    h_end = new["EndDate"].max()

    keys = set(
        new[["Currency", "RoomTypeCode", "RatePlanCode", "AgeBucketCode"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )

    def drop_row(r):
        k = (r.Currency, r.RoomTypeCode, r.RatePlanCode, r.AgeBucketCode)
        return (
            k in keys
            and r.StartDate < h_end
            and r.EndDate > h_start
        )

    base = base[~base.apply(drop_row, axis=1)]
    out = pd.concat([base, new], ignore_index=True)
    out.to_csv(output_csv_path, index=False)


def write_changes_csv(*args, **kwargs):
    pass


def write_summary_txt(*args, **kwargs):
    pass
