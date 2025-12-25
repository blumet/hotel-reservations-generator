# /src/lib/pricing.py
"""
Pricing engine with OBSERVED demand wash (cancellations + no-shows).

Key principles:
- Use real cancellations / no-shows if provided
- Fall back to config assumptions if not
- Never claim what we do not measure
- Produce PMS-safe DAILY pricing rows
"""

from __future__ import annotations

import csv
import datetime as dt
import math
from typing import Dict, Any, Optional, Tuple, List
import os

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


# =====================================================
# Helpers
# =====================================================

def _rnd(x: Optional[float]) -> Optional[int]:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return int(math.floor(float(x) + 0.5))


def _safe_float(v) -> Optional[float]:
    try:
        if v in ("", None):
            return None
        return float(v)
    except Exception:
        return None


def _to_str(v: Optional[int]) -> str:
    return "" if v is None else str(int(v))


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# =====================================================
# Load observed cancellations / no-shows
# =====================================================

def load_observed_wash(
    cancellations_path: str,
    noshows_path: str,
) -> pd.DataFrame:
    """
    Returns:
      stay_date | observed_wash_rooms | observed
    """
    dfs = []

    if os.path.exists(cancellations_path):
        c = pd.read_csv(cancellations_path)
        c["StayDate"] = pd.to_datetime(c["StayDate"]).dt.date
        c = c.groupby("StayDate", as_index=False)["RoomsCancelled"].sum()
        c.rename(columns={"RoomsCancelled": "wash"}, inplace=True)
        dfs.append(c)

    if os.path.exists(noshows_path):
        n = pd.read_csv(noshows_path)
        n["StayDate"] = pd.to_datetime(n["StayDate"]).dt.date
        n = n.groupby("StayDate", as_index=False)["RoomsNoShow"].sum()
        n.rename(columns={"RoomsNoShow": "wash"}, inplace=True)
        dfs.append(n)

    if not dfs:
        return pd.DataFrame(columns=["stay_date", "observed_wash", "observed"])

    df = pd.concat(dfs)
    df = df.groupby("StayDate", as_index=False)["wash"].sum()
    df.rename(columns={"StayDate": "stay_date", "wash": "observed_wash"}, inplace=True)
    df["observed"] = True
    return df


# =====================================================
# Multipliers (with observed wash)
# =====================================================

def compute_multipliers(
    daily_df: pd.DataFrame,
    event_df: pd.DataFrame,
    cfg: Dict[str, Any],
    as_of: dt.date,
) -> pd.DataFrame:
    df = daily_df.copy()

    # Merge observed wash if present
    wash_df = load_observed_wash(
        "data/Cancellations.csv",
        "data/NoShows.csv",
    )

    df = df.merge(wash_df, on="stay_date", how="left")

    # Fallback to assumed wash if no observed data
    def expected_wash(row):
        if row.get("observed"):
            return row["observed_wash"]
        dow = row["stay_date"].weekday()
        rate = (
            cfg["defaults"]["cancel_rate"]["weekend"]
            + cfg["defaults"]["no_show_rate"]["weekend"]
            if dow >= 4
            else cfg["defaults"]["cancel_rate"]["weekday"]
            + cfg["defaults"]["no_show_rate"]["weekday"]
        )
        return row["rooms_sold"] * rate

    df["wash_rooms"] = df.apply(expected_wash, axis=1)
    df["net_demand"] = df["rooms_sold"] - df["wash_rooms"]
    df["forecast_occupancy"] = df["net_demand"] / df["capacity"]

    # Occupancy multiplier
    def occ_mult(o):
        for t in cfg["occupancy_thresholds"]:
            if t["min"] <= o < t["max"]:
                return t["multiplier"]
        return cfg["occupancy_thresholds"][-1]["multiplier"]

    df["occ_mult"] = df["forecast_occupancy"].apply(occ_mult)

    # Event multiplier
    df = df.merge(event_df, on="stay_date", how="left")
    df["event_multiplier"] = df["event_multiplier"].fillna(1.0)

    df["raw_multiplier"] = df["occ_mult"] * df["event_multiplier"]

    # Guardrails
    ceiling = cfg["guardrails"]["ceiling"]
    df["final_multiplier"] = df["raw_multiplier"].apply(
        lambda x: _clamp(x, 0.8, ceiling)
    )

    df["reason"] = df.apply(
        lambda r: (
            "Observed wash applied"
            if r.get("observed")
            else "Assumed wash applied"
        ),
        axis=1,
    )

    return df[
        ["stay_date", "final_multiplier", "reason", "event_multiplier"]
    ]


# =====================================================
# Pricing rows (flat BAR-based)
# =====================================================

def generate_pricing_rows(
    template_df: pd.DataFrame,
    multipliers_df: pd.DataFrame,
    cfg: Dict[str, Any],
    start_date: dt.date,
    end_date: dt.date,
) -> pd.DataFrame:

    bar = template_df[template_df["RatePlanCode"] == "BAR00"]
    bar = bar.sort_values("EndDate").groupby(
        ["RoomTypeCode", "AgeBucketCode"], as_index=False
    ).tail(1)

    base = {
        (r.RoomTypeCode, r.AgeBucketCode): _safe_float(r.TwoGuests)
        for r in bar.itertuples()
        if r.AgeBucketCode == "A1"
    }

    mult_map = dict(
        zip(multipliers_df["stay_date"], multipliers_df["final_multiplier"])
    )

    rows = []

    d = start_date
    while d < end_date:
        m = mult_map.get(d, 1.0)
        nd = d + dt.timedelta(days=1)

        for (rt, _), base_price in base.items():
            if base_price is None:
                continue

            bar_flat = _to_str(_rnd(base_price * m))

            rows.append({
                "Currency": "EUR",
                "RoomTypeCode": rt,
                "RatePlanCode": "BAR00",
                "StartDate": d.isoformat(),
                "EndDate": nd.isoformat(),
                "AgeBucketCode": "",
                "Flat": bar_flat,
                "OneGuest": "",
                "TwoGuests": "",
                "ExtraGuest": "",
            })

        d = nd

    return pd.DataFrame(rows, columns=SCHEMA)


# =====================================================
# Writers
# =====================================================

def write_pricing_csv(template_path, new_df, output_path):
    base = pd.read_csv(template_path, dtype=str).fillna("")
    base["Currency"] = base["Currency"].replace("", "EUR")
    new_df["Currency"] = new_df["Currency"].replace("", "EUR")

    out = pd.concat([base, new_df], ignore_index=True)
    out.to_csv(output_path, index=False)


def write_summary_txt(
    multipliers_df: pd.DataFrame,
    pricing_df: pd.DataFrame,
    summary_path: str,
    as_of: dt.date,
):
    bar = pricing_df[pricing_df["RatePlanCode"] == "BAR00"]
    bar["Flat"] = bar["Flat"].astype(float)

    avg = int(bar["Flat"].mean())
    mn = int(bar["Flat"].min())
    mx = int(bar["Flat"].max())

    observed_days = multipliers_df[multipliers_df["reason"] == "Observed wash applied"]

    lines = [
        f"Revenue Manager Summary — {as_of.isoformat()}",
        "",
        "Rates were adjusted using observed cancellations and no-shows where available.",
        "When no observed data existed, conservative assumed wash rates were applied.",
        "",
        f"Next 14 days BAR00 flat pricing:",
        f"- Average: €{avg}",
        f"- Min: €{mn}",
        f"- Max: €{mx}",
        "",
        f"Observed wash applied on {len(observed_days)} days.",
    ]

    with open(summary_path, "w") as f:
        f.write("\n".join(lines))
