"""
OTB (On-The-Books) processing logic.

Responsibilities:
- Load BusinessOnTheBooks.csv
- Normalize dates
- Aggregate by stay date
- Compute:
  - Net OTB
  - Expected wash (cancellations + no-shows)
  - Effective OTB
  - Forecast occupancy

This module intentionally knows NOTHING about pricing.
"""

from __future__ import annotations

import datetime as dt
from typing import Dict, Any

import pandas as pd


def load_otb_dataframe(path: str, cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Load BusinessOnTheBooks.csv and normalize the stay date column.
    """
    df = pd.read_csv(path)

    date_col = cfg["otb"]["date_column"]
    if date_col not in df.columns:
        raise ValueError(f"OTB date column not found: {date_col}")

    df[date_col] = pd.to_datetime(df[date_col]).dt.date
    return df


def _is_weekend(d: dt.date) -> bool:
    # Saturday (5) or Sunday (6)
    return d.weekday() >= 5


def build_daily_metrics(
    otb_df: pd.DataFrame,
    cfg: Dict[str, Any],
    as_of: dt.date,
) -> pd.DataFrame:
    """
    Aggregate OTB to stay-date level and compute forecast metrics.

    Returns DataFrame indexed by StayDate with columns:
      - capacity
      - rooms_sold
      - group_rooms
      - net_otb
      - expected_wash
      - effective_otb
      - forecast_occupancy
    """

    date_col = cfg["otb"]["date_column"]
    cap_col = cfg["otb"]["capacity_column"]
    sold_col = cfg["otb"]["rooms_sold_column"]
    group_col = cfg["otb"].get("group_rooms_column")

    # Aggregate by stay date
    agg = {
        cap_col: "max",      # capacity should be constant per date
        sold_col: "sum",
    }
    if group_col and group_col in otb_df.columns:
        agg[group_col] = "sum"

    daily = (
        otb_df
        .groupby(date_col, as_index=False)
        .agg(agg)
        .rename(columns={
            date_col: "stay_date",
            cap_col: "capacity",
            sold_col: "rooms_sold",
        })
    )

    if group_col and group_col in otb_df.columns:
        daily = daily.rename(columns={group_col: "group_rooms"})
    else:
        daily["group_rooms"] = 0.0

    # Net OTB
    daily["net_otb"] = daily["rooms_sold"] + daily["group_rooms"]

    # Expected wash
    cancel_cfg = cfg["defaults"]["cancel_rate"]
    noshow_cfg = cfg["defaults"]["no_show_rate"]

    expected_wash = []
    for _, row in daily.iterrows():
        d = row["stay_date"]
        if _is_weekend(d):
            cancel_rate = cancel_cfg["weekend"]
            noshow_rate = noshow_cfg["weekend"]
        else:
            cancel_rate = cancel_cfg["weekday"]
            noshow_rate = noshow_cfg["weekday"]

        wash = row["net_otb"] * (cancel_rate + noshow_rate)
        expected_wash.append(wash)

    daily["expected_wash"] = expected_wash

    # Effective OTB after wash
    daily["effective_otb"] = daily["net_otb"] - daily["expected_wash"]

    # Forecast occupancy
    daily["forecast_occupancy"] = daily.apply(
        lambda r: r["effective_otb"] / r["capacity"] if r["capacity"] > 0 else 0.0,
        axis=1,
    )

    # Drop dates already in the past
    daily = daily[daily["stay_date"] >= as_of].reset_index(drop=True)

    return daily
