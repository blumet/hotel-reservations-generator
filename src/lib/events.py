"""
Event calendar logic.

Responsibilities:
- Load events.yaml (EndDate exclusive)
- Expand events into core dates + shoulder days
- Build a per-date event multiplier series for a target date window

Design choices:
- If multiple events overlap on the same date, we use the MAX multiplier.
  This is conservative (protects upside during compression).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Dict, Any, List

import pandas as pd
import yaml


@dataclass(frozen=True)
class Event:
    name: str
    start_date: dt.date
    end_date: dt.date  # exclusive
    demand_level: str
    shoulder_days: int = 0


def load_events(path: str) -> List[Event]:
    """
    Load events from events.yaml.

    Expected structure:
      version: "1.0"
      timezone: "Europe/Madrid"
      events:
        - name: ...
          start_date: "YYYY-MM-DD"
          end_date: "YYYY-MM-DD"   # exclusive
          demand_level: Peak|High|Medium
          shoulder_days: 1
    """
    with open(path, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)

    raw_events = doc.get("events", [])
    out: List[Event] = []

    for e in raw_events:
        name = str(e["name"])
        start_date = dt.date.fromisoformat(e["start_date"])
        end_date = dt.date.fromisoformat(e["end_date"])
        demand_level = str(e["demand_level"])
        shoulder_days = int(e.get("shoulder_days", 0) or 0)

        if end_date <= start_date:
            raise ValueError(f"Invalid event window for '{name}': end_date must be after start_date (exclusive).")

        out.append(
            Event(
                name=name,
                start_date=start_date,
                end_date=end_date,
                demand_level=demand_level,
                shoulder_days=shoulder_days,
            )
        )

    return out


def _daterange(start: dt.date, end_exclusive: dt.date):
    d = start
    while d < end_exclusive:
        yield d
        d += dt.timedelta(days=1)


def build_event_multiplier_series(
    events: List[Event],
    start_date: dt.date,
    end_date: dt.date,
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    """
    Build a DataFrame of event multipliers for each date in [start_date, end_date).

    Returns a DataFrame with columns:
      - stay_date
      - event_multiplier
      - event_name (best/strongest event that set the multiplier; optional but useful for "reason")

    Rules:
      - Core event dates get full event weight.
      - Shoulder dates get shoulder_factor * uplift (only the uplift portion).
        Example: Peak weight 1.20 => uplift 0.20; shoulder_factor 0.5 => shoulder weight 1.10
      - Overlaps use MAX multiplier; event_name is the event that wins.
    """
    weights = cfg["events"]["weights"]  # e.g. Peak:1.20, High:1.10, Medium:1.05
    shoulder_factor = float(cfg["events"].get("shoulder_factor", 0.5))

    # Initialize dates with neutral multiplier
    dates = list(_daterange(start_date, end_date))
    best_mult = {d: 1.0 for d in dates}
    best_name = {d: "" for d in dates}

    for ev in events:
        if ev.demand_level not in weights:
            # Unknown level => skip (explicitly conservative)
            continue

        full_weight = float(weights[ev.demand_level])
        uplift = full_weight - 1.0
        shoulder_weight = 1.0 + uplift * shoulder_factor

        # Expand shoulder window around event dates
        shoulder_days = max(ev.shoulder_days, 0)
        expanded_start = ev.start_date - dt.timedelta(days=shoulder_days)
        expanded_end = ev.end_date + dt.timedelta(days=shoulder_days)

        for d in _daterange(expanded_start, expanded_end):
            if d < start_date or d >= end_date:
                continue

            # Determine whether this date is core or shoulder
            is_core = (ev.start_date <= d < ev.end_date)
            mult = full_weight if is_core else shoulder_weight

            # If this event is stronger than current best, take it
            if mult > best_mult[d]:
                best_mult[d] = mult
                best_name[d] = ev.name

    df = pd.DataFrame(
        {
            "stay_date": dates,
            "event_multiplier": [best_mult[d] for d in dates],
            "event_name": [best_name[d] for d in dates],
        }
    )
    return df
