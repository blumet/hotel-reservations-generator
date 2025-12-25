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

IMPORTANT RULES (current):
Flat-only (AgeBucketCode blank, Flat populated, per-person columns blank) for:
- BAR00, BAREX, RACK, CORL25, WAL, GRPHIG, BARAPT, BARSUIT

Flat basis:
- For non-PRE room types, flat is taken from BAR00 TwoGuests(A1) after multiplier+rounding.
- PRE remains flat-only using template Flat (AgeBucketCode blank) after multiplier+rounding.

Per-person plans still generated (A1/C1):
- OPQ, STAFF, BB, WEL3, HB, WEL, GRPBASE, SUHB

CRITICAL PMS IMPORT FIX:
- When writing RatePricing.csv, remove any TEMPLATE rows that overlap the generated horizon
  for the SAME (Currency, RoomTypeCode, RatePlanCode, AgeBucketCode) keys.
  This prevents overlapping pricing periods in the PMS.
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


# --------------------------
# Helpers
# --------------------------

def _rnd(x: Optional[float]) -> Optional[int]:
    """Normal rounding to whole EUR."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return int(math.floor(float(x) + 0.5))


def _safe_float(v) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, int):
        return float(v)
    if isinstance(v, str):
        t = v.strip()
        if t == "":
            return None
        try:
            return float(t)
        except ValueError:
            return None
    try:
        return float(v)
    except Exception:
        return None


def _to_str_or_blank(v: Optional[int]) -> str:
    return "" if v is None else str(int(v))


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# --------------------------
# Template loading
# --------------------------

def load_pricing_template(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")
    missing = [c for c in SCHEMA if c not in df.columns]
    if missing:
        raise ValueError(f"Template missing required columns: {missing}")

    df["StartDate"] = pd.to_datetime(df["StartDate"]).dt.date
    df["EndDate"] = pd.to_datetime(df["EndDate"]).dt.date
    df["AgeBucketCode"] = df["AgeBucketCode"].fillna("")

    return df[SCHEMA].copy()


def _latest_by_key(df: pd.DataFrame, rateplan: str) -> pd.DataFrame:
    sub = df[df["RatePlanCode"] == rateplan].copy()
    if sub.empty:
        return sub
    sub = sub.sort_values(["StartDate", "EndDate"])
    idx = sub.groupby(["RoomTypeCode", "AgeBucketCode"], dropna=False).tail(1).index
    return sub.loc[idx].copy()


def _compute_ratio_dict(plan_latest: pd.DataFrame, bar_latest: pd.DataFrame) -> Dict[Tuple[str, str], Dict[str, float]]:
    """
    Compute per-key ratios (plan / BAR00) for OneGuest/TwoGuests/ExtraGuest and optionally Flat.
    Only uses columns present (non-blank) in both plan and BAR00.
    """
    ratios: Dict[Tuple[str, str], Dict[str, float]] = {}
    merged = plan_latest.merge(
        bar_latest,
        on=["RoomTypeCode", "AgeBucketCode"],
        how="inner",
        suffixes=("_plan", "_bar"),
    )
    for _, r in merged.iterrows():
        key = (r["RoomTypeCode"], r["AgeBucketCode"])
        ratios[key] = {}
        for col in ["Flat", "OneGuest", "TwoGuests", "ExtraGuest"]:
            v_plan = _safe_float(r.get(f"{col}_plan"))
            v_bar = _safe_float(r.get(f"{col}_bar"))
            if v_plan is None or v_bar is None or v_bar == 0:
                continue
            ratios[key][col] = v_plan / v_bar
    return ratios


# --------------------------
# Multipliers computation
# --------------------------

def compute_multipliers(
    daily_metrics_df: pd.DataFrame,
    event_mult_df: pd.DataFrame,
    cfg: Dict[str, Any],
    as_of: dt.date,
) -> pd.DataFrame:
    """
    Returns DataFrame with:
      stay_date, final_multiplier, reason, event_name, event_multiplier
    """
    df = daily_metrics_df.copy()

    ev = event_mult_df.copy()
    df = df.merge(ev, on="stay_date", how="left")
    df["event_multiplier"] = df["event_multiplier"].fillna(1.0)
    df["event_name"] = df["event_name"].fillna("")

    thresholds = cfg["occupancy_thresholds"]

    def base_from_occ(occ: float) -> float:
        for t in thresholds:
            if float(t["min"]) <= occ < float(t["max"]):
                return float(t["multiplier"])
        return float(thresholds[-1]["multiplier"])

    df["base_occ_multiplier"] = df["forecast_occupancy"].apply(base_from_occ)

    # v1 pickup placeholder
    df["pickup_multiplier"] = 1.0

    # wash multiplier (small safety if wash rate is high)
    def wash_mult(row) -> float:
        net_otb = float(row.get("net_otb", 0.0))
        expected_wash = float(row.get("expected_wash", 0.0))
        if net_otb <= 0:
            return 1.0
        wash_rate = expected_wash / net_otb
        return 0.97 if wash_rate > 0.12 else 1.00

    df["wash_multiplier"] = df.apply(wash_mult, axis=1)

    # group displacement
    gcfg = cfg["groups"]
    thresh = float(gcfg["displacement_threshold"])
    uplift_low = float(gcfg["uplift_low"])
    uplift_high = float(gcfg["uplift_high"])

    def group_mult(row) -> float:
        cap = float(row.get("capacity", 0.0))
        grp = float(row.get("group_rooms", 0.0))
        if cap <= 0:
            return 1.0
        share = grp / cap
        if share >= thresh and float(row.get("forecast_occupancy", 0.0)) >= 0.75:
            return uplift_high
        if share >= thresh:
            return uplift_low
        return 1.0

    df["group_multiplier"] = df.apply(group_mult, axis=1)

    df["raw_multiplier"] = (
        df["base_occ_multiplier"]
        * df["pickup_multiplier"]
        * df["wash_multiplier"]
        * df["group_multiplier"]
        * df["event_multiplier"]
    )

    guard = cfg["guardrails"]
    ceiling = float(guard["ceiling"])
    low_floor = float(guard["floors"]["low_season"])

    def floor_for_date(d: dt.date) -> float:
        if d.month in (7, 8):
            return float(guard["floors"]["summer"])
        if (d.month == 1 and d.day >= 7) or (d.month == 11 and d.day >= 10):
            return float(guard["floors"]["low_season"])
        return low_floor

    df["floor"] = df["stay_date"].apply(floor_for_date)

    compression_floor = float(guard["floors"]["compression_event"])
    peak_weight = float(cfg["events"]["weights"].get("Peak", 1.20))
    df["floor"] = df.apply(
        lambda r: max(float(r["floor"]), compression_floor)
        if float(r["event_multiplier"]) >= peak_weight else float(r["floor"]),
        axis=1,
    )

    df["final_multiplier"] = df.apply(
        lambda r: _clamp(float(r["raw_multiplier"]), float(r["floor"]), ceiling),
        axis=1,
    )

    def reason(row) -> str:
        parts = [f"occ={row['forecast_occupancy']:.2f}->{row['base_occ_multiplier']:.2f}"]
        if float(row["event_multiplier"]) > 1.0:
            parts.append(f"event={row['event_multiplier']:.2f}:{row['event_name']}")
        if float(row["group_multiplier"]) > 1.0:
            parts.append(f"grp={row['group_multiplier']:.2f}")
        if float(row["wash_multiplier"]) < 1.0:
            parts.append(f"wash={row['wash_multiplier']:.2f}")
        return " | ".join(parts)

    df["reason"] = df.apply(reason, axis=1)

    return df[["stay_date", "final_multiplier", "reason", "event_name", "event_multiplier"]].copy()


# --------------------------
# Row generation (PMS-safe, daily rows)
# --------------------------

def generate_pricing_rows(
    template_df: pd.DataFrame,
    multipliers_df: pd.DataFrame,
    cfg: Dict[str, Any],
    start_date: dt.date,
    end_date: dt.date,
) -> pd.DataFrame:
    """
    Generate pricing rows for each day in [start_date, end_date) as 1-night bands.
    Applies your "flat-only" plan rules.
    """
    bar_latest = _latest_by_key(template_df, "BAR00")
    if bar_latest.empty:
        raise ValueError("Template has no BAR00 rows; cannot build any pricing.")

    # Ratio plans come from template structure
    hb_latest = _latest_by_key(template_df, "HB")
    wel_latest = _latest_by_key(template_df, "WEL")
    grp_latest = _latest_by_key(template_df, "GRPBASE")

    hb_ratios = _compute_ratio_dict(hb_latest, bar_latest) if not hb_latest.empty else {}
    wel_ratios = _compute_ratio_dict(wel_latest, bar_latest) if not wel_latest.empty else {}
    grp_ratios = _compute_ratio_dict(grp_latest, bar_latest) if not grp_latest.empty else {}

    # Baseline BAR00 values by (RoomTypeCode, AgeBucketCode)
    baseline: Dict[Tuple[str, str], Dict[str, Optional[float]]] = {}
    for _, r in bar_latest.iterrows():
        key = (r["RoomTypeCode"], r["AgeBucketCode"])
        baseline[key] = {
            "Flat": _safe_float(r["Flat"]),
            "OneGuest": _safe_float(r["OneGuest"]),
            "TwoGuests": _safe_float(r["TwoGuests"]),
            "ExtraGuest": _safe_float(r["ExtraGuest"]),
        }

    mult_map = {row["stay_date"]: float(row["final_multiplier"]) for _, row in multipliers_df.iterrows()}

    rows: List[Dict[str, str]] = []

    def add_row(currency, rt, plan, sd, ed, ab, flat, og, tg, eg):
        rows.append({
            "Currency": currency,
            "RoomTypeCode": rt,
            "RatePlanCode": plan,
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
        mult = mult_map.get(d, 1.0)
        next_d = d + dt.timedelta(days=1)

        # Internal per-person BAR00 per (rt, ab) for other per-person plans
        bar_pp_per_key: Dict[Tuple[str, str], Tuple[str, str, str]] = {}

        # Output BAR00 flat per room type (flat basis = TwoGuests(A1))
        bar_flat_by_room: Dict[str, str] = {}

        # ---------- 1) Build internal BAR00 per-person + output BAR00 flat ----------
        for (rt, ab), base_vals in baseline.items():
            currency = "EUR"

            # PRE remains template flat-only (AgeBucket blank)
            if rt == "PRE" and ab == "":
                base_flat = base_vals.get("Flat")
                if base_flat is None:
                    continue
                flat = _to_str_or_blank(_rnd(base_flat * mult))
                bar_flat_by_room["PRE"] = flat
                add_row(currency, rt, "BAR00", d, next_d, "", flat, "", "", "")
                continue

            # internal per-person BAR00 (used by per-person plans)
            if ab != "":
                base_og = base_vals.get("OneGuest")
                base_tg = base_vals.get("TwoGuests")
                base_eg = base_vals.get("ExtraGuest")
                if base_og is not None and base_tg is not None and base_eg is not None:
                    og = _to_str_or_blank(_rnd(base_og * mult))
                    tg = _to_str_or_blank(_rnd(base_tg * mult))
                    eg = _to_str_or_blank(_rnd(base_eg * mult))
                    bar_pp_per_key[(rt, ab)] = (og, tg, eg)

                    # Flat basis rule: TwoGuests(A1) -> Flat BAR00 (for non-PRE)
                    if ab == "A1":
                        bar_flat_by_room[rt] = tg

        # Output BAR00 flat rows for all room types (non-PRE)
        for rt, flat in bar_flat_by_room.items():
            if rt == "PRE":
                continue
            add_row("EUR", rt, "BAR00", d, next_d, "", flat, "", "", "")

        # ---------- 2) Flat-only derived plans from BAR00 flat ----------
        def add_flat_plan(plan: str, factor: float, room_filter=None):
            for rt, bar_flat in bar_flat_by_room.items():
                if room_filter and rt not in room_filter:
                    continue
                new_flat = _to_str_or_blank(_rnd(float(bar_flat) * factor))
                add_row("EUR", rt, plan, d, next_d, "", new_flat, "", "", "")

        add_flat_plan("BAREX", 1.25)
        add_flat_plan("RACK", 1.35)
        add_flat_plan("CORL25", 0.75)
        add_flat_plan("WAL", 0.85)

        # BARAPT flat-only (A1KB/A2KB only)
        barapt_factor = float(cfg["rate_plans"]["BARAPT_multiplier"])
        add_flat_plan("BARAPT", barapt_factor, room_filter={"A1KB", "A2KB"})

        # BARSUIT flat-only (KCST/KGST/PRE only)
        barsuit_factor = float(cfg["rate_plans"]["BARSUIT_multiplier"])
        add_flat_plan("BARSUIT", barsuit_factor, room_filter={"KCST", "KGST", "PRE"})

        # ---------- 3) Per-person plans (unchanged) based on internal BAR00 per-person ----------
        def derive_pp_simple(plan: str, factor: float):
            for (rt, ab), (og, tg, eg) in bar_pp_per_key.items():
                add_row(
                    "EUR", rt, plan, d, next_d, ab, "",
                    _to_str_or_blank(_rnd(float(og) * factor)),
                    _to_str_or_blank(_rnd(float(tg) * factor)),
                    _to_str_or_blank(_rnd(float(eg) * factor)),
                )

        derive_pp_simple("OPQ", 0.80)
        derive_pp_simple("STAFF", 0.50)

        # BB per-person (+35)
        for (rt, ab), (og, tg, eg) in bar_pp_per_key.items():
            add_row(
                "EUR", rt, "BB", d, next_d, ab, "",
                _to_str_or_blank(_rnd(float(og) + 35)),
                _to_str_or_blank(_rnd(float(tg) + 35)),
                _to_str_or_blank(_rnd(float(eg) + 35)),
            )

        # WEL3 weekends-only per-person (Fri/Sat nights)
        if d.weekday() in (4, 5):
            for (rt, ab), (og, tg, eg) in bar_pp_per_key.items():
                add_row(
                    "EUR", rt, "WEL3", d, next_d, ab, "",
                    _to_str_or_blank(_rnd(float(og) * 1.35)),
                    _to_str_or_blank(_rnd(float(tg) * 1.35)),
                    _to_str_or_blank(_rnd(float(eg) * 1.35)),
                )

        # Ratio-based per-person plans (HB/WEL/GRPBASE) remain per-person
        def apply_ratio_plan(plan: str, ratios: Dict[Tuple[str, str], Dict[str, float]]):
            for (rt, ab), (og, tg, eg) in bar_pp_per_key.items():
                ratio = ratios.get((rt, ab))
                if not ratio:
                    continue
                add_row(
                    "EUR", rt, plan, d, next_d, ab, "",
                    _to_str_or_blank(_rnd(float(og) * ratio.get("OneGuest", 1.0))),
                    _to_str_or_blank(_rnd(float(tg) * ratio.get("TwoGuests", 1.0))),
                    _to_str_or_blank(_rnd(float(eg) * ratio.get("ExtraGuest", 1.0))),
                )

        apply_ratio_plan("HB", hb_ratios)
        apply_ratio_plan("WEL", wel_ratios)
        apply_ratio_plan("GRPBASE", grp_ratios)

        # SUHB per-person = HB * 1.25
        for (rt, ab), (og, tg, eg) in bar_pp_per_key.items():
            ratio = hb_ratios.get((rt, ab))
            if not ratio:
                continue
            h_og = _rnd(float(og) * ratio.get("OneGuest", 1.0))
            h_tg = _rnd(float(tg) * ratio.get("TwoGuests", 1.0))
            h_eg = _rnd(float(eg) * ratio.get("ExtraGuest", 1.0))
            add_row(
                "EUR", rt, "SUHB", d, next_d, ab, "",
                _to_str_or_blank(_rnd(h_og * 1.25 if h_og is not None else None)),
                _to_str_or_blank(_rnd(h_tg * 1.25 if h_tg is not None else None)),
                _to_str_or_blank(_rnd(h_eg * 1.25 if h_eg is not None else None)),
            )

        # ---------- 4) GRPHIG flat-only (derive from BAR00 flat) ----------
        # Use template GRPBASE TwoGuests ratio if present for (rt,"A1"), else fallback 0.70
        for rt, bar_flat in bar_flat_by_room.items():
            r = grp_ratios.get((rt, "A1"), {})
            grpbase_factor = float(r.get("TwoGuests", 0.70))
            grphig_flat = _to_str_or_blank(_rnd(float(bar_flat) * grpbase_factor * 1.25))
            add_row("EUR", rt, "GRPHIG", d, next_d, "", grphig_flat, "", "", "")

        d = next_d

    return pd.DataFrame(rows, columns=SCHEMA)


# --------------------------
# Writers
# --------------------------

def write_pricing_csv(
    baseline_csv_path: str,
    appended_rows_df: pd.DataFrame,
    output_csv_path: str,
) -> None:
    """
    Write output CSV as: (Template rows kept) + (Generated rows),
    BUT remove any template rows that overlap the generated horizon for the SAME key:
      (Currency, RoomTypeCode, RatePlanCode, AgeBucketCode)

    This prevents "Overlapping pricing periods" errors in the PMS.
    """
    base = pd.read_csv(baseline_csv_path, dtype=str).fillna("")
    base = base[SCHEMA].copy()

    if appended_rows_df is None or appended_rows_df.empty:
        base.to_csv(output_csv_path, index=False, quoting=csv.QUOTE_MINIMAL)
        return

    # Parse dates
    base_dt = base.copy()
    base_dt["StartDate"] = pd.to_datetime(base_dt["StartDate"], errors="coerce")
    base_dt["EndDate"] = pd.to_datetime(base_dt["EndDate"], errors="coerce")

    new_dt = appended_rows_df.copy()
    new_dt["StartDate"] = pd.to_datetime(new_dt["StartDate"], errors="coerce")
    new_dt["EndDate"] = pd.to_datetime(new_dt["EndDate"], errors="coerce")

    horizon_start = new_dt["StartDate"].min()
    horizon_end = new_dt["EndDate"].max()

    key_cols = ["Currency", "RoomTypeCode", "RatePlanCode", "AgeBucketCode"]
    gen_keys = set(map(tuple, new_dt[key_cols].drop_duplicates().values.tolist()))

    def should_drop(row) -> bool:
        key = (row["Currency"], row["RoomTypeCode"], row["RatePlanCode"], row["AgeBucketCode"])
        if key not in gen_keys:
            return False
        if pd.isna(row["StartDate"]) or pd.isna(row["EndDate"]):
            return False
        # overlap if: base.Start < horizon_end AND base.End > horizon_start
        return (row["StartDate"] < horizon_end) and (row["EndDate"] > horizon_start)

    drop_mask = base_dt.apply(should_drop, axis=1)
    base_kept = base.loc[~drop_mask].copy()

    out = pd.concat([base_kept, appended_rows_df], ignore_index=True)
    out.to_csv(output_csv_path, index=False, quoting=csv.QUOTE_MINIMAL)


def write_changes_csv(
    baseline_csv_path: str,
    appended_rows_df: pd.DataFrame,
    changes_csv_path: str,
    cfg: Dict[str, Any],
    as_of: dt.date,
) -> None:
    base = pd.read_csv(baseline_csv_path, dtype=str).fillna("")
    base = base[SCHEMA].copy()

    key_cols = ["RoomTypeCode", "RatePlanCode", "StartDate", "EndDate", "AgeBucketCode"]
    base_k = base.set_index(key_cols)
    new_k = appended_rows_df.set_index(key_cols)

    rows = []
    for key, new_row in new_k.iterrows():
        old_row = base_k.loc[key] if key in base_k.index else None

        for field in ["Flat", "OneGuest", "TwoGuests", "ExtraGuest"]:
            new_v = _safe_float(new_row[field])
            old_v = _safe_float(old_row[field]) if old_row is not None else None

            if new_v is None and old_v is None:
                continue

            delta = None if (new_v is None or old_v is None) else (new_v - old_v)
            delta_pct = None
            if delta is not None and old_v not in (None, 0):
                delta_pct = delta / old_v

            rows.append({
                "RunDate": as_of.isoformat(),
                "RoomTypeCode": key[0],
                "RatePlanCode": key[1],
                "StartDate": key[2],
                "EndDate": key[3],
                "AgeBucketCode": key[4],
                "Field": field,
                "Old": "" if old_v is None else int(old_v),
                "New": "" if new_v is None else int(new_v),
                "Delta": "" if delta is None else int(delta),
                "DeltaPct": "" if delta_pct is None else round(delta_pct, 4),
            })

    pd.DataFrame(rows).to_csv(changes_csv_path, index=False)


def write_summary_txt(
    multipliers_df: pd.DataFrame,
    changes_csv_path: str,
    summary_txt_path: str,
    as_of: dt.date,
) -> None:
    try:
        ch = pd.read_csv(changes_csv_path)
    except Exception:
        ch = pd.DataFrame()

    total_lines = int(len(ch)) if not ch.empty else 0
    up = int((ch["Delta"] > 0).sum()) if (not ch.empty and "Delta" in ch.columns) else 0
    down = int((ch["Delta"] < 0).sum()) if (not ch.empty and "Delta" in ch.columns) else 0

    today = as_of
    d14 = today + dt.timedelta(days=14)
    d30 = today + dt.timedelta(days=30)

    m = multipliers_df.copy()
    m["stay_date"] = pd.to_datetime(m["stay_date"]).dt.date

    w14 = m[(m["stay_date"] >= today) & (m["stay_date"] < d14)]

    def top_days(df, n=5):
        if df.empty:
            return []
        tmp = df.sort_values("final_multiplier", ascending=False).head(n)
        return [(r["stay_date"], float(r["final_multiplier"]), str(r.get("event_name", ""))) for _, r in tmp.iterrows()]

    def low_days(df, n=5):
        if df.empty:
            return []
        tmp = df.sort_values("final_multiplier", ascending=True).head(n)
        return [(r["stay_date"], float(r["final_multiplier"]), str(r.get("event_name", ""))) for _, r in tmp.iterrows()]

    peaks_14 = top_days(w14, 5)
    lows_14 = low_days(w14, 5)

    event_days = m[m.get("event_multiplier", 1.0) > 1.0].copy().sort_values("stay_date")
    next_events = event_days[event_days["stay_date"] < d30].head(12)

    lines = []
    lines.append(f"Revenue Manager Summary — {today.isoformat()}")
    lines.append("")
    lines.append(
        "I updated rates based on current on-the-books occupancy (net of expected cancellations/no-shows), "
        "plus event-related compression and group displacement where applicable."
    )
    lines.append("")

    if total_lines > 0:
        lines.append(
            f"In total, {total_lines:,} rate points changed: {up:,} increases and {down:,} decreases."
        )
    else:
        lines.append("No rate deltas were detected versus the baseline file in this run.")

    if not w14.empty:
        avg_mult_14 = float(w14["final_multiplier"].mean())
        lines.append("")
        lines.append(
            f"Over the next 14 days, the average pricing multiplier is {avg_mult_14:.2f}. "
            "Rates were raised on higher-occupancy / higher-demand dates to protect ADR, and softened on lower-occupancy "
            "dates to stimulate pickup."
        )

    if peaks_14:
        lines.append("")
        lines.append("Highest-pressure dates (next 14 days):")
        for d_, mult, evn in peaks_14:
            label = f"{d_.isoformat()} (x{mult:.2f})"
            if evn:
                label += f" — event impact: {evn}"
            lines.append(f" - {label}")

    if lows_14:
        lines.append("")
        lines.append("Softest-demand dates (next 14 days):")
        for d_, mult, _evn in lows_14:
            lines.append(f" - {d_.isoformat()} (x{mult:.2f})")

    if not next_events.empty:
        lines.append("")
        lines.append("Event-related compression in the next 30 days:")
        for _, r in next_events.iterrows():
            ev = str(r.get("event_name", "")).strip()
            if not ev:
                continue
            lines.append(
                f" - {r['stay_date'].isoformat()}: {ev} (event multiplier x{float(r.get('event_multiplier', 1.0)):.2f})"
            )

    lines.append("")
    lines.append(
        "Operationally, updates were kept within guardrails (floors/ceiling). "
        "If pickup accelerates or wash increases materially, the next run will adjust automatically."
    )
    lines.append("")

    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
