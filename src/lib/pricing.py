"""
Pricing engine.

Responsibilities:
- Load pricing template CSV (10-column schema)
- Compute final rate multipliers per stay date
- Generate BAR00 rows for horizon (with PRE flat-only handling)
- Generate derived plans:
  BAR00, BAREX, RACK, CORL25, OPQ, GRPBASE, GRPHIG, STAFF, WAL, HB, SUHB, WEL, BB, WEL3 (weekends-only),
  BARAPT (A1KB/A2KB only), BARSUIT (KCST/KGST/PRE only, PRE flat-only)
- Write:
  - pricing output CSV (overwrite existing file path, keeping header)
  - changes CSV showing old vs new + reasons

Notes:
- EndDate is exclusive.
- Rounding: whole EUR via normal rounding (0.5 rounds up).
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
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return int(math.floor(float(x) + 0.5))


def _is_weekend(d: dt.date) -> bool:
    return d.weekday() >= 5  # Sat/Sun


def _parse_date(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


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
        return float(t)
    return float(v)


def _to_str_or_blank(v: Optional[int]) -> str:
    return "" if v is None else str(int(v))


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# --------------------------
# Template loading
# --------------------------

def load_pricing_template(path: str) -> pd.DataFrame:
    """
    Load template pricing CSV and normalize types.
    Keeps blanks as empty strings in numeric fields where needed later.
    """
    df = pd.read_csv(path, dtype=str).fillna("")
    missing = [c for c in SCHEMA if c not in df.columns]
    if missing:
        raise ValueError(f"Template missing required columns: {missing}")

    # Normalize dates
    df["StartDate"] = pd.to_datetime(df["StartDate"]).dt.date
    df["EndDate"] = pd.to_datetime(df["EndDate"]).dt.date

    # Normalize AgeBucketCode blanks to ""
    df["AgeBucketCode"] = df["AgeBucketCode"].fillna("")

    return df[SCHEMA].copy()


def _latest_by_key(df: pd.DataFrame, rateplan: str) -> pd.DataFrame:
    """
    Latest rows per (RoomTypeCode, AgeBucketCode) for a rateplan.
    """
    sub = df[df["RatePlanCode"] == rateplan].copy()
    if sub.empty:
        return sub
    sub = sub.sort_values(["StartDate", "EndDate"])
    # group key includes AgeBucketCode (blank allowed)
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
            v_plan = _safe_float(r[f"{col}_plan"])
            v_bar = _safe_float(r[f"{col}_bar"])
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
    Combine:
      - occupancy thresholds
      - pickup adjustment (v1: disabled unless we add snapshots later; kept for extension)
      - wash adjustment (v1: small reduction if unusually high; optional)
      - groups displacement
      - event multipliers
      - guardrails (caps only here; per-run change caps are applied when we have baseline published rates)

    Returns DataFrame with:
      stay_date, final_multiplier, reason, event_name
    """
    df = daily_metrics_df.copy()

    # Merge event multipliers
    ev = event_mult_df.copy()
    df = df.merge(ev, on="stay_date", how="left")
    df["event_multiplier"] = df["event_multiplier"].fillna(1.0)
    df["event_name"] = df["event_name"].fillna("")

    # Base multiplier from occupancy thresholds
    thresholds = cfg["occupancy_thresholds"]
    def base_from_occ(occ: float) -> float:
        for t in thresholds:
            if float(t["min"]) <= occ < float(t["max"]):
                return float(t["multiplier"])
        # If >=1.0, treat as top bucket
        return float(thresholds[-1]["multiplier"])

    df["base_occ_multiplier"] = df["forecast_occupancy"].apply(base_from_occ)

    # Pickup adjustment placeholder (v1: 1.0 always unless you add snapshot logic)
    df["pickup_multiplier"] = 1.0

    # Wash multiplier: small reduction only if wash is high (v1: optional simple rule)
    # If expected wash / net_otb > 0.12 => 0.97 else 1.00
    def wash_mult(row) -> float:
        net_otb = float(row.get("net_otb", 0.0))
        expected_wash = float(row.get("expected_wash", 0.0))
        if net_otb <= 0:
            return 1.0
        wash_rate = expected_wash / net_otb
        if wash_rate > 0.12:
            return 0.97
        return 1.0

    df["wash_multiplier"] = df.apply(wash_mult, axis=1)

    # Groups displacement
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

    # Combine
    df["raw_multiplier"] = (
        df["base_occ_multiplier"]
        * df["pickup_multiplier"]
        * df["wash_multiplier"]
        * df["group_multiplier"]
        * df["event_multiplier"]
    )

    # Floors/ceilings (simple seasonal floor; event floor handled downstream with event_name)
    guard = cfg["guardrails"]
    ceiling = float(guard["ceiling"])

    def floor_for_date(d: dt.date) -> float:
        # Summer floor (Jul–Aug)
        if d.month in (7, 8):
            return float(guard["floors"]["summer"])
        # Low season floor (mid-Jan and late Nov approximations)
        if (d.month == 1 and d.day >= 7) or (d.month == 11 and d.day >= 10):
            return float(guard["floors"]["low_season"])
        return float(guard["floors"]["low_season"])  # default safe floor; tightened below
    # Use a general minimum floor of low_season to avoid 0.
    df["floor"] = df["stay_date"].apply(floor_for_date)

    # Compression events floor (if event is Peak, enforce compression_event floor)
    compression_floor = float(guard["floors"]["compression_event"])
    # Determine if event level implies compression: we don't store demand_level in df; if multiplier >= Peak weight, treat as compression
    peak_weight = float(cfg["events"]["weights"].get("Peak", 1.20))
    df["floor"] = df.apply(
        lambda r: max(float(r["floor"]), compression_floor) if float(r["event_multiplier"]) >= peak_weight else float(r["floor"]),
        axis=1,
    )

    df["final_multiplier"] = df.apply(
        lambda r: _clamp(float(r["raw_multiplier"]), float(r["floor"]), ceiling),
        axis=1,
    )

    # Reason string (compact, used in change log)
    def reason(row) -> str:
        parts = [
            f"occ={row['forecast_occupancy']:.2f}->{row['base_occ_multiplier']:.2f}",
        ]
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
# Row generation
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
    (Daily rows keep it simple and avoids complex band splitting; you can optimize later.)

    We use template's latest BAR00 as baseline and ratios for HB/WEL/GRPBASE.
    """
    # Get latest baseline rows
    bar_latest = _latest_by_key(template_df, "BAR00")
    if bar_latest.empty:
        raise ValueError("Template has no BAR00 rows; cannot build any pricing.")

    hb_latest = _latest_by_key(template_df, "HB")
    wel_latest = _latest_by_key(template_df, "WEL")
    grp_latest = _latest_by_key(template_df, "GRPBASE")

    hb_ratios = _compute_ratio_dict(hb_latest, bar_latest) if not hb_latest.empty else {}
    wel_ratios = _compute_ratio_dict(wel_latest, bar_latest) if not wel_latest.empty else {}
    grp_ratios = _compute_ratio_dict(grp_latest, bar_latest) if not grp_latest.empty else {}

    # Build baseline BAR00 dict
    # Key is (RoomTypeCode, AgeBucketCode)
    baseline: Dict[Tuple[str, str], Dict[str, Optional[float]]] = {}
    for _, r in bar_latest.iterrows():
        key = (r["RoomTypeCode"], r["AgeBucketCode"])
        baseline[key] = {
            "Flat": _safe_float(r["Flat"]),
            "OneGuest": _safe_float(r["OneGuest"]),
            "TwoGuests": _safe_float(r["TwoGuests"]),
            "ExtraGuest": _safe_float(r["ExtraGuest"]),
        }

    # Multipliers keyed by date
    mult_map = {row["stay_date"]: (float(row["final_multiplier"]), str(row["reason"])) for _, row in multipliers_df.iterrows()}

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

    # Iterate each date as 1-night period
    d = start_date
    while d < end_date:
        mult, _reason = mult_map.get(d, (1.0, ""))
        next_d = d + dt.timedelta(days=1)

        # Build BAR00 per key
        bar_rows_per_key: Dict[Tuple[str, str], Tuple[str, str, str]] = {}  # (rt,ab)->(og,tg,eg) str
        pre_flat: Optional[str] = None

        for (rt, ab), base_vals in baseline.items():
            currency = "EUR"

            if rt == "PRE" and ab == "":
                # Flat-only handling
                base_flat = base_vals.get("Flat")
                if base_flat is None:
                    continue
                flat = _to_str_or_blank(_rnd(base_flat * mult))
                pre_flat = flat
                add_row(currency, rt, "BAR00", d, next_d, "", flat, "", "", "")
                continue

            if ab == "":
                # Non-PRE rows should be per-person (skip if no age bucket)
                continue

            base_og = base_vals.get("OneGuest")
            base_tg = base_vals.get("TwoGuests")
            base_eg = base_vals.get("ExtraGuest")
            if base_og is None or base_tg is None or base_eg is None:
                continue

            og = _to_str_or_blank(_rnd(base_og * mult))
            tg = _to_str_or_blank(_rnd(base_tg * mult))
            eg = _to_str_or_blank(_rnd(base_eg * mult))

            add_row(currency, rt, "BAR00", d, next_d, ab, "", og, tg, eg)
            bar_rows_per_key[(rt, ab)] = (og, tg, eg)

        # Derived plans from BAR00
        def derive_simple(plan: str, factor: float, room_filter=None):
            for (rt, ab), (og, tg, eg) in bar_rows_per_key.items():
                if room_filter and rt not in room_filter:
                    continue
                add_row("EUR", rt, plan, d, next_d, ab, "", _to_str_or_blank(_rnd(float(og) * factor)),
                        _to_str_or_blank(_rnd(float(tg) * factor)), _to_str_or_blank(_rnd(float(eg) * factor)))

            # PRE flat if present and plan should apply to PRE
            if pre_flat is not None and (room_filter is None or "PRE" in room_filter):
                add_row("EUR", "PRE", plan, d, next_d, "", _to_str_or_blank(_rnd(float(pre_flat) * factor)), "", "", "")

        # Excluded plans are not generated here by design

        derive_simple("BAREX", 1.25)
        derive_simple("RACK", 1.35)
        derive_simple("CORL25", 0.75)
        derive_simple("OPQ", 0.80)
        derive_simple("STAFF", 0.50)
        derive_simple("WAL", 0.85)

        # BB = BAR00 + 35 per person (A1 and C1 rows)
        for (rt, ab), (og, tg, eg) in bar_rows_per_key.items():
            add_row("EUR", rt, "BB", d, next_d, ab, "",
                    _to_str_or_blank(_rnd(float(og) + 35)),
                    _to_str_or_blank(_rnd(float(tg) + 35)),
                    _to_str_or_blank(_rnd(float(eg) + 35)))

        # WEL3 weekends-only: Fri and Sat nights (stay_date Fri or Sat)
        if d.weekday() in (4, 5):  # Fri, Sat
            for (rt, ab), (og, tg, eg) in bar_rows_per_key.items():
                add_row("EUR", rt, "WEL3", d, next_d, ab, "",
                        _to_str_or_blank(_rnd(float(og) * 1.35)),
                        _to_str_or_blank(_rnd(float(tg) * 1.35)),
                        _to_str_or_blank(_rnd(float(eg) * 1.35)))

        # Plans that preserve template structure via ratios vs BAR00
        def apply_ratio_plan(plan: str, ratios: Dict[Tuple[str, str], Dict[str, float]]):
            for (rt, ab), (og, tg, eg) in bar_rows_per_key.items():
                ratio = ratios.get((rt, ab))
                if not ratio:
                    continue
                add_row("EUR", rt, plan, d, next_d, ab, "",
                        _to_str_or_blank(_rnd(float(og) * ratio.get("OneGuest", 1.0))),
                        _to_str_or_blank(_rnd(float(tg) * ratio.get("TwoGuests", 1.0))),
                        _to_str_or_blank(_rnd(float(eg) * ratio.get("ExtraGuest", 1.0))))
            # PRE is not generated for these unless template had it per-person (we keep PRE flat-only)
        apply_ratio_plan("HB", hb_ratios)
        apply_ratio_plan("WEL", wel_ratios)
        apply_ratio_plan("GRPBASE", grp_ratios)

        # GRPHIG = GRPBASE * 1.25
        # We need GRPBASE rows just generated; easiest is to derive again from BAR00 using ratio then multiply.
        for (rt, ab), (og, tg, eg) in bar_rows_per_key.items():
            ratio = grp_ratios.get((rt, ab))
            if not ratio:
                continue
            g_og = _rnd(float(og) * ratio.get("OneGuest", 1.0))
            g_tg = _rnd(float(tg) * ratio.get("TwoGuests", 1.0))
            g_eg = _rnd(float(eg) * ratio.get("ExtraGuest", 1.0))
            add_row("EUR", rt, "GRPHIG", d, next_d, ab, "",
                    _to_str_or_blank(_rnd(g_og * 1.25 if g_og is not None else None)),
                    _to_str_or_blank(_rnd(g_tg * 1.25 if g_tg is not None else None)),
                    _to_str_or_blank(_rnd(g_eg * 1.25 if g_eg is not None else None)))

        # SUHB = HB * 1.25 (using HB ratios then x1.25)
        for (rt, ab), (og, tg, eg) in bar_rows_per_key.items():
            ratio = hb_ratios.get((rt, ab))
            if not ratio:
                continue
            h_og = _rnd(float(og) * ratio.get("OneGuest", 1.0))
            h_tg = _rnd(float(tg) * ratio.get("TwoGuests", 1.0))
            h_eg = _rnd(float(eg) * ratio.get("ExtraGuest", 1.0))
            add_row("EUR", rt, "SUHB", d, next_d, ab, "",
                    _to_str_or_blank(_rnd(h_og * 1.25 if h_og is not None else None)),
                    _to_str_or_blank(_rnd(h_tg * 1.25 if h_tg is not None else None)),
                    _to_str_or_blank(_rnd(h_eg * 1.25 if h_eg is not None else None)))

        # BARAPT = BAR00 * 1.40 (A1KB/A2KB only)
        barapt_factor = float(cfg["rate_plans"]["BARAPT_multiplier"])
        derive_simple("BARAPT", barapt_factor, room_filter={"A1KB", "A2KB"})

        # BARSUIT = BAR00 * 1.25 (KCST/KGST/PRE only; PRE flat-only)
        barsuit_factor = float(cfg["rate_plans"]["BARSUIT_multiplier"])
        derive_simple("BARSUIT", barsuit_factor, room_filter={"KCST", "KGST", "PRE"})

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
    Overwrite the output pricing file with:
      - header
      - baseline rows (unchanged)
      - appended rows

    This matches your "do not touch existing rows; append only" rule in the workflow,
    but since you asked to overwrite the file content with same name/path, we write a fresh file containing:
      baseline + appended
    """
    base = pd.read_csv(baseline_csv_path, dtype=str).fillna("")
    base = base[SCHEMA].copy()

    out = pd.concat([base, appended_rows_df], ignore_index=True)
    out.to_csv(output_csv_path, index=False, quoting=csv.QUOTE_MINIMAL)


def write_changes_csv(
    baseline_csv_path: str,
    appended_rows_df: pd.DataFrame,
    changes_csv_path: str,
    cfg: Dict[str, Any],
    as_of: dt.date,
) -> None:
    """
    Create a compact changes file.
    We compare baseline vs appended for the overlapping keys:
      (RoomTypeCode, RatePlanCode, StartDate, EndDate, AgeBucketCode)

    For each numeric field, we report old/new/delta/deltaPct plus a simple reason placeholder.
    """
    base = pd.read_csv(baseline_csv_path, dtype=str).fillna("")
    base = base[SCHEMA].copy()

    # Key
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
                "Reason": "",  # filled later if you pass reasons through the pipeline
            })

    pd.DataFrame(rows).to_csv(changes_csv_path, index=False)


# Optional helper if you want strict 10-column output formatting later
def write_csv_strict_10cols(path: str, df: pd.DataFrame) -> None:
    """
    Write CSV ensuring exactly 10 columns and preserving blanks.
    """
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(SCHEMA)
        for _, r in df.iterrows():
            w.writerow([r.get(c, "") for c in SCHEMA])
