# /src/lib/pricing.py
"""
Pricing engine (stable API expected by run_pricing.py)

Exports:
- load_pricing_template
- compute_multipliers
- generate_pricing_rows
- write_pricing_csv
- write_changes_csv
- write_summary_txt
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
# Template loading / ratios
# --------------------------

def load_pricing_template(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")
    missing = [c for c in SCHEMA if c not in df.columns]
    if missing:
        raise ValueError(f"Template missing required columns: {missing}")

    df["StartDate"] = pd.to_datetime(df["StartDate"], errors="coerce").dt.date
    df["EndDate"] = pd.to_datetime(df["EndDate"], errors="coerce").dt.date
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
# Multipliers
# --------------------------

def compute_multipliers(
    daily_metrics_df: pd.DataFrame,
    event_mult_df: pd.DataFrame,
    cfg: Dict[str, Any],
    as_of: dt.date,
) -> pd.DataFrame:
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

    df["pickup_multiplier"] = 1.0
    df["wash_multiplier"] = 1.0
    df["group_multiplier"] = 1.0

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

    df["final_multiplier"] = df["raw_multiplier"].apply(lambda x: _clamp(float(x), low_floor, ceiling))
    df["reason"] = df.apply(lambda r: f"occ={r['forecast_occupancy']:.2f}->{r['base_occ_multiplier']:.2f}", axis=1)

    return df[["stay_date", "final_multiplier", "reason", "event_name", "event_multiplier"]].copy()


# --------------------------
# Pricing rows (daily)
# --------------------------

def generate_pricing_rows(
    template_df: pd.DataFrame,
    multipliers_df: pd.DataFrame,
    cfg: Dict[str, Any],
    start_date: dt.date,
    end_date: dt.date,
) -> pd.DataFrame:
    bar_latest = _latest_by_key(template_df, "BAR00")
    if bar_latest.empty:
        raise ValueError("Template has no BAR00 rows; cannot build pricing.")

    hb_latest = _latest_by_key(template_df, "HB")
    wel_latest = _latest_by_key(template_df, "WEL")
    grp_latest = _latest_by_key(template_df, "GRPBASE")

    hb_ratios = _compute_ratio_dict(hb_latest, bar_latest) if not hb_latest.empty else {}
    wel_ratios = _compute_ratio_dict(wel_latest, bar_latest) if not wel_latest.empty else {}
    grp_ratios = _compute_ratio_dict(grp_latest, bar_latest) if not grp_latest.empty else {}

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

    def add_row(rt, plan, sd, ed, ab, flat, og, tg, eg, currency="EUR"):
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

        bar_pp_per_key: Dict[Tuple[str, str], Tuple[str, str, str]] = {}
        bar_flat_by_room: Dict[str, str] = {}

        for (rt, ab), base_vals in baseline.items():
            if rt == "PRE" and ab == "":
                base_flat = base_vals.get("Flat")
                if base_flat is not None:
                    flat = _to_str_or_blank(_rnd(base_flat * mult))
                    bar_flat_by_room["PRE"] = flat
                    add_row("PRE", "BAR00", d, next_d, "", flat, "", "", "")
                continue

            if ab != "":
                base_og = base_vals.get("OneGuest")
                base_tg = base_vals.get("TwoGuests")
                base_eg = base_vals.get("ExtraGuest")
                if base_og is None or base_tg is None or base_eg is None:
                    continue

                og = _to_str_or_blank(_rnd(base_og * mult))
                tg = _to_str_or_blank(_rnd(base_tg * mult))
                eg = _to_str_or_blank(_rnd(base_eg * mult))
                bar_pp_per_key[(rt, ab)] = (og, tg, eg)

                if ab == "A1":
                    bar_flat_by_room[rt] = tg

        # BAR00 flat-only output (non-PRE)
        for rt, flat in bar_flat_by_room.items():
            if rt == "PRE":
                continue
            add_row(rt, "BAR00", d, next_d, "", flat, "", "", "")

        # Flat-only derived
        def add_flat_plan(plan: str, factor: float, room_filter=None):
            for rt, bar_flat in bar_flat_by_room.items():
                if room_filter and rt not in room_filter:
                    continue
                new_flat = _to_str_or_blank(_rnd(float(bar_flat) * factor))
                add_row(rt, plan, d, next_d, "", new_flat, "", "", "")

        add_flat_plan("BAREX", 1.25)
        add_flat_plan("RACK", 1.35)
        add_flat_plan("CORL25", 0.75)
        add_flat_plan("WAL", 0.85)
        add_flat_plan("BARAPT", float(cfg["rate_plans"]["BARAPT_multiplier"]), room_filter={"A1KB", "A2KB"})
        add_flat_plan("BARSUIT", float(cfg["rate_plans"]["BARSUIT_multiplier"]), room_filter={"KCST", "KGST", "PRE"})

        # Per-person simple
        def derive_pp_simple(plan: str, factor: float):
            for (rt, ab), (og, tg, eg) in bar_pp_per_key.items():
                add_row(
                    rt, plan, d, next_d, ab, "",
                    _to_str_or_blank(_rnd(float(og) * factor)),
                    _to_str_or_blank(_rnd(float(tg) * factor)),
                    _to_str_or_blank(_rnd(float(eg) * factor)),
                )

        derive_pp_simple("OPQ", 0.80)
        derive_pp_simple("STAFF", 0.50)

        # BB (+35)
        for (rt, ab), (og, tg, eg) in bar_pp_per_key.items():
            add_row(
                rt, "BB", d, next_d, ab, "",
                _to_str_or_blank(_rnd(float(og) + 35)),
                _to_str_or_blank(_rnd(float(tg) + 35)),
                _to_str_or_blank(_rnd(float(eg) + 35)),
            )

        # WEL3 weekends (Fri/Sat)
        if d.weekday() in (4, 5):
            for (rt, ab), (og, tg, eg) in bar_pp_per_key.items():
                add_row(
                    rt, "WEL3", d, next_d, ab, "",
                    _to_str_or_blank(_rnd(float(og) * 1.35)),
                    _to_str_or_blank(_rnd(float(tg) * 1.35)),
                    _to_str_or_blank(_rnd(float(eg) * 1.35)),
                )

        # Ratio-based per-person plans
        def apply_ratio_plan(plan: str, ratios: Dict[Tuple[str, str], Dict[str, float]]):
            for (rt, ab), (og, tg, eg) in bar_pp_per_key.items():
                ratio = ratios.get((rt, ab))
                if not ratio:
                    continue
                add_row(
                    rt, plan, d, next_d, ab, "",
                    _to_str_or_blank(_rnd(float(og) * ratio.get("OneGuest", 1.0))),
                    _to_str_or_blank(_rnd(float(tg) * ratio.get("TwoGuests", 1.0))),
                    _to_str_or_blank(_rnd(float(eg) * ratio.get("ExtraGuest", 1.0))),
                )

        apply_ratio_plan("HB", hb_ratios)
        apply_ratio_plan("WEL", wel_ratios)
        apply_ratio_plan("GRPBASE", grp_ratios)

        # SUHB = HB * 1.25
        for (rt, ab), (og, tg, eg) in bar_pp_per_key.items():
            ratio = hb_ratios.get((rt, ab))
            if not ratio:
                continue
            h_og = _rnd(float(og) * ratio.get("OneGuest", 1.0))
            h_tg = _rnd(float(tg) * ratio.get("TwoGuests", 1.0))
            h_eg = _rnd(float(eg) * ratio.get("ExtraGuest", 1.0))
            add_row(
                rt, "SUHB", d, next_d, ab, "",
                _to_str_or_blank(_rnd(h_og * 1.25 if h_og is not None else None)),
                _to_str_or_blank(_rnd(h_tg * 1.25 if h_tg is not None else None)),
                _to_str_or_blank(_rnd(h_eg * 1.25 if h_eg is not None else None)),
            )

        # GRPHIG flat-only from BAR00 flat using GRPBASE ratio (fallback 0.70)
        for rt, bar_flat in bar_flat_by_room.items():
            r = grp_ratios.get((rt, "A1"), {})
            grpbase_factor = float(r.get("TwoGuests", 0.70))
            grphig_flat = _to_str_or_blank(_rnd(float(bar_flat) * grpbase_factor * 1.25))
            add_row(rt, "GRPHIG", d, next_d, "", grphig_flat, "", "", "")

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
    base = pd.read_csv(baseline_csv_path, dtype=str).fillna("")
    base = base[SCHEMA].copy()

    if appended_rows_df is None or appended_rows_df.empty:
        base.to_csv(output_csv_path, index=False, quoting=csv.QUOTE_MINIMAL)
        return

    # Currency normalization (template may have blanks)
    base["Currency"] = base["Currency"].replace("", "EUR")
    new = appended_rows_df.copy()
    new["Currency"] = new["Currency"].replace("", "EUR")

    base_dt = base.copy()
    base_dt["StartDate"] = pd.to_datetime(base_dt["StartDate"], errors="coerce")
    base_dt["EndDate"] = pd.to_datetime(base_dt["EndDate"], errors="coerce")

    new_dt = new.copy()
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
        return (row["StartDate"] < horizon_end) and (row["EndDate"] > horizon_start)

    drop_mask = base_dt.apply(should_drop, axis=1)
    base_kept = base.loc[~drop_mask].copy()

    out = pd.concat([base_kept, new], ignore_index=True)
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

    if appended_rows_df is None or appended_rows_df.empty:
        pd.DataFrame([]).to_csv(changes_csv_path, index=False)
        return

    key_cols = ["RoomTypeCode", "RatePlanCode", "StartDate", "EndDate", "AgeBucketCode"]
    base_k = base.set_index(key_cols)
    new_k = appended_rows_df.set_index(key_cols)

    rows_out = []
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

            rows_out.append({
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

    pd.DataFrame(rows_out).to_csv(changes_csv_path, index=False)


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

    m = multipliers_df.copy()
    m["stay_date"] = pd.to_datetime(m["stay_date"]).dt.date
    w14 = m[(m["stay_date"] >= today) & (m["stay_date"] < d14)]

    lines = []
    lines.append(f"Revenue Manager Summary — {today.isoformat()}")
    lines.append("")
    lines.append(
        "Rates were adjusted using forecast occupancy and event weighting. "
        "Cancellations/no-shows are not directly observed in this build unless provided as separate reports."
    )
    lines.append("")

    # Delta counts may be zero if rows are new (no baseline match)
    lines.append(f"In total, {total_lines:,} rate points generated/compared: {up:,} increases and {down:,} decreases.")
    if not w14.empty:
        avg_mult_14 = float(w14["final_multiplier"].mean())
        lines.append("")
        lines.append(f"Over the next 14 days, the average pricing multiplier is {avg_mult_14:.2f}.")

    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
