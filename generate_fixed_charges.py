#!/usr/bin/env python3
"""
Generate Fixed Charges CSV from arrivals + departures reports.

Workflow:
- Reads arrivals.csv and departures.csv from ./input_files
- Reads transaction prices + allowed schedule types from ./config/transaction_prices.csv

From ARRIVALS:
    * ConfIdent  = numeric "Name" row where Arrival looks like a date
    * ExternalId = numeric "Name" row nearby where Arrival is HOLD / PRE / OFF

From DEPARTURES:
    * AccountId (BNC-...)
    * Arrival & ETA (arrival date)
    * Departure & ETD (departure date)
    * ConfIdent column (Conf. # / Ident. #, or fallback to Name)

Rules:
- ScheduleType from transaction_prices.csv: once, daily, or both
- For ONCE:
      pick 1 valid date in stay (not past, not checkout)
- For DAILY:
      pick valid range inside stay (not past, not checkout)
- Max records = 499

Output: output_files/fixed_charges_output.csv
"""

import argparse
import os
import random
import re
from datetime import datetime, timedelta, date

import pandas as pd

MAX_RECORDS = 499


# -------------------------------------------------------
# ARGUMENTS
# -------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Fixed Charges CSV from arrivals and departures."
    )
    parser.add_argument(
        "--arrivals",
        default=os.path.join("input_files", "arrivals.csv"),
        help="Path to arrivals CSV."
    )
    parser.add_argument(
        "--departures",
        default=os.path.join("input_files", "departures.csv"),
        help="Path to departures CSV."
    )
    parser.add_argument(
        "--txfile",
        default=os.path.join("config", "transaction_prices.csv"),
        help="Path to transaction prices CSV."
    )
    parser.add_argument(
        "--output",
        default=os.path.join("output_files", "fixed_charges_output.csv"),
        help="Output CSV path."
    )
    return parser.parse_args()


# -------------------------------------------------------
# DATE HELPERS
# -------------------------------------------------------

def _to_iso_date(s: str) -> str:
    """Try multiple date formats and return ISO yyyy-mm-dd."""
    if not isinstance(s, str):
        s = str(s)
    s = s.strip()
    if not s:
        return ""

    patterns = [
        r"\d{2}/\d{2}/\d{4}",
        r"\d{2}-\d{2}-\d{4}",
        r"\d{4}-\d{2}-\d{2}",
        r"\d{2}/\d{2}/\d{2}",
        r"\d{2}-\d{2}-\d{2}",
        r"\d{2}-[A-Za-z]{3}-\d{2,4}",
    ]

    date_str = None
    for p in patterns:
        m = re.search(p, s)
        if m:
            date_str = m.group(0)
            break

    if date_str is None:
        date_str = s

    fmts = [
        "%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d",
        "%d/%m/%y", "%d-%m-%y",
        "%d-%b-%Y", "%d-%b-%y",
    ]

    for fmt in fmts:
        try:
            dt = datetime.strptime(date_str, fmt).date()
            return dt.isoformat()
        except ValueError:
            continue

    return ""


def _parse_to_date(s: str):
    iso = _to_iso_date(s)
    if not iso:
        return None
    return datetime.strptime(iso, "%Y-%m-%d").date()


def _looks_like_date(s: str) -> bool:
    return _to_iso_date(s) != ""


def _choose_once_posting_date(arrival_text, departure_text, business_date: date) -> str:
    """Pick a single valid date (not past, not checkout)."""
    arr = _parse_to_date(arrival_text)
    dep = _parse_to_date(departure_text)
    if not arr or not dep:
        return ""

    earliest = max(arr, business_date)
    latest = dep - timedelta(days=1)
    if earliest > latest:
        return ""

    offset = random.randint(0, (latest - earliest).days)
    return (earliest + timedelta(days=offset)).isoformat()


def _choose_daily_period(arrival_text, departure_text, business_date: date):
    """Pick FROM and TO for daily schedule, inside valid stay."""
    arr = _parse_to_date(arrival_text)
    dep = _parse_to_date(departure_text)
    if not arr or not dep:
        return "", ""

    earliest = max(arr, business_date)
    latest = dep - timedelta(days=1)
    if earliest > latest:
        return "", ""

    start_offset = random.randint(0, (latest - earliest).days)
    start = earliest + timedelta(days=start_offset)
    remaining = (latest - start).days
    end_offset = random.randint(0, remaining) if remaining > 0 else 0
    end = start + timedelta(days=end_offset)
    return start.isoformat(), end.isoformat()


# -------------------------------------------------------
# OTHER HELPERS
# -------------------------------------------------------

def _is_digits(s: str):
    if not isinstance(s, str):
        s = str(s)
    return s.strip().isdigit()


def _find_column(df: pd.DataFrame, candidates, context: str) -> str:
    """
    Flexible column detection: lowercased, trimmed.
    Returns the actual column name from df.
    """
    normalized = {col.strip().lower(): col for col in df.columns}

    for cand in candidates:
        key = cand.strip().lower()
        if key in normalized:
            return normalized[key]

    raise ValueError(
        f"Departures file must contain one of: {candidates} for {context}.\n"
        f"Found columns: {list(df.columns)}"
    )


def _normalize_id_str(v) -> str:
    """
    Normalize ConfIdent-like values so 30319.0 -> '30319', 30319 -> '30319', '30319' -> '30319'.
    """
    if pd.isna(v):
        return ""
    s = str(v).strip()
    if not s:
        return ""
    if "." in s:
        left, right = s.split(".", 1)
        if left.isdigit() and set(right) <= {"0"}:
            s = left
    return s


def load_transaction_prices(path):
    df = pd.read_csv(path)

    if "TransactionCode" not in df.columns or "UnitPrice" not in df.columns:
        raise ValueError("transaction_prices.csv missing required columns TransactionCode and UnitPrice.")

    prices = {}
    schedule_modes = {}

    for _, row in df.iterrows():
        code = str(row["TransactionCode"]).strip()
        if not code:
            continue

        try:
            price = float(row["UnitPrice"])
        except Exception:
            continue

        prices[code] = price

        raw = str(row.get("ScheduleType", "once")).strip().lower()
        if raw == "daily":
            schedule_modes[code] = {"daily"}
        elif raw == "both":
            schedule_modes[code] = {"once", "daily"}
        else:
            schedule_modes[code] = {"once"}

    if not prices:
        raise ValueError("No valid transaction prices loaded from transaction_prices.csv.")

    return prices, schedule_modes


# -------------------------------------------------------
# ARRIVAL → ConfIdent / ExternalId
# -------------------------------------------------------

def extract_arrival_ids(arrivals_df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    From arrivals:
    - ConfIdent = numeric Name where Arrival looks like a date
    - ExternalId = numeric Name nearby where Arrival is HOLD/PRE/OFF
    """
    df = arrivals_df.copy()

    if "Name" not in df.columns or "Arrival" not in df.columns:
        raise ValueError("Arrivals file must contain Name and Arrival columns.")

    df["idx"] = df.index
    numeric = df[df["Name"].astype(str).apply(_is_digits)]

    conf_rows = numeric[numeric["Arrival"].apply(_looks_like_date)]
    ext_rows = numeric[numeric["Arrival"].isin(["HOLD", "PRE", "OFF"])]

    if conf_rows.empty or ext_rows.empty:
        raise ValueError("Could not detect ConfIdent and ExternalId rows from arrivals.")

    pairs = []
    for _, ext in ext_rows.iterrows():
        idx = ext["idx"]
        nearby = conf_rows[(conf_rows["idx"] >= idx - window) & (conf_rows["idx"] <= idx + window)]
        if nearby.empty:
            continue
        nearest = nearby.iloc[(nearby["idx"] - idx).abs().argsort().iloc[0]]
        conf_id = str(nearest["Name"]).strip()
        ext_id = str(ext["Name"]).strip()
        pairs.append((conf_id, ext_id))

    df_pairs = pd.DataFrame(pairs, columns=["ConfIdent", "ExternalId"])
    df_pairs["ConfIdent"] = df_pairs["ConfIdent"].apply(_normalize_id_str)
    df_pairs = df_pairs.drop_duplicates(subset=["ConfIdent"])
    return df_pairs


# -------------------------------------------------------
# MAIN BUILD FUNCTION
# -------------------------------------------------------

def build_fixed_charges(
    arrivals_df: pd.DataFrame,
    departures_df: pd.DataFrame,
    transaction_prices: dict,
    schedule_modes: dict,
) -> pd.DataFrame:
    business_date = date.today()

    # 1) ConfIdent / ExternalId from arrivals
    arr_ids = extract_arrival_ids(arrivals_df)

    # 2) Departures columns
    dep = departures_df.copy()

    conf_col = _find_column(
        dep,
        [
            "Conf. # / Ident. #",
            "Conf#/Ident#",
            "Confirmation",
            "Conf No",
            "Confirmation #",
            "Reservation #",
            "Name",  # fallback
        ],
        "ConfIdent",
    )
    acc_col = _find_column(
        dep,
        ["Account #", "Acc. #", "Acc #", "Account", "Account No", "Account No."],
        "AccountId",
    )
    arr_col = _find_column(
        dep,
        ["Arrival & ETA", "Arr. Date & Time", "Arrival Date & Time", "Arrival Date", "Arrival"],
        "ArrivalDate",
    )
    dep_col = _find_column(
        dep,
        ["Departure & ETD", "Dep. Date & Time", "Departure Date & Time", "Departure Date", "Departure"],
        "DepartureDate",
    )

    dep_small = dep[[conf_col, acc_col, arr_col, dep_col]].copy()
    dep_small = dep_small.rename(
        columns={
            conf_col: "ConfIdent",
            acc_col: "AccountId",
            arr_col: "Arr",
            dep_col: "Dep",
        }
    )

    # Normalize IDs before joining
    dep_small["ConfIdent"] = dep_small["ConfIdent"].apply(_normalize_id_str)
    arr_ids["ConfIdent"] = arr_ids["ConfIdent"].apply(_normalize_id_str)

    merged = arr_ids.merge(dep_small, on="ConfIdent", how="inner")

    # Only BNC accounts
    merged = merged[merged["AccountId"].astype(str).str.startswith("BNC")]

    records = []
    codes = list(transaction_prices.keys())

    for _, row in merged.iterrows():
        if len(records) >= MAX_RECORDS:
            break

        external_id = row["ExternalId"]
        account_id = row["AccountId"]
        arr_text = row["Arr"]
        dep_text = row["Dep"]

        tx_code = random.choice(codes)
        price = transaction_prices[tx_code]
        allowed = schedule_modes.get(tx_code, {"once"})

        if allowed == {"once", "daily"}:
            schedule = random.choice(["Once", "Daily"])
        elif allowed == {"daily"}:
            schedule = "Daily"
        else:
            schedule = "Once"

        if schedule == "Once":
            post = _choose_once_posting_date(arr_text, dep_text, business_date)
            if not post:
                continue
            from_date = post
            to_date = post
        else:  # Daily
            from_date, to_date = _choose_daily_period(arr_text, dep_text, business_date)
            if not from_date:
                continue

        records.append(
            {
                "ExternalSystemCode": "PMS",
                "ExternalId": external_id,
                "AccountId": account_id,
                "TransactionCode": tx_code,
                "Quantity": 1,
                "UnitPrice": price,
                "Comment": "Migration",
                "ScheduleType": schedule,
                "FromDate": from_date,
                "ToDate": to_date,
                "CustomPostingDates": "",
                "DayOfMonth": "",
                "MonthsBetweenPostings": "",
                "DaysOfWeek": "",
                "OrdinalOccurenceOfDayOfWeek": "",
            }
        )

    return pd.DataFrame(records)


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------

def main():
    args = parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    if not os.path.exists(args.arrivals):
        raise FileNotFoundError(f"Arrivals file not found: {args.arrivals}")
    if not os.path.exists(args.departures):
        raise FileNotFoundError(f"Departures file not found: {args.departures}")

    arrivals = pd.read_csv(args.arrivals)
    departures = pd.read_csv(args.departures)

    tx_prices, schedule_modes = load_transaction_prices(args.txfile)

    fixed_df = build_fixed_charges(arrivals, departures, tx_prices, schedule_modes)

    fixed_df.to_csv(args.output, index=False)
    print(f"Generated: {args.output}")
    print(f"Rows: {len(fixed_df)}")


if __name__ == "__main__":
    main()
