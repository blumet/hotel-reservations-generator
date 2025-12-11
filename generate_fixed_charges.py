#!/usr/bin/env python3
"""
Generate Fixed Charges CSV from arrivals + departures reports.

Workflow:
- Reads arrivals.csv and departures.csv from ./input_files
- Reads transaction prices + allowed schedule types from ./config/transaction_prices.csv

From ARRIVALS:
    * ConfIdent  = numeric "Name" row where Arrival looks like a date
                   (short ID like 28726)
    * ExternalId = numeric "Name" row nearby where Arrival is HOLD / PRE / OFF
                   (long ID like 5006885)

From DEPARTURES:
    * AccountId (BNC-...)
    * Arrival / Departure dates (column names detected flexibly, e.g. "Arrival",
      "Arrival & ETA", "Departure & ETD", etc.)
    * ConfIdent column detected flexibly (including "Name" as fallback)

For each matched reservation (max 499):
    * Choose a TransactionCode at random
    * For that code, choose ScheduleType from its allowed modes:
         - once  -> Only "Once"
         - daily -> Only "Daily"
         - both  -> Randomly "Once" or "Daily"
    * Choose posting dates such that:
         - Dates are on/after business date (today)
         - Dates are within the stay
         - Dates never include checkout day

    * If ScheduleType == "Once":
         FromDate = ToDate = one valid posting date
      If ScheduleType == "Daily":
         FromDate = range_start, ToDate = range_end

Writes: output_files/fixed_charges_output.csv
"""

import argparse
import os
import random
import re
from datetime import datetime, timedelta, date

import pandas as pd

# Maximum number of records in the generated CSV
MAX_RECORDS = 499


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Fixed Charges CSV from arrivals and departures."
    )
    parser.add_argument(
        "--arrivals",
        default=os.path.join("input_files", "arrivals.csv"),
        help="Path to arrivals CSV (export from PMS). Default: input_files/arrivals.csv",
    )
    parser.add_argument(
        "--departures",
        default=os.path.join("input_files", "departures.csv"),
        help="Path to departures CSV (export from PMS). Default: input_files/departures.csv",
    )
    parser.add_argument(
        "--txfile",
        default=os.path.join("config", "transaction_prices.csv"),
        help="Path to transaction prices CSV. Default: config/transaction_prices.csv",
    )
    parser.add_argument(
        "--output",
        default=os.path.join("output_files", "fixed_charges_output.csv"),
        help="Path for generated fixed charges CSV. Default: output_files/fixed_charges_output.csv",
    )
    return parser.parse_args()


# ----------------------------
# DATE HELPERS
# ----------------------------

def _to_iso_date(s: str) -> str:
    """
    Try to parse a date from the input string in several common formats
    and return it as ISO yyyy-mm-dd. Returns '' if parsing fails.
    """
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
    for pat in patterns:
        m = re.search(pat, s)
        if m:
            date_str = m.group(0)
            break

    if date_str is None:
        date_str = s

    formats = [
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%Y-%m-%d",
        "%d/%m/%y",
        "%d-%m-%y",
        "%d-%b-%Y",
        "%d-%b-%y",
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(date_str, fmt).date()
            return dt.isoformat()  # always yyyy-mm-dd
        except ValueError:
            continue

    return ""


def _parse_to_date(s: str):
    """Return a datetime.date parsed from s, or None if not parsable."""
    iso = _to_iso_date(s)
    if not iso:
        return None
    return datetime.strptime(iso, "%Y-%m-%d").date()


def _looks_like_date(s: str) -> bool:
    """Use _to_iso_date to decide if this looks like a date."""
    return _to_iso_date(s) != ""


def _choose_once_posting_date(arrival_text: str, departure_text: str, business_date: date) -> str:
    """
    Choose a single posting date that:
    - Is on or after both arrival_date and business_date
    - Is strictly before departure_date (cannot be checkout day)
    If no such date exists, return ''.
    """
    arr = _parse_to_date(arrival_text)
    dep = _parse_to_date(departure_text)

    if not arr or not dep:
        return ""

    latest = dep - timedelta(days=1)  # can't be checkout
    earliest = max(arr, business_date)

    if earliest > latest:
        return ""

    delta_days = (latest - earliest).days
    offset = random.randint(0, delta_days)
    posting = earliest + timedelta(days=offset)
    return posting.isoformat()


def _choose_daily_period(arrival_text: str, departure_text: str, business_date: date):
    """
    Choose a date range [start, end] for a Daily schedule:
    - start >= max(arrival_date, business_date)
    - end <= departure_date - 1 day (not checkout)
    - start <= end
    Returns (start_iso, end_iso) or ('', '') if impossible.
    """
    arr = _parse_to_date(arrival_text)
    dep = _parse_to_date(departure_text)

    if not arr or not dep:
        return "", ""

    latest = dep - timedelta(days=1)
    earliest = max(arr, business_date)

    if earliest > latest:
        return "", ""

    total_days = (latest - earliest).days
    start_offset = random.randint(0, total_days)
    start = earliest + timedelta(days=start_offset)

    remaining_days = (latest - start).days
    end_offset = random.randint(0, remaining_days) if remaining_days > 0 else 0
    end = start + timedelta(days=end_offset)

    return start.isoformat(), end.isoformat()


# ----------------------------
# OTHER HELPERS
# ----------------------------

def _is_digits(s: str) -> bool:
    """True if string is only digits (e.g. 27919, 28010, 5006885)."""
    if not isinstance(s, str):
        s = str(s)
    return s.strip().isdigit()


def _find_column(df: pd.DataFrame, candidates, context: str) -> str:
    """
    Find the first existing column in df among candidates (case-insensitive, trimmed).
    Return the actual column name from the DataFrame.
    Raise a clear error if none is found.
    """
    normalized = {col.strip().lower(): col for col in df.columns}

    for cand in candidates:
        key = cand.strip().lower()
        if key in normalized:
            return normalized[key]

    raise ValueError(
        f"Departures file must contain one of columns {candidates} for {context}. "
        f"Found columns: {list(df.columns)}"
    )


def load_transaction_prices(path: str):
    """
    Load transaction prices and allowed schedule modes from CSV.

    Expected columns:
    - TransactionCode
    - UnitPrice
    Optional:
    - ScheduleType: 'once', 'daily', 'both'

    Returns:
      prices: dict[code] -> float
      schedule_modes: dict[code] -> set({'once','daily'})
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Transaction prices file not found: {path}")

    df = pd.read_csv(path)

    expected_cols = {"TransactionCode", "UnitPrice"}
    if not expected_cols.issubset(df.columns):
        raise ValueError(
            f"Transaction prices file must contain columns: {expected_cols}. "
            f"Found: {list(df.columns)}"
        )

    has_schedule = "ScheduleType" in df.columns

    prices = {}
    schedule_modes = {}

    for _, row in df.iterrows():
        code = str(row["TransactionCode"]).strip()
        if code == "":
            continue
        try:
            price = float(row["UnitPrice"])
        except Exception:
            continue
        prices[code] = price

        modes = {"once"}  # default
        if has_schedule:
            raw = str(row["ScheduleType"]).strip().lower()
            if raw == "daily":
                modes = {"daily"}
            elif raw == "both":
                modes = {"once", "daily"}
            elif raw == "once" or raw == "":
                modes = {"once"}
            else:
                modes = {"once"}

        schedule_modes[code] = modes

    if not prices:
        raise ValueError("No valid transaction prices loaded from file.")

    return prices, schedule_modes


def extract_arrival_ids(arrivals_df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    From the arrivals report, extract for each reservation:

    - ConfIdent: numeric row where Arrival looks like a date
                 (matches 'Conf. # / Ident. #' in departures, or Name fallback)
    - ExternalId: numeric row nearby where Arrival is HOLD / PRE / OFF
                  (long external ID, e.g. 5006885)

    We match by proximity in the file (within +/- `window` rows).
    """

    df = arrivals_df.copy()

    required_cols = {"Name", "Arrival"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Arrivals file must contain columns: {required_cols}. Missing: {missing}")

    df["idx"] = df.index

    numeric = df[df["Name"].astype(str).apply(_is_digits)]

    conf_rows = numeric[numeric["Arrival"].apply(_looks_like_date)]
    ext_rows = numeric[numeric["Arrival"].isin(["HOLD", "PRE", "OFF"])]

    if conf_rows.empty or ext_rows.empty:
        raise ValueError("Could not find enough numeric ConfIdent/ExternalId rows in arrivals file.")

    pairs = []
    for _, ext in ext_rows.iterrows():
        idx = ext["idx"]
        nearby = conf_rows[
            (conf_rows["idx"] >= idx - window) &
            (conf_rows["idx"] <= idx + window)
        ]
        if nearby.empty:
            continue
        nearest = nearby.iloc[(nearby["idx"] - idx).abs().argsort().iloc[0]]
        conf_id = str(nearest["Name"]).strip()
        ext_id = str(ext["Name"]).strip()
        pairs.append((conf_id, ext_id))

    if not pairs:
        raise ValueError("No ConfIdent/ExternalId pairs could be matched in arrivals file.")

    pair_df = pd.DataFrame(pairs, columns=["ConfIdent", "ExternalId"])
    pair_df = pair_df.drop_duplicates(subset=["ConfIdent"])
    return pair_df


def build_fixed_charges(
    arrivals_df: pd.DataFrame,
    departures_df: pd.DataFrame,
    transaction_prices: dict,
    schedule_modes: dict,
) -> pd.DataFrame:
    """
    Join arrivals + departures, then build fixed charges table (max 499 rows).

    For each matched reservation:
    - ExternalId: long ID (500xxxx) from arrivals
    - AccountId: BNC-... from departures
    - TransactionCode: random code from transaction_prices
    - ScheduleType: random choice from schedule_modes[code]
    - Posting dates obey business-date and checkout rules.
    """

    if not transaction_prices:
        raise ValueError("transaction_prices dictionary is empty.")

    transaction_codes = list(transaction_prices.keys())
    business_date = date.today()

    # 1) ConfIdent + ExternalId from arrivals
    arr_ids = extract_arrival_ids(arrivals_df)

    # 2) AccountId + stay dates from departures
    dep = departures_df.copy()

    # Detect columns flexibly
    conf_col = _find_column(
        dep,
        [
            "Conf. # / Ident. #",
            "Conf # / Ident #",
            "Conf#/Ident#",
            "Confirmation",
            "Conf No",
            "Confirmation #",
            "Reservation #",
            "Reservation No",
            "Res #",
            "Name",  # fallback for exports where the numeric ID is in Name
        ],
        "confirmation / ConfIdent",
    )
    acc_col = _find_column(
        dep,
        ["Acc. #", "Acc #", "Account #", "Account No", "Account No.", "Account"],
        "account id",
    )
    arr_col = _find_column(
        dep,
        [
            "Arr. Date & Time",
            "Arrival Date & Time",
            "Arr. Date",
            "Arrival Date",
            "Arrival",
            "Arrival & ETA",
        ],
        "arrival date",
    )
    dep_col = _find_column(
        dep,
        [
            "Dep. Date & Time",
            "Departure Date & Time",
            "Dep. Date",
            "Departure Date",
            "Departure",
            "Departure & ETD",
        ],
        "departure date",
    )

    dep_small = dep[[conf_col, acc_col, arr_col, dep_col]].copy()
    dep_small = dep_small.rename(
        columns={
            conf_col: "ConfIdent",
            acc_col: "AccountId",
            arr_col: "Arr. Date & Time",
            dep_col: "Dep. Date & Time",
        }
    )

    dep_small["ConfIdent"] = dep_small["ConfIdent"].astype(str).str.strip()
    arr_ids["ConfIdent"] = arr_ids["ConfIdent"].astype(str).str.strip()

    # 3) Join on ConfIdent (short code)
    merged = arr_ids.merge(dep_small, on="ConfIdent", how="inner")

    # Keep only BNC accounts
    merged = merged[merged["AccountId"].astype(str).str.startswith("BNC")].copy()

    # 4) Build fixed charges rows (respect MAX_RECORDS)
    records = []
    for _, row in merged.iterrows():
        if len(records) >= MAX_RECORDS:
            break

        external_id = str(row["ExternalId"])
        account_id = str(row["AccountId"])

        tx_code = random.choice(transaction_codes)
        unit_price = transaction_prices[tx_code]

        allowed = schedule_modes.get(tx_code, {"once"})
        if "daily" in allowed and "once" in allowed:
            chosen_mode = random.choice(["Once", "Daily"])
        elif "daily" in allowed:
            chosen_mode = "Daily"
        else:
            chosen_mode = "Once"

        arr_text = row["Arr. Date & Time"]
        dep_text = row["Dep. Date & Time"]

        if chosen_mode == "Once":
            posting_date = _choose_once_posting_date(arr_text, dep_text, business_date)
            if not posting_date:
                continue
            from_date = posting_date
            to_date = posting_date
        else:
            from_date, to_date = _choose_daily_period(arr_text, dep_text, business_date)
            if not from_date:
                continue

        rec = {
            "ExternalSystemCode": "PMS",
            "ExternalId": external_id,
            "AccountId": account_id,
            "TransactionCode": tx_code,
            "Quantity": 1,
            "UnitPrice": unit_price,
            "Comment": "Migration",
            "ScheduleType": chosen_mode,
            "FromDate": from_date,
            "ToDate": to_date,
            "CustomPostingDates": "",
            "DayOfMonth": "",
            "MonthsBetweenPostings": "",
            "DaysOfWeek": "",
            "OrdinalOccurenceOfDayOfWeek": "",
        }
        records.append(rec)

    fixed_df = pd.DataFrame(
        records,
        columns=[
            "ExternalSystemCode",
            "ExternalId",
            "AccountId",
            "TransactionCode",
            "Quantity",
            "UnitPrice",
            "Comment",
            "ScheduleType",
            "FromDate",
            "ToDate",
            "CustomPostingDates",
            "DayOfMonth",
            "MonthsBetweenPostings",
            "DaysOfWeek",
            "OrdinalOccurenceOfDayOfWeek",
        ],
    )

    fixed_df = fixed_df.fillna("")
    return fixed_df


def main():
    args = parse_args()

    # Ensure output folder exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(args.arrivals):
        raise FileNotFoundError(f"Arrivals file not found: {args.arrivals}")
    if not os.path.exists(args.departures):
        raise FileNotFoundError(f"Departures file not found: {args.departures}")

    arrivals = pd.read_csv(args.arrivals)
    departures = pd.read_csv(args.departures)

    transaction_prices, schedule_modes = load_transaction_prices(args.txfile)

    fixed_df = build_fixed_charges(arrivals, departures, transaction_prices, schedule_modes)

    fixed_df.to_csv(args.output, index=False)
    print(f"Generated fixed charges file: {args.output}")
    print(f"Rows: {len(fixed_df)}")


if __name__ == "__main__":
    main()
