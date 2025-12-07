#!/usr/bin/env python3
"""
Generate Fixed Charges CSV from arrivals + departures reports.

Workflow:
- Reads arrivals.csv and departures.csv from ./input_files
- Reads transaction prices from ./config/transaction_prices.csv

From ARRIVALS:
    * ConfIdent  = numeric "Name" row where Arrival looks like a date
                   (short ID like 28726)
    * ExternalId = numeric "Name" row nearby where Arrival is HOLD / PRE / OFF
                   (long ID like 5006885)

From DEPARTURES:
    * AccountId (BNC-...)
    * Arr. Date & Time, Dep. Date & Time (stay dates)

For each matched reservation (max 499):
    * Choose a PostingDate such that:
        - PostingDate is between arrival and departure (inclusive of arrival,
          exclusive of departure)
        - PostingDate is NOT in the past vs today's date (business date)
    * FromDate = PostingDate  (YYYY-MM-DD)
    * ToDate   = PostingDate

If no valid posting date exists for a reservation (e.g. stay fully in the past),
that reservation is skipped.

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

    # Try to find a date-like chunk inside the string
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

    # If we didn't find a substring, maybe the whole string *is* the date
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


def _choose_posting_date(arrival_text: str, departure_text: str, business_date: date) -> str:
    """
    Choose a posting date that:
    - Is on or after both arrival_date and business_date
    - Is strictly before departure_date (cannot be checkout day)

    If no such date exists, return ''.
    """
    arr = _parse_to_date(arrival_text)
    dep = _parse_to_date(departure_text)

    if not arr or not dep:
        return ""

    # Latest valid day is day before departure (can't be checkout)
    latest = dep - timedelta(days=1)
    # Earliest valid is max(arrival, business_date)
    earliest = max(arr, business_date)

    if earliest > latest:
        # No valid date range (stay in the past, or dep == arr, etc.)
        return ""

    delta_days = (latest - earliest).days
    offset = random.randint(0, delta_days)
    posting = earliest + timedelta(days=offset)
    return posting.isoformat()


# ----------------------------
# OTHER HELPERS
# ----------------------------

def _is_digits(s: str) -> bool:
    """True if string is only digits (e.g. 27919, 28010, 5006885)."""
    if not isinstance(s, str):
        s = str(s)
    return s.strip().isdigit()


def load_transaction_prices(path: str) -> dict:
    """
    Load transaction prices from CSV.

    Expected columns:
    - TransactionCode
    - UnitPrice
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

    prices = {}
    for _, row in df.iterrows():
        code = str(row["TransactionCode"]).strip()
        if code == "":
            continue
        try:
            price = float(row["UnitPrice"])
        except Exception:
            continue
        prices[code] = price

    if not prices:
        raise ValueError("No valid transaction prices loaded from file.")

    return prices


def extract_arrival_ids(arrivals_df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    From the arrivals report, extract for each reservation:

    - ConfIdent: numeric row where Arrival looks like a date
                 (matches 'Conf. # / Ident. #' in departures)
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

    # Numeric rows
    numeric = df[df["Name"].astype(str).apply(_is_digits)]

    # ConfIdent candidates: numeric + Arrival looks like a date
    conf_rows = numeric[numeric["Arrival"].apply(_looks_like_date)]

    # ExternalId candidates: numeric + Arrival in HOLD/PRE/OFF
    ext_rows = numeric[numeric["Arrival"].isin(["HOLD", "PRE", "OFF"])]

    if conf_rows.empty or ext_rows.empty:
        raise ValueError("Could not find enough numeric ConfIdent/ExternalId rows in arrivals file.")

    pairs = []
    for _, ext in ext_rows.iterrows():
        idx = ext["idx"]
        # Look for nearby conf rows
        nearby = conf_rows[
            (conf_rows["idx"] >= idx - window) &
            (conf_rows["idx"] <= idx + window)
        ]
        if nearby.empty:
            continue
        # Take the closest by index
        nearest = nearby.iloc[(nearby["idx"] - idx).abs().argsort().iloc[0]]
        conf_id = str(nearest["Name"]).strip()
        ext_id = str(ext["Name"]).strip()
        pairs.append((conf_id, ext_id))

    if not pairs:
        raise ValueError("No ConfIdent/ExternalId pairs could be matched in arrivals file.")

    pair_df = pd.DataFrame(pairs, columns=["ConfIdent", "ExternalId"])

    # Deduplicate by ConfIdent (one ExternalId per reservation)
    pair_df = pair_df.drop_duplicates(subset=["ConfIdent"])

    return pair_df


def build_fixed_charges(
    arrivals_df: pd.DataFrame,
    departures_df: pd.DataFrame,
    transaction_prices: dict,
) -> pd.DataFrame:
    """
    Join arrivals + departures, then build fixed charges table (max 499 rows).

    For each matched reservation:
    - ExternalId: long ID (500xxxx) from arrivals
    - AccountId: BNC-... from departures
    - PostingDate: random valid posting date respecting business date and not checkout
    - FromDate = PostingDate
    - ToDate   = PostingDate
    """

    if not transaction_prices:
        raise ValueError("transaction_prices dictionary is empty.")

    transaction_codes = list(transaction_prices.keys())

    # Business date = today's date
    business_date = date.today()

    # 1) ConfIdent + ExternalId from arrivals
    arr_ids = extract_arrival_ids(arrivals_df)

    # 2) AccountId + stay dates from departures
    dep = departures_df.copy()

    required_dep_cols = {
        "Conf. # / Ident. #",
        "Acc. #",
        "Arr. Date & Time",
        "Dep. Date & Time",
    }
    missing_dep = required_dep_cols.difference(dep.columns)
    if missing_dep:
        raise ValueError(
            f"Departures file must contain columns: {required_dep_cols}. Missing: {missing_dep}"
        )

    dep = dep.rename(
        columns={
            "Conf. # / Ident. #": "ConfIdent",
            "Acc. #": "AccountId",
        }
    )

    dep_small = dep[
        ["ConfIdent", "AccountId", "Arr. Date & Time", "Dep. Date & Time"]
    ].copy()
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

        posting_date = _choose_posting_date(
            row["Arr. Date & Time"],
            row["Dep. Date & Time"],
            business_date=business_date,
        )

        if not posting_date:
            # No valid posting date -> skip this reservation
            continue

        tx_code = random.choice(transaction_codes)
        unit_price = transaction_prices[tx_code]

        rec = {
            "ExternalSystemCode": "PMS",
            "ExternalId": external_id,
            "AccountId": account_id,
            "TransactionCode": tx_code,
            "Quantity": 1,
            "UnitPrice": unit_price,
            "Comment": "Migration",
            "ScheduleType": "Once",
            "FromDate": posting_date,
            "ToDate": posting_date,
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

    # Read input CSVs
    if not os.path.exists(args.arrivals):
        raise FileNotFoundError(f"Arrivals file not found: {args.arrivals}")
    if not os.path.exists(args.departures):
        raise FileNotFoundError(f"Departures file not found: {args.departures}")

    arrivals = pd.read_csv(args.arrivals)
    departures = pd.read_csv(args.departures)

    # Load transaction prices
    transaction_prices = load_transaction_prices(args.txfile)

    fixed_df = build_fixed_charges(arrivals, departures, transaction_prices)

    # Write output
    fixed_df.to_csv(args.output, index=False)
    print(f"Generated fixed charges file: {args.output}")
    print(f"Rows: {len(fixed_df)}")


if __name__ == "__main__":
    main()
