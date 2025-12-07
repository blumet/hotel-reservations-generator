#!/usr/bin/env python3
"""
Generate Fixed Charges CSV from arrivals + departures reports.

Workflow:
- Reads arrivals.csv and departures.csv from ./input_files
- Reads transaction prices from ./config/transaction_prices.csv
- From ARRIVALS:
    * ConfIdent  = numeric line where Arrival looks like a date
    * ExternalId = numeric line where Arrival is HOLD / PRE / OFF
- From DEPARTURES:
    * AccountId (BNC-...)
    * Arr. Date & Time, Dep. Date & Time (stay dates)
- For each matched reservation (max 499):
    * Choose a random PostingDate between arrival & departure (inclusive)
    * FromDate = PostingDate  (YYYY-MM-DD)
    * ToDate   = PostingDate
"""

import argparse
import os
import random
import re
from datetime import datetime, timedelta

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
        r"\d{2}/\d{2}/\d{2}",
        r"\d{2}-\d{2}-\d{2}",
        r"\d{2}-[A-Za-z]{3}-\d{2,4}",
        r"\d{4}-\d{2}-\d{2}",
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
        "%d/%m/%y",
        "%d-%m-%y",
        "%d-%b-%Y",
        "%d-%b-%y",
        "%Y-%m-%d",
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


def _choose_stay_date(arrival_text: str, departure_text: str) -> str:
    """
    Choose a random date between arrival and departure (inclusive).
    Fallbacks:
    - If only arrival is valid -> use arrival
    - If only departure is valid -> use departure
    - If neither -> return empty string
    """
    arr = _parse_to_date(arrival_text)
    dep = _parse_to_date(departure_text)

    if arr and dep and dep >= arr:
        delta = (dep - arr).days
        offset = random.randint(0, delta)
        return (arr + timedelta(days=offset)).isoformat()

    if arr:
        return arr.isoformat()
    if dep:
        return dep.isoformat()
    return ""


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


def extract_arrival_ids(arrivals_df: pd.DataFrame) -> pd.DataFrame:
    """
    From the arrivals report, extract for each reservation group:

    - ConfIdent: numeric Name where Arrival looks like a date
                 (this matches 'Conf. # / Ident. #' in departures)
    - ExternalId: numeric Name where Arrival is HOLD / PRE / OFF
                  (this is the long external ID like 5006885)

    Returns DataFrame with columns: ConfIdent, ExternalId
    """

    df = arrivals_df.copy()

    required_cols = {"Room Number", "Name", "Arrival"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Arrivals file must contain columns: {required_cols}. Missing: {missing}")

    # Build group_id by "Room Number" header rows
    group_ids = []
    current_group = -1
    for rn in df["Room Number"]:
        if not (isinstance(rn, float) and pd.isna(rn)) and str(rn).strip() != "":
            current_group += 1
        group_ids.append(current_group)
    df["group_id"] = group_ids

    rows = []
    for gid, grp in df.groupby("group_id"):
        if gid < 0:
            continue

        # Numeric NAME rows in this group
        numeric = grp[grp["Name"].astype(str).apply(_is_digits)]

        if numeric.empty:
            continue

        conf_rows = numeric[numeric["Arrival"].apply(_looks_like_date)]
        ext_rows = numeric[numeric["Arrival"].isin(["HOLD", "PRE", "OFF"])]

        if conf_rows.empty or ext_rows.empty:
            # If we don't find both, skip this reservation
            continue

        conf_id = str(conf_rows["Name"].iloc[0]).strip()
        ext_id = str(ext_rows["Name"].iloc[0]).strip()

        rows.append({"ConfIdent": conf_id, "ExternalId": ext_id})

    if not rows:
        raise ValueError("No ConfIdent/ExternalId pairs could be extracted from arrivals file.")

    return pd.DataFrame(rows)


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
    - PostingDate: random date between Arr. Date & Time and Dep. Date & Time
    - FromDate = PostingDate
    - ToDate   = PostingDate
    """

    if not transaction_prices:
        raise ValueError("transaction_prices dictionary is empty.")

    transaction_codes = list(transaction_prices.keys())

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

    # 3) Join on ConfIdent (short code)
    arr_ids["ConfIdent"] = arr_ids["ConfIdent"].astype(str).str.strip()

    merged = arr_ids.merge(dep_small, on="ConfIdent", how="inner")

    # Keep only BNC accounts
    merged = merged[merged["AccountId"].astype(str).str.startswith("BNC")].copy()

    # 4) Build fixed charges rows (respect MAX_RECORDS)
    records = []
    for idx, (_, row) in enumerate(merged.iterrows()):
        if idx >= MAX_RECORDS:
            break

        external_id = str(row["ExternalId"])
        account_id = str(row["AccountId"])

        tx_code = random.choice(transaction_codes)
        unit_price = transaction_prices[tx_code]

        posting_date = _choose_stay_date(
            row["Arr. Date & Time"],
            row["Dep. Date & Time"],
        )

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
