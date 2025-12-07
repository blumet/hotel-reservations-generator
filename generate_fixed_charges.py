#!/usr/bin/env python3
"""
Generate Fixed Charges CSV from arrivals + departures reports.

Workflow:
- Reads arrivals.csv and departures.csv from ./input_files
- Reads transaction prices from ./config/transaction_prices.csv
- Extracts:
    - ExternalId from arrivals file
    - AccountId (BNC-...) from departures file
    - FromDate = arrival date (ISO yyyy-mm-dd)
- Builds fixed charges CSV ready for migration.

Usage (from repo root, with Python 3 installed):

    python generate_fixed_charges.py

Optional arguments:

    python generate_fixed_charges.py \
        --arrivals input_files/arrivals.csv \
        --departures input_files/departures.csv \
        --txfile config/transaction_prices.csv \
        --output output_files/fixed_charges_output.csv
"""

import argparse
import os
import random
import re
from datetime import datetime

import pandas as pd


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
# HELPER FUNCTIONS
# ----------------------------

def _looks_like_date(s: str) -> bool:
    """True if string contains dd/mm/yyyy."""
    if not isinstance(s, str):
        return False
    return bool(re.search(r"\d{2}/\d{2}/\d{4}", s))


def _is_digits(s: str) -> bool:
    """True if string is only digits (e.g. 27919, 28010)."""
    if not isinstance(s, str):
        return False
    return s.strip().isdigit()


def _to_iso_date(s: str) -> str:
    """
    Extract dd/mm/yyyy from the string and convert to ISO yyyy-mm-dd.
    Returns '' if parsing fails.
    """
    if not isinstance(s, str):
        return ""
    m = re.search(r"(\d{2}/\d{2}/\d{4})", s)
    if not m:
        return ""
    dt = datetime.strptime(m.group(1), "%d/%m/%Y").date()
    return dt.isoformat()


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

    # Convert to dict: { "20-01": 32.0, ... }
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


def extract_arrival_external_ids(arrivals_df: pd.DataFrame) -> pd.DataFrame:
    """
    From the arrivals report, extract:
    - ExternalId: numeric "Name" rows where the Arrival column is a date
      (these match the Conf. # / Ident. # in departures)
    - header_arrival: the arrival date from the header guest line in that block

    The arrivals export is multi-line per reservation; we:
    - Assign a group_id that increments whenever "Room Number" is non-null.
    - For each group, the first row with a Room Number is the guest header.
    - Within that group, we pick the row where:
        Name is digits (e.g. '27919') AND Arrival looks like a date.
    """
    df = arrivals_df.copy()

    if "Room Number" not in df.columns or "Name" not in df.columns or "Arrival" not in df.columns:
        raise ValueError("Arrivals file must contain columns: 'Room Number', 'Name', 'Arrival'")

    # Build group_id by "Room Number" header rows
    group_ids = []
    current_group = -1
    for rn in df["Room Number"]:
        # New group when Room Number is not NaN and not empty
        if not (isinstance(rn, float) and pd.isna(rn)) and str(rn).strip() != "":
            current_group += 1
        group_ids.append(current_group)
    df["group_id"] = group_ids

    # For each group, store the header arrival (guest row)
    header_arrival = {}
    for gid, grp in df.groupby("group_id"):
        if gid < 0:
            continue
        guest_rows = grp[grp["Room Number"].notna()]
        if guest_rows.empty:
            continue
        header_row = guest_rows.iloc[0]
        header_arrival[gid] = header_row["Arrival"]

    df["header_arrival"] = df["group_id"].map(header_arrival)

    # ExternalId candidates: Name is digits AND Arrival looks like a date
    mask = df["Name"].apply(_is_digits) & df["Arrival"].apply(_looks_like_date)
    arr_ids = df.loc[mask, ["group_id", "Name", "header_arrival"]].copy()
    arr_ids.rename(columns={"Name": "ExternalId"}, inplace=True)

    return arr_ids


def build_fixed_charges(
    arrivals_df: pd.DataFrame,
    departures_df: pd.DataFrame,
    transaction_prices: dict,
) -> pd.DataFrame:
    """
    Core logic: join arrivals + departures, build fixed charges table.

    Requirements:
    - ExternalSystemCode: always "PMS"
    - ExternalId: from arrivals (numeric row)
    - AccountId: from departures (Acc. #, must start with BNC)
    - TransactionCode: random from transaction_prices keys
    - Quantity: always 1
    - UnitPrice: from transaction_prices
    - Comment: "Migration"
    - ScheduleType: "Once"
    - FromDate: arrival date (ISO yyyy-mm-dd, mirroring the arrival date)
    - Other schedule fields: blank
    """

    if not transaction_prices:
        raise ValueError("transaction_prices dictionary is empty.")

    transaction_codes = list(transaction_prices.keys())

    # 1) ExternalId + arrival date from arrivals
    arr_ids = extract_arrival_external_ids(arrivals_df)

    # 2) AccountId from departures
    dep = departures_df.copy()

    if "Conf. # / Ident. #" not in dep.columns or "Acc. #" not in dep.columns:
        raise ValueError("Departures file must contain columns: 'Conf. # / Ident. #', 'Acc. #'")

    dep = dep.rename(
        columns={
            "Conf. # / Ident. #": "ConfIdent",
            "Acc. #": "AccountId",
        }
    )

    dep_small = dep[["ConfIdent", "AccountId"]].copy()

    # 3) Join arrivals & departures on ExternalId <-> ConfIdent
    merged = arr_ids.merge(
        dep_small,
        left_on="ExternalId",
        right_on="ConfIdent",
        how="inner",
    )

    # Keep only BNC accounts
    merged = merged[merged["AccountId"].astype(str).str.startswith("BNC")].copy()

    # 4) Build fixed charges rows
    records = []
    for _, row in merged.iterrows():
        external_id = str(row["ExternalId"])
        account_id = str(row["AccountId"])

        # random transaction code from configured list
        tx_code = random.choice(transaction_codes)
        unit_price = transaction_prices[tx_code]

        from_date = _to_iso_date(row["header_arrival"])

        rec = {
            "ExternalSystemCode": "PMS",
            "ExternalId": external_id,
            "AccountId": account_id,
            "TransactionCode": tx_code,
            "Quantity": 1,
            "UnitPrice": unit_price,
            "Comment": "Migration",
            "ScheduleType": "Once",
            "FromDate": from_date,
            "ToDate": "",
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
