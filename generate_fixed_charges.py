#!/usr/bin/env python3
"""
Generate Fixed Charges CSV from arrivals + departures reports.

Workflow:
- Reads arrivals.csv and departures.csv from ./input_files
- Reads transaction prices from ./config/transaction_prices.csv
- Extracts:
    - ExternalId from arrivals file
    - AccountId (BNC-...) from departures file
    - FromDate = stay arrival date (ISO yyyy-mm-dd)
    - ToDate   = stay departure date (ISO yyyy-mm-dd)
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

def _to_iso_date(s: str) -> str:
    """
    Try to parse a date from the input string in several common formats
    and return it as ISO yyyy-mm-dd. Returns '' if parsing fails.

    This makes the output independent of how the date is formatted
    in the CSV (25/12/2025, 25-12-2025, 25-Dec-25, 2025-12-25, etc.).
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

    # Try multiple formats
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
        except ValueErr
