#!/usr/bin/env python3
"""
Generate a daily reservations CSV (499 rows) following business rules,
using weighted room type distribution and persistent ExternalIDs.

Now also uses an external Business-on-the-Books file from GitHub to
bias arrivals towards low-occupancy dates (Column F = Occupancy%).
"""

from pathlib import Path
from datetime import datetime, timedelta, time
import random
import json
import csv
from urllib.request import urlopen

OUTPUT_DIR = Path("./output")
STATE_DIR = Path("./state")
STATE_FILE = STATE_DIR / "state.json"

# URL of the PMS occupancy file stored in GitHub (Column F = Occupancy%)
OCCUPANCY_URL = (
    "https://raw.githubusercontent.com/blumet/hotel-reservations-generator/"
    "refs/heads/main/data/BusinessontheBooks.csv"
)
LOW_OCCUPANCY_THRESHOLD = 60.0  # percentage

CONFIG = {
    "external_id_start": 5007119,  # New starting ExternalID
    "profile_set_1": [2000042, 2002529],
    "profile_set_2": [1000042, 1001335],
    "allow_COR25": False,
    "date_distribution": {  # Fallback: weighted random date periods
        "next_30_days": 0.5,   # 50%
        "next_60_days": 0.2,   # 20%
        "next_90_days": 0.2,   # 20%
        "next_120_days": 0.1   # 10%
    },
    "room_distribution": {  # Weighted room type probabilities
        "KGDX": 0.30,
        "TWDX": 0.20,
        "KGSP": 0.10,
        "TWSP": 0.05,
        "A1KB": 0.02,
        "KGST": 0.083,
        "KCDX": 0.083,
        "TCDX": 0.083,
        "KINGR": 0.083,
        "KCST": 0.083,
    }
}

MARKET_SEGMENTS = ["FIT1", "PRMF", "CORG", "CORN", "CORL"]
GUARANTEE_TYPES = ["PRE", "HOLD", "OFF"]
CHANNELS = ["CRS", "GDS", "HOT", "IBE", "OTH", "SIT", "SYN", "CRO"]
SOURCES = ["COR", "IBE", "PMS", "SAL", "TVL"]

COMPANIES = [
    "Accenture", "Telefonica", "BerkshireHathaway", "GrupoACS", "Rolex AG", "Exxon", "Volkswagen",
    "Tesla", "Saudi Aramco", "Bilbao", "MercedesB", "GrupoCatalana", "Siemens", "Amazon Switzerland",
    "Adidas GmbH", "Amazon Austria", "Amazon Germany", "Apple Incorporated Spain", "Austrian Airlines",
    "ABN Amro", "CNN News Group", "BMW", "Deloitte"
]

RATE_CODES_POOL = ["BAR00", "BAR10", "OTA1", "CORL25", "OPQ", "DAY", "BARAD", "RACK"]
if CONFIG["allow_COR25"]:
    RATE_CODES_POOL.append("COR25")

PREFERENCE_CODES = [
    "ACCESS", "BAL", "BAL2", "DBL2", "FPLACE", "KING", "PVIEW", "GVIEW", "SOFB", "SOFL",
    "2BTH", "CON", "DIN", "EXT", "HF", "HTCN", "KITC", "LF", "LUG", "MIN", "PLA",
    "POW", "QUI", "SAF", "SEA", "SM", "SPAB", "TWIN"
]

# ------------------------------------------------------------
# Occupancy handling: load low-occupancy days from external CSV
# ------------------------------------------------------------

def load_occupancy_data():
    """Download occupancy data from OCCUPANCY_URL and parse dates & Occupancy%."""
    rows = []
    try:
        with urlopen(OCCUPANCY_URL) as resp:
            content = resp.read().decode("utf-8-sig")  # handle BOM if present
    except Exception as exc:  # network fallback
        print(f"⚠️ Could not load occupancy data from {OCCUPANCY_URL}: {exc}")
        return rows

    reader = csv.DictReader(content.splitlines())
    for row in reader:
        date_str = (row.get("Date") or "").strip()
        occ_str = (row.get("Occupancy%") or "").strip()
        if not date_str or not occ_str:
            continue

        # Normalize decimal separator just in case
        occ_str = occ_str.replace(",", ".")
        try:
            occupancy = float(occ_str)
        except ValueError:
            continue

        # PMS export uses DD/MM/YYYY
        try:
            date_obj = datetime.strptime(date_str, "%d/%m/%Y").date()
        except ValueError:
            continue

        rows.append({"date": date_obj, "occupancy": occupancy})

    return rows


def compute_low_occupancy(threshold=LOW_OCCUPANCY_THRESHOLD):
    """Return (dates, weights) for days below the given occupancy threshold."""
    data = load_occupancy_data()
    if not data:
        return [], []

    low_days = [entry for entry in data if entry["occupancy"] < threshold]

    # If nothing is strictly below threshold, fall back to using all days
    if not low_days:
        low_days = data
