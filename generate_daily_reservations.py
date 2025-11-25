#!/usr/bin/env python3
"""
Generate a daily reservations CSV (499 rows) following business rules,
using weighted room type distribution and persistent ExternalIDs.

Now also uses an external Business-on-the-Books file from GitHub to
bias arrivals towards low-occupancy dates (Occupancy% column in PMS export).

ExternalID/profile state is only persisted when the script is run with --commit.
"""

from pathlib import Path
from datetime import datetime, timedelta, time
import random
import json
import csv
from urllib.request import urlopen
import argparse

OUTPUT_DIR = Path("./output")
STATE_DIR = Path("./state")
STATE_FILE = STATE_DIR / "state.json"

# URL of the PMS occupancy file stored in GitHub
OCCUPANCY_URL = (
    "https://raw.githubusercontent.com/blumet/hotel-reservations-generator/"
    "refs/heads/main/data/BusinessontheBooks.csv"
)

LOW_OCCUPANCY_THRESHOLD = 60.0  # percentage (days below this are "low occupancy")

CONFIG = {
    "external_id_start": 5007119,  # ONLY used when state.json doesn't exist yet
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

# Rate pool (BARRES added)
RATE_CODES_POOL = ["BAR00", "BAR10", "OTA1", "CORL25", "OPQ", "DAY", "BARAD", "RACK", "BARRES"]
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
    """
    Download occupancy data from OCCUPANCY_URL and parse dates & occupancy %.

    Assumptions (from your PMS export):
      - Column 1 (index 1) = "Date" (DD/MM/YYYY)
      - Column 5 (index 5) = "Occupancy%"
    """
    rows = []
    try:
        with urlopen(OCCUPANCY_URL) as resp:
            content = resp.read().decode("utf-8-sig")  # handle BOM if present
    except Exception as exc:
        print(f"⚠️ Could not load occupancy data from {OCCUPANCY_URL}: {exc}")
        return rows

    reader = csv.reader(content.splitlines())
    try:
        headers = next(reader)
    except StopIteration:
        return rows

    if len(headers) < 6:
        print("⚠️ Occupancy CSV has fewer than 6 columns; cannot read Occupancy%.")
        return rows

    date_idx = 1  # "Date" column
    occ_idx = 5   # "Occupancy%" column

    for row in reader:
        if len(row) <= occ_idx:
            continue

        date_str = (row[date_idx] or "").strip()
        occ_str = (row[occ_idx] or "").strip()
        if not date_str or not occ_str:
            continue

        occ_str = occ_str.replace(",", ".")
        try:
            occupancy = float(occ_str)
        except ValueError:
            continue

        # Try DD/MM/YYYY, fallback to YYYY-MM-DD
        dt = None
        for fmt in ("%d/%m/%Y", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(date_str, fmt).date()
                break
            except ValueError:
                continue
        if dt is None:
            continue

        rows.append({"date": dt, "occupancy": occupancy})

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

    dates = [entry["date"] for entry in low_days]
    weights = []
    for entry in low_days:
        gap = threshold - entry["occupancy"]
        # The lower the occupancy, the larger the weight → more arrivals generated
        if gap <= 0:
            weight = 1.0
        else:
            weight = gap + 1.0
        weights.append(weight)

    return dates, weights


LOW_OCCUPANCY_DATES, LOW_OCCUPANCY_WEIGHTS = compute_low_occupancy()

# Map of date -> occupancy% for easy lookup (used for >85% rule)
_OCC_DATA = load_occupancy_data()
OCCUPANCY_BY_DATE = {entry["date"]: entry["occupancy"] for entry in _OCC_DATA}

# ------------------------------------------------------------
# State handling (persistent tracking for ExternalID and ProfileID)
# ------------------------------------------------------------

def build_profile_cycle():
    s1 = list(range(CONFIG["profile_set_1"][0], CONFIG["profile_set_1"][1] + 1))
    s2 = list(range(CONFIG["profile_set_2"][0], CONFIG["profile_set_2"][1] + 1))
    return sorted(s1 + s2)

PROFILE_CYCLE = build_profile_cycle()

def load_state():
    """Load state from file, or initialize if missing."""
    if STATE_FILE.exists():
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    # First ever run: start from CONFIG
    return {"last_external_id": CONFIG["external_id_start"], "next_profile_index": 0}

def save_state(state):
    """Save state persistently."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

def next_external_id(state):
    """Generate next sequential ExternalID."""
    state["last_external_id"] += 1
    return state["last_external_id"]

def next_profile_id(state):
    """Cycle through available profile IDs."""
    idx = state["next_profile_index"]
    pid = PROFILE_CYCLE[idx]
    state["next_profile_index"] = (idx + 1) % len(PROFILE_CYCLE)
    return pid

# ------------------------------------------------------------
# Helper: random time in ISO format for ETA/ETD
# ------------------------------------------------------------

def random_time_iso(date_obj):
    """Return ISO 8601 timestamp with random time between 05:00 and 23:45."""
    hour = random.randint(5, 23)
    minute = random.choice([0, 15, 30, 45])
    dt = datetime.combine(date_obj, time(hour, minute))
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")

# ------------------------------------------------------------
# Helper: arrival date (prefer low-occupancy days, fallback to CONFIG)
# ------------------------------------------------------------

def pick_arrival_date(today):
    """Pick an arrival date, preferring low-occupancy days from the PMS export.

    If LOW_OCCUPANCY_DATES is available, choose from those dates using
    LOW_OCCUPANCY_WEIGHTS so that the lowest occupancy dates get the most
    new reservations. If the occupancy file cannot be loaded, fall back
    to the original CONFIG-based weighted date logic.
    """
    if LOW_OCCUPANCY_DATES:
        return random.choices(
            LOW_OCCUPANCY_DATES,
            weights=LOW_OCCUPANCY_WEIGHTS,
            k=1
        )[0]

    # Fallback: original behaviour using relative date windows
    dist = CONFIG["date_distribution"]
    roll = random.random()
    cumulative = 0

    if roll < (cumulative := cumulative + dist["next_30_days"]):
        start, end = 1, 30
    elif roll < (cumulative := cumulative + dist["next_60_days"]):
        start, end = 31, 60
    elif roll < (cumulative := cumulative + dist["next_90_days"]):
        start, end = 61, 90
    else:
        start, end = 91, 120

    return today + timedelta(days=random.randint(start, end))

# ------------------------------------------------------------
# Room type picker (weighted)
# ------------------------------------------------------------

def pick_room_type():
    """Randomly select a room type using weighted probabilities."""
    room_types = list(CONFIG["room_distribution"].keys())
    weights = list(CONFIG["room_distribution"].values())
    return random.choices(room_types, weights=weights, k=1)[0]

# ------------------------------------------------------------
# Business logic for rates, companies, preferences
# ------------------------------------------------------------

def pick_rate_and_company(room_type, arrival_date):
    """
    Assign rate plan and company based on:
    - Room type (BAREX for KCDX/TCDX/KCST)
    - Occupancy% for the arrival date (>85% → only RACK/BARRES/BAR00)
    - Normal rate pool otherwise
    """
    # Rule 1: BAREX exclusively for certain room types
    if room_type in {"KCDX", "TCDX", "KCST"}:
        return "BAREX", ""

    # Look up occupancy for the arrival date, if available
    occupancy = OCCUPANCY_BY_DATE.get(arrival_date)

    if occupancy is not None and occupancy > 85:
        # High occupancy: restrict rate codes
        candidate_pool = ["RACK", "BARRES", "BAR00"]
    else:
        # Normal day: full rate pool
        candidate_pool = RATE_CODES_POOL

    rate = random.choice(candidate_pool)

    # Corporate rule for COR25 (if ever allowed in RATE_CODES_POOL)
    if rate == "COR25":
        company = random.choice(["Deloitte", "Saudi Aramco", "GrupoACS", "Volkswagen"])
    else:
        company = random.choice(COMPANIES) if random.random() < 0.7 else ""

    return rate, company

def pick_preference():
    """Randomly choose one preference code or leave blank."""
    if random.random() < 0.4:
        return random.choice(PREFERENCE_CODES)
    return ""

# ------------------------------------------------------------
# Row generator
# ------------------------------------------------------------

def generate_row(state, today=None):
    if today is None:
        today = datetime.today().date()

    arrival = pick_arrival_date(today)
    stay_len = random.randint(1, 10)
    departure = arrival + timedelta(days=stay_len)
    room_type = pick_room_type()
    rate, company = pick_rate_and_company(room_type, arrival)

    if room_type == "A1KB":
        no_of_children = 1
        child_age_bucket = "C1"
    else:
        no_of_children = ""
        child_age_bucket = "C1"

    eta = random_time_iso(arrival)
    etd = random_time_iso(departure)
    preference = pick_preference()

    return {
        "profileId": next_profile_id(state),
        "arrivaldate": arrival.strftime("%Y-%m-%d"),
        "departuredate": departure.strftime("%Y-%m-%d"),
        "RoomType": room_type,
        "Room": "",
        "DoNotMove": "",
        "AdultAgeBucket": "A1",
        "NoOfAdults": random.choice([1, 2]),
        "ChildAgeBucket": child_age_bucket,
        "NoOfChildren": no_of_children,
        "RoomTypeToCharge": room_type,
        "RatePlan": rate,
        "CurrencyCode": "EUR",
        "MarketSegment": random.choice(MARKET_SEGMENTS),
        "GuaranteeType": random.choice(GUARANTEE_TYPES),
        "Channel": random.choice(CHANNELS),
        "Source": random.choice(SOURCES),
        "companyProfile": company,
        "TravelAgentProfile": "",
        "Preferences": preference,
        "AccompanyingGuestProfiles": "",
        "Membership": "",
        "ETA": eta,
        "ETD": etd,
        "TotalAmount": "",
        "BreakdownAmount": "",
        "Purpose": "",
        "NoPost": "FALSE",
        "ArrivalTransportType": "",
        "ArrivalFlightNumber": "",
        "ArrivalPickUpDateTime": "",
        "ArrivalDetails": "",
        "DepartureTransportType": "",
        "DepartureFlightNumber": "",
        "DeparturePickUpDateTime": "",
        "DepartureDetails": "",
        "GroupCode": "",
        "LockPrice": "",
        "LockReason": "",
        "ExternalID": next_external_id(state),
        "ExternalSystemCode": "PMS",
        "AdditionalExternalSystems": "",
        "ExternalSegmentNumber": "",
        "BlockCode": ""
    }

# ------------------------------------------------------------
# Main CSV generator (with --commit flag)
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--commit",
        action="store_true",
        help="Persist updated ExternalID/profile state after generating.",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    state = load_state()
    today = datetime.today().date()

    out_name = f"Reservations_{today.strftime('%Y%m%d')}.csv"
    out_path = OUTPUT_DIR / out_name

    columns = [
        "profileId","arrivaldate","departuredate","RoomType","Room","DoNotMove",
        "AdultAgeBucket","NoOfAdults","ChildAgeBucket","NoOfChildren","RoomTypeToCharge",
        "RatePlan","CurrencyCode","MarketSegment","GuaranteeType","Channel","Source",
        "companyProfile","TravelAgentProfile","Preferences","AccompanyingGuestProfiles",
        "Membership","ETA","ETD","TotalAmount","BreakdownAmount","Purpose","NoPost",
        "ArrivalTransportType","ArrivalFlightNumber","ArrivalPickUpDateTime","ArrivalDetails",
        "DepartureTransportType","DepartureFlightNumber","DeparturePickUpDateTime","DepartureDetails",
        "GroupCode","LockPrice","LockReason","ExternalID","ExternalSystemCode",
        "AdditionalExternalSystems","ExternalSegmentNumber","BlockCode"
    ]

    rows = [generate_row(state, today=today) for _ in range(499)]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Generated {len(rows)} reservations -> {out_path}")
    print(f"🔢 Last ExternalID used in this file: {state['last_external_id']}")

    if args.commit:
        save_state(state)
        print("💾 State committed (future runs will continue from this ExternalID).")
    else:
        print("ℹ️ TEST RUN: state NOT saved. To update ExternalID state, run with --commit.")

if __name__ == "__main__":
    main()
