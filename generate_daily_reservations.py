#!/usr/bin/env python3
"""
Generate a daily reservations CSV (499 rows) following business rules,
using weighted room type distribution and persistent ExternalIDs.
"""

from pathlib import Path
from datetime import datetime, timedelta, time
import random
import json
import csv

OUTPUT_DIR = Path("./output")
STATE_DIR = Path("./state")
STATE_FILE = STATE_DIR / "state.json"

CONFIG = {
    "external_id_start": 5006620,  # New starting ExternalID
    "profile_set_1": [2000042, 2002529],
    "profile_set_2": [1000042, 1001335],
    "allow_COR25": False,
    "date_distribution": {  # Adjustable weighted random date periods
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
# Helper: weighted arrival date based on CONFIG
# ------------------------------------------------------------

def pick_arrival_date(today):
    """Select a random arrival date based on weighted probabilities."""
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

def pick_rate_and_company(room_type):
    """Assign rate plan and company based on room type and business rules."""
    if room_type in {"KCDX", "TCDX", "KCST"}:
        return "BAREX", ""
    rate = random.choice(RATE_CODES_POOL)
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
    rate, company = pick_rate_and_company(room_type)

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
# Main CSV generator
# ------------------------------------------------------------

def main():
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

    # 🔢 Generate 499 reservations instead of 100
    rows = [generate_row(state, today=today) for _ in range(499)]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    save_state(state)
    print(f"✅ Generated {len(rows)} reservations -> {out_path}")
    print(f"🔢 Last ExternalID used: {state['last_external_id']}")

if __name__ == "__main__":
    main()
