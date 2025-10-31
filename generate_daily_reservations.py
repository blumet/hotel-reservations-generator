#!/usr/bin/env python3
"""
Generate a daily reservations CSV (100 rows) following business rules.
"""

from pathlib import Path
from datetime import datetime, timedelta
import random
import json
import csv

OUTPUT_DIR = Path("./output")
STATE_DIR = Path("./state")
STATE_FILE = STATE_DIR / "state.json"

CONFIG = {
    "external_id_start": 5006091,  # last known ExternalID
    "profile_set_1": [2000042, 2002529],
    "profile_set_2": [1000042, 1001335],
    "allow_COR25": False
}

ROOM_TYPES = ["KGDX", "TWDX", "KGSP", "TWSP", "A1KB", "KGST", "KCDX", "TCDX", "KINGR", "KCST"]
MARKET_SEGMENTS = ["FIT1", "PRMF", "CORG", "CORN", "CORL"]
GUARANTEE_TYPES = ["PRE", "HOLD", "OFF"]
CHANNELS = ["CRS", "GDS", "HOT", "IBE", "OTH", "SIT", "SYN", "CRO"]
SOURCES = ["COR", "IBE", "PMS", "SAL", "TVL"]
COMPANIES = [
    "Accenture", "Telefonica", "BerkshireHathaway", "GrupoACS", "Rolex AG", "Exxon", "Volkswagen",
    "Tesla", "Saudi Aramco", "Bilbao", "MercedesB", "GrupoCatalana", "Siemens", "Amazon Switzerland",
    "Adidas GmbH", "Amazon Austria", "Amazon Germany", "Apple Incorporated Spain", "Austrian Airlines",
    "ABN Amro", "CNN News Group", "BMW", "Deloitte", "Volkswagen AG", "Grupo ACS"
]
RATE_CODES_POOL = ["BAR00", "BAR10", "OTA1", "CORL25", "OPQ", "DAY", "BARAD", "RACK"]
if CONFIG["allow_COR25"]:
    RATE_CODES_POOL.append("COR25")

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
# Business logic for rates, companies, and record generation
# ------------------------------------------------------------

def pick_rate_and_company(room_type):
    if room_type in {"KCDX", "TCDX", "KCST"}:
        return "BAREX", ""
    rate = random.choice(RATE_CODES_POOL)
    if rate == "COR25":
        company = random.choice(["Deloitte", "Saudi Aramco", "Grupo ACS", "Volkswagen AG"])
    else:
        company = random.choice(COMPANIES) if random.random() < 0.7 else ""
    return rate, company

def generate_row(state, today=None):
    if today is None:
        today = datetime.today().date()

    arrival_lower = today + timedelta(days=1)
    arrival_upper = today + timedelta(days=45)
    arrival = arrival_lower + timedelta(days=random.randint(0, (arrival_upper - arrival_lower).days))
    stay_len = random.randint(1, 10)
    departure = arrival + timedelta(days=stay_len)
    room_type = random.choice(ROOM_TYPES)
    rate, company = pick_rate_and_company(room_type)

    return {
        "profileId": next_profile_id(state),
        "arrivaldate": arrival.strftime("%Y-%m-%d"),
        "departuredate": departure.strftime("%Y-%m-%d"),
        "RoomType": room_type,
        "Room": "",
        "DoNotMove": "",
        "AdultAgeBucket": "A1",
        "NoOfAdults": random.choice([1, 2]),
        "ChildAgeBucket": "C1",
        "NoOfChildren": "",
        "RoomTypeToCharge": room_type,
        "RatePlan": rate,
        "CurrencyCode": "EUR",
        "MarketSegment": random.choice(MARKET_SEGMENTS),
        "GuaranteeType": random.choice(GUARANTEE_TYPES),
        "Channel": random.choice(CHANNELS),
        "Source": random.choice(SOURCES),
        "companyProfile": company,
        "TravelAgentProfile": "",
        "Preferences": "",
        "AccompanyingGuestProfiles": "",
        "Membership": "",
        "ETA": "",
        "ETD": "",
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

    rows = [generate_row(state, today=today) for _ in range(100)]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    save_state(state)
    print(f"✅ Generated {len(rows)} reservations -> {out_path}")
    print(f"🔢 Last ExternalID used: {state['last_external_id']}")

if __name__ == "__main__":
    main()
