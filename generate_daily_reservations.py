#!/usr/bin/env python3
"""
Generate a reservations CSV (499 rows) following business rules.

Key behaviors:
- Progressive fill: generates arrivals starting from tomorrow, moving forward only when needed
- Overbooking-safe: NEVER exceeds MAX_OCCUPANCY_PCT on ANY night of the stay
- Uses PMS occupancy file (2 years) for occupancy-by-date
- ExternalID and profile cycling persist in state/state.json (updated every run)
"""

from pathlib import Path
from datetime import datetime, timedelta, time
import random
import json
import csv
from urllib.request import urlopen

# ----------------------------
# Paths
# ----------------------------
OUTPUT_DIR = Path("./output")
STATE_DIR = Path("./state")
STATE_FILE = STATE_DIR / "state.json"

# ----------------------------
# Occupancy source (2-year file)
# ----------------------------
OCCUPANCY_URL = (
    "https://raw.githubusercontent.com/blumet/hotel-reservations-generator/"
    "refs/heads/main/data/BusinessontheBooks.csv"
)

# ----------------------------
# Hotel rules
# ----------------------------
HOTEL_ROOMS = 183
MAX_OCCUPANCY_PCT = 90.0  # hard cap: do not exceed on any night
DELTA_PCT_PER_BOOKING_NIGHT = 100.0 / HOTEL_ROOMS  # ~0.546%

# Rate restriction rule (unchanged):
# If arrival-date occupancy > 85%, allow only these rates (unless BAREX forced).
HIGH_OCC_RATE_THRESHOLD = 85.0
HIGH_OCC_ALLOWED_RATES = ["RACK", "BARRES", "BAR00"]

# ----------------------------
# Config
# ----------------------------
CONFIG = {
    "rows_per_run": 499,
    "external_id_start": 5007119,  # used only if state.json doesn't exist yet
    "profile_set_1": [2000042, 2002529],
    "profile_set_2": [1000042, 1001335],
    "allow_COR25": False,
    "room_distribution": {
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

# Rate pool (BARRES included)
RATE_CODES_POOL = ["BAR00", "BAR10", "OTA1", "CORL25", "OPQ", "DAY", "BARAD", "RACK", "BARRES"]
if CONFIG["allow_COR25"]:
    RATE_CODES_POOL.append("COR25")

PREFERENCE_CODES = [
    "ACCESS", "BAL", "BAL2", "DBL2", "FPLACE", "KING", "PVIEW", "GVIEW", "SOFB", "SOFL",
    "2BTH", "CON", "DIN", "EXT", "HF", "HTCN", "KITC", "LF", "LUG", "MIN", "PLA",
    "POW", "QUI", "SAF", "SEA", "SM", "SPAB", "TWIN"
]

# ============================================================
# Occupancy handling
# ============================================================

def load_occupancy_data():
    """
    Downloads occupancy data from OCCUPANCY_URL and parses dates + occupancy%.

    Assumptions (from your PMS export):
      - Column 1 (index 1) = "Date" (DD/MM/YYYY) (day name may follow)
      - Column 5 (index 5) = "Occupancy%"
    """
    rows = []
    try:
        with urlopen(OCCUPANCY_URL) as resp:
            content = resp.read().decode("utf-8-sig")
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

    date_idx = 1
    occ_idx = 5

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

        dt = None
        # First token usually holds the date even if day name exists
        date_token = date_str.split()[0]
        for fmt in ("%d/%m/%Y", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(date_token, fmt).date()
                break
            except ValueError:
                continue
        if dt is None:
            continue

        rows.append({"date": dt, "occupancy": occupancy})

    return rows


_OCC_DATA = load_occupancy_data()
OCCUPANCY_BY_DATE = {e["date"]: e["occupancy"] for e in _OCC_DATA}
OCCUPANCY_DATES_SORTED = sorted(OCCUPANCY_BY_DATE.keys())

# ============================================================
# State handling (ExternalID, ProfileID, and progressive cursor)
# ============================================================

def build_profile_cycle():
    s1 = list(range(CONFIG["profile_set_1"][0], CONFIG["profile_set_1"][1] + 1))
    s2 = list(range(CONFIG["profile_set_2"][0], CONFIG["profile_set_2"][1] + 1))
    return sorted(s1 + s2)


PROFILE_CYCLE = build_profile_cycle()


def load_state():
    """
    state.json schema:
      - last_external_id: int
      - next_profile_index: int
      - cursor_date: YYYY-MM-DD  (earliest date to try filling next)
    """
    if STATE_FILE.exists():
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)
    else:
        state = {"last_external_id": CONFIG["external_id_start"], "next_profile_index": 0}

    if "cursor_date" not in state:
        state["cursor_date"] = (datetime.today().date() + timedelta(days=1)).strftime("%Y-%m-%d")

    return state


def save_state(state):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def next_external_id(state):
    state["last_external_id"] += 1
    return state["last_external_id"]


def next_profile_id(state):
    idx = state["next_profile_index"]
    pid = PROFILE_CYCLE[idx]
    state["next_profile_index"] = (idx + 1) % len(PROFILE_CYCLE)
    return pid

# ============================================================
# Overbooking-safe occupancy math
# ============================================================

def projected_occupancy(date_obj, added_pct_by_date):
    base = OCCUPANCY_BY_DATE.get(date_obj, None)
    if base is None:
        # Your file covers 2 years, but this protects against gaps.
        return None
    return base + added_pct_by_date.get(date_obj, 0.0)


def can_place_stay(arrival, nights, added_pct_by_date):
    for i in range(nights):
        d = arrival + timedelta(days=i)
        proj = projected_occupancy(d, added_pct_by_date)
        if proj is None:
            return False
        if proj + DELTA_PCT_PER_BOOKING_NIGHT > MAX_OCCUPANCY_PCT:
            return False
    return True


def commit_stay(arrival, nights, added_pct_by_date):
    for i in range(nights):
        d = arrival + timedelta(days=i)
        added_pct_by_date[d] = added_pct_by_date.get(d, 0.0) + DELTA_PCT_PER_BOOKING_NIGHT

# ============================================================
# Helpers
# ============================================================

def random_time_iso(date_obj):
    hour = random.randint(5, 23)
    minute = random.choice([0, 15, 30, 45])
    dt = datetime.combine(date_obj, time(hour, minute))
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def pick_room_type():
    room_types = list(CONFIG["room_distribution"].keys())
    weights = list(CONFIG["room_distribution"].values())
    return random.choices(room_types, weights=weights, k=1)[0]


def pick_preference():
    return random.choice(PREFERENCE_CODES) if random.random() < 0.4 else ""


def pick_rate_and_company(room_type, arrival_date):
    # BAREX exclusively for these room types
    if room_type in {"KCDX", "TCDX", "KCST"}:
        return "BAREX", ""

    occ = OCCUPANCY_BY_DATE.get(arrival_date)
    if occ is not None and occ > HIGH_OCC_RATE_THRESHOLD:
        candidate_pool = HIGH_OCC_ALLOWED_RATES
    else:
        candidate_pool = RATE_CODES_POOL

    rate = random.choice(candidate_pool)

    if rate == "COR25":
        company = random.choice(["Deloitte", "Saudi Aramco", "GrupoACS", "Volkswagen"])
    else:
        company = random.choice(COMPANIES) if random.random() < 0.7 else ""

    return rate, company


def advance_cursor_one_day(state):
    cd = datetime.strptime(state["cursor_date"], "%Y-%m-%d").date()
    state["cursor_date"] = (cd + timedelta(days=1)).strftime("%Y-%m-%d")


def pick_arrival_date_progressive(state, today, added_pct_by_date):
    """
    Progressive fill:
    - Start from cursor_date (>= tomorrow).
    - Return the earliest date that still has headroom for at least 1 night.
    - If a date is full, advance day by day.
    """
    cursor = datetime.strptime(state["cursor_date"], "%Y-%m-%d").date()
    if cursor <= today:
        cursor = today + timedelta(days=1)
        state["cursor_date"] = cursor.strftime("%Y-%m-%d")

    # Safety limit: don't loop forever
    for _ in range(10000):
        proj = projected_occupancy(cursor, added_pct_by_date)
        if proj is not None and proj + DELTA_PCT_PER_BOOKING_NIGHT <= MAX_OCCUPANCY_PCT:
            state["cursor_date"] = cursor.strftime("%Y-%m-%d")
            return cursor

        cursor += timedelta(days=1)
        state["cursor_date"] = cursor.strftime("%Y-%m-%d")

    raise RuntimeError("Could not find any date with remaining headroom under the occupancy cap.")

# ============================================================
# Row generation (progressive + safe)
# ============================================================

def generate_row(state, today, added_pct_by_date):
    """
    Generates one reservation, filling dates progressively, without exceeding MAX_OCCUPANCY_PCT.
    Prefers shorter stays first (1..10) so we can always fit.
    """
    for _ in range(200):  # retry budget
        arrival = pick_arrival_date_progressive(state, today, added_pct_by_date)

        # Prefer shorter stays first
        stay_len = None
        for candidate_len in range(1, 11):
            if can_place_stay(arrival, candidate_len, added_pct_by_date):
                stay_len = candidate_len
                break

        if stay_len is None:
            # Arrival date can't take even a 1-night stay; move forward
            advance_cursor_one_day(state)
            continue

        # Commit occupancy impact for all nights
        commit_stay(arrival, stay_len, added_pct_by_date)

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
            "Preferences": pick_preference(),
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

    raise RuntimeError("Could not place a reservation without exceeding the occupancy cap.")

# ============================================================
# Main
# ============================================================

def main():
    if not OCCUPANCY_BY_DATE:
        raise RuntimeError("No occupancy data loaded. Cannot run progressive fill safely.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    state = load_state()
    today = datetime.today().date()

    # Track added occupancy during THIS run so we never exceed cap
    added_pct_by_date = {}

    rows = []
    for i in range(CONFIG["rows_per_run"]):
        rows.append(generate_row(state, today=today, added_pct_by_date=added_pct_by_date))

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

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    # Persist state every run
    save_state(state)

    print(f"✅ Generated {len(rows)} reservations -> {out_path}")
    print(f"🔢 Last ExternalID stored in state.json: {state['last_external_id']}")
    print(f"📅 Next progressive cursor_date: {state['cursor_date']}")
    print(f"🏨 Cap enforced: {MAX_OCCUPANCY_PCT}% | Rooms: {HOTEL_ROOMS} | Delta/night: {DELTA_PCT_PER_BOOKING_NIGHT:.3f}%")

if __name__ == "__main__":
    main()
