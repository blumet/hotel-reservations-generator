# Revenue Management Pricing Engine – README

## Purpose

This document defines the **commercial logic, pricing rules, and extension guidelines** for the pricing engine that generates PMS-ready rate tables.

It is the **single source of truth** for how prices are calculated. Any future changes (new rate plans, room types, packages, or logic tweaks) must follow the rules defined here.

---

## 1. Core Principles

1. **BAR00 is the main demand signal**

   * It reacts the most to occupancy, events, and weekends.
   * All other public rates are derived from BAR00.

2. **Packages follow demand, but do not lead it**

   * Packages are calmer than BAR00.
   * They use tiered premiums instead of continuous oscillation.

3. **Negotiated / staff rates are non-dynamic**

   * They do NOT change with occupancy, weekends, or events.

4. **Date bands are allowed**

   * A new band is created only when the final price changes.

5. **All prices are rounded to whole EUR**

---

## 2. Occupancy & Demand Inputs

### Occupancy Source

* Hotel-wide occupancy derived from `BusinessOnTheBooks`

### Effective Occupancy (optional)

If your OTB file provides components like pickup, wash (cancellations/no-shows), or group rooms, you **may** compute an **effective occupancy** before pricing:

* NetOTB = RoomsSold + GroupRooms
* ExpectedWash = NetOTB × (CancelRate + NoShowRate)
* EffectiveOTB = NetOTB − ExpectedWash
* ForecastOcc = EffectiveOTB / Capacity

**If any of these inputs are missing or unclear, do not invent multipliers.** In that case, use the simplest hotel-wide occupancy available in `BusinessOnTheBooks`.

### High Demand Definition (GLOBAL)

High demand applies when:

* **Occupancy ≥ 75%**, OR
* **High event day** (from `event.py`)

This trigger is used consistently across all rate plans.

---

## 3. BAR00 Pricing Logic (Base Rate)

### Low Occupancy Bands (KGDX reference)

| Occupancy | BAR00 (KGDX) |
| --------- | ------------ |
| ≤30%      | 489          |
| 31–50%    | 520          |
| 51–65%    | 560          |

### High Occupancy Ramp (66–99%)

BAR00 becomes continuous and more elastic:

```
KGDX_BAR00 = round(610 + (occ - 66) * 179 / 33)
```

* 66% → 610
* 99% → 789 (ceiling for KGDX)

### Weekend Effect

* **Friday & Saturday**
* Standard rooms: **+12%**
* Club rooms: **+15%**

### Event Effect

* Applied using factors from `event.py`
* Club rooms react more strongly than standard rooms

---

## 4. Room Type Pricing

All room prices are derived from **KGDX BAR00**, except PRE.

### Room Premiums (vs KGDX)

| Room Type | Premium          |
| --------- | ---------------- |
| KGDX      | +0               |
| KGSP      | +75              |
| TWSP      | +89              |
| TWDX      | +120             |
| KINGR     | +189             |
| KCDX      | +250             |
| TCDX      | +250             |
| KGST      | +350             |
| A1KB      | +450             |
| PRE       | **21,000 fixed** |

### Club Rooms (KCDX / TCDX)

* Use a **steeper occupancy ramp** between 66–99%

```
ClubRamp = round(610 + (occ - 66) * 279 / 33)
Final = ClubRamp + 250
```

* Club rooms also use **stronger weekend & event effects**

---

## 5. Per-Person Pricing Rules

### Per-Person Room Types

* **A1KB**
* **KGST**

All other room types are flat-priced.

### Adult (A1) Ratios

* OneGuest: `0.703 × base`
* TwoGuest: `0.844 × base`
* ExtraGuest: `0.334 × base`

### Child (C1) Ratios

* OneGuest: `0.070 × base`
* TwoGuest: `0.097 × base`
* ExtraGuest: `0.105 × base`

---

## 6. Rate Plan Classification

### A. Fully Dynamic (Follow BAR00 exactly)

These plans react fully to occupancy, weekends, and events:

* **BAR00**
* **RACK** (markup on BAR00)
* **BARAPT**
* **BARSUIT**

### B. Follower Packages (Tiered, Calm)

These follow BAR00 but only switch between **low / high tiers**:

| Plan | Low Tier | High Tier | Notes           |
| ---- | -------: | --------: | --------------- |
| BB   |      +18 |       +25 | Per person      |
| HB   |      +55 |       +75 | Per person      |
| SUHB |     +175 |      +255 | HB + premium    |
| WEL3 |     +150 |      +250 | WEL-lite        |
| WEL  |     +250 |      +350 | Highest package |

**Tier selection:**

* Low tier: Occupancy <75% AND no high event
* High tier: Occupancy ≥75% OR high event

### C. Non-Dynamic / Negotiated Rates

These DO NOT react to demand:

* **CORL25** → 25% off BAR00 (static reference)
* **OPQ** → 15% off BAR00 (static)
* **STAFF** → 50% off BAR00 (static)

---

## 7. How to Add a New Rate Plan

When adding a new rate plan, you must answer **all** of the following:

1. **Category**

   * Fully Dynamic
   * Follower Package
   * Non-Dynamic / Negotiated

2. **Pricing Basis**

   * Flat
   * Per-person (A1 + C1)

3. **Premium Logic**

   * Fixed markup
   * Tiered (low/high)
   * Percentage of BAR00

4. **Demand Sensitivity**

   * Full (like BAR00)
   * Tiered only
   * None

5. **Room-Type Applicability**

   * All rooms
   * Specific room types only

> If any of these are undefined, the plan must NOT be added.

---

## 8. Banding Strategy (to be implemented next)

* Date bands are allowed
* A new band starts when:

  * Final price changes by ≥ 1 EUR
* Bands must never overlap

---

## 9. Non-Negotiables (Safety Rules)

* Currency: **EUR**
* Rounding: **whole EUR only**
* No negative or zero prices
* PRE is always fixed at 21,000
* Unexpected room types or rate plans should fail validation

---

## 10. Change Management

Any future change must:

1. Be reflected in this README
2. Specify whether it affects:

   * Demand logic
   * Room pricing
   * Package behavior
3. Be validated against row count impact

---

**This document exists to prevent logic drift and uncontrolled complexity.**
