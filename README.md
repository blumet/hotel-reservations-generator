# Revenue Autopilot (Barcelona) — Occupancy-Based Pricing

This repo contains a small revenue management automation that updates hotel pricing based on:
- Occupancy forecast from `BusinessOnTheBooks.csv`
- Pickup trend (bookings moving up/down)
- Expected wash (cancellations + no-shows)
- Group displacement (group blocks pushing compression)
- Barcelona event calendar (congresses/festivals/holidays)

It runs on GitHub Actions and overwrites the existing pricing output file in-place.

## Files

### Inputs
- `BusinessOnTheBooks.csv`
  - On-the-books (OTB) data used to forecast occupancy and pickup.
- `events.yaml` (or `events.json`)
  - Editable calendar of major demand events (MWC, ISE, festivals, holidays).
- `RatePricing-Template.csv`
  - Pricing template used as source-of-truth to preserve structure and derive plans.

### Outputs
- `RatePricing.csv`
  - The main pricing output file. **This workflow overwrites this file using the same name/path.**
- `RatePricing_changes.csv`
  - Human-readable summary of how rates changed (old/new/delta/reasons).

## Pricing Logic (High Level)

For each stay date (and optionally room type):
1. Compute effective occupancy:
   - NetOTB = RoomsSold + GroupRooms
   - ExpectedWash = NetOTB * (CancelRate + NoShowRate)
   - EffectiveOTB = NetOTB - ExpectedWash
   - ForecastOcc = EffectiveOTB / Capacity

2. Compute multipliers:
   - BaseOccMultiplier: derived from ForecastOcc thresholds
   - PickupMultiplier: adjusts based on recent pickup trend
   - WashMultiplier: small reduction when wash is unusually high
   - GroupMultiplier: increases transient pricing when groups compress inventory
   - EventMultiplier: increases pricing around major events + shoulders

3. Guardrails:
   - Max change per run (default ±8%)
   - Seasonal floors (e.g., summer not below 0.95)
   - Compression event lock (MWC/ISE never discounted below 1.10)

4. Pricing output:
   - Generate BAR00 for the target dates using the final multiplier.
   - Derive additional plans:
     - BAREX = BAR00 * 1.25
     - RACK = BAR00 * 1.35
     - CORL25 = BAR00 * 0.75
     - OPQ = BAR00 * 0.80
     - GRPBASE = BAR00 * 0.70
     - GRPHIG = GRPBASE * 1.25
     - STAFF = BAR00 * 0.50
     - WAL = BAR00 * 0.85
     - HB / WEL preserve template structure (ratios vs BAR00)
     - SUHB = HB * 1.25
     - BB = BAR00 + 35 per person
     - WEL3 = weekends-only, BAR00 * 1.35
     - BARAPT = BAR00 * 1.40 (A2KB/A1KB only)
     - BARSUIT = BAR00 * 1.25 (KCST/KGST/PRE only; PRE flat-only)

## Configuration
- `config.yml` controls:
  - Column mapping for `BusinessOnTheBooks.csv`
  - Thresholds and guardrails
  - Default cancel/no-show rates (if not provided in OTB)
  - Event weights (Peak/High/Medium)

## GitHub Actions
A scheduled workflow runs periodically:
- Reads inputs
- Generates new `RatePricing.csv`
- Writes `RatePricing_changes.csv`
- Commits and pushes updates

## Safety Notes
This is a rules-based RM autopilot designed to be practical and revenue-positive quickly.
For high-stakes use, validate outputs for 2 weeks before increasing aggressiveness.
