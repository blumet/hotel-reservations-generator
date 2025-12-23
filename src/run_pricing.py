# /src/run_pricing.py
"""
Revenue Autopilot runner.

What it does (v1):
- Loads /src/config.yml
- Loads /data/events.yaml
- Loads OTB from BusinessOnTheBooks.csv
- Loads pricing template from RatePricing-Template.csv (or RatePricing-Template.csv in repo root if you keep it there)
- Computes date-level multipliers (occ + pickup + wash + groups + events + guardrails)
- Regenerates pricing rows for the configured horizon (default 90 days)
- Overwrites the output pricing file (default RatePricing.csv) and writes a change log file

This file is intentionally small. The real logic lives in /src/lib/*.py.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from typing import Dict, Any, Tuple

import yaml


def _repo_root() -> str:
    """
    Resolve repo root assuming this file is /src/run_pricing.py.
    """
    here = os.path.abspath(os.path.dirname(__file__))
    return os.path.abspath(os.path.join(here, ".."))


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Revenue Autopilot - Occupancy-based pricing generator")
    p.add_argument("--config", default="src/config.yml", help="Path to config.yml (relative to repo root)")
    p.add_argument("--events", default="data/events.yaml", help="Path to events.yaml (relative to repo root)")
    p.add_argument("--otb", default=None, help="Override OTB path; otherwise uses config.otb.file")
    p.add_argument("--template", default=None, help="Override template path; otherwise tries data/RatePricing-Template.csv then RatePricing-Template.csv")
    p.add_argument("--output", default=None, help="Override output pricing file path; otherwise uses config.output.pricing_file")
    p.add_argument("--changes", default=None, help="Override change log path; otherwise uses config.output.changes_file")
    p.add_argument("--as_of", default=None, help="Override run date YYYY-MM-DD (default: today)")
    return p.parse_args()


def _resolve_paths(repo: str, cfg: Dict[str, Any], args: argparse.Namespace) -> Tuple[str, str, str, str, str]:
    """
    Returns: (events_path, otb_path, template_path, pricing_out_path, changes_out_path)
    """
    events_path = os.path.join(repo, args.events)

    otb_file = args.otb or cfg["otb"]["file"]
    otb_path = otb_file if os.path.isabs(otb_file) else os.path.join(repo, otb_file)

    # Template: allow override, else try data/RatePricing-Template.csv then RatePricing-Template.csv
    if args.template:
        template_path = args.template if os.path.isabs(args.template) else os.path.join(repo, args.template)
    else:
        candidate1 = os.path.join(repo, "data", "RatePricing-Template.csv")
        candidate2 = os.path.join(repo, "RatePricing-Template.csv")
        template_path = candidate1 if os.path.exists(candidate1) else candidate2

    pricing_out = args.output or cfg["output"]["pricing_file"]
    pricing_out_path = pricing_out if os.path.isabs(pricing_out) else os.path.join(repo, pricing_out)

    changes_out = args.changes or cfg["output"]["changes_file"]
    changes_out_path = changes_out if os.path.isabs(changes_out) else os.path.join(repo, changes_out)

    return events_path, otb_path, template_path, pricing_out_path, changes_out_path


def main() -> int:
    repo = _repo_root()
    args = _parse_args()

    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(repo, args.config)
    if not os.path.exists(cfg_path):
        print(f"[ERROR] config not found: {cfg_path}", file=sys.stderr)
        return 2

    cfg = _load_yaml(cfg_path)

    events_path, otb_path, template_path, pricing_out_path, changes_out_path = _resolve_paths(repo, cfg, args)

    if not os.path.exists(events_path):
        print(f"[ERROR] events.yaml not found: {events_path}", file=sys.stderr)
        return 2
    if not os.path.exists(otb_path):
        print(f"[ERROR] OTB file not found: {otb_path}", file=sys.stderr)
        return 2
    if not os.path.exists(template_path):
        print(f"[ERROR] template not found: {template_path}", file=sys.stderr)
        return 2

    if args.as_of:
        as_of = dt.date.fromisoformat(args.as_of)
    else:
        as_of = dt.date.today()

    horizon_days = int(cfg.get("pricing_horizon_days", 90))
    start_date = as_of
    end_date = as_of + dt.timedelta(days=horizon_days)

    # Lazy imports so this script can show helpful file-not-found errors even if deps are missing.
    from lib.otb import load_otb_dataframe, build_daily_metrics
    from lib.events import load_events, build_event_multiplier_series
    from lib.pricing import (
        load_pricing_template,
        compute_multipliers,
        generate_pricing_rows,
        write_pricing_csv,
        write_changes_csv,
    )

    print("[INFO] Loading inputs...")
    events = load_events(events_path)
    otb_df = load_otb_dataframe(otb_path, cfg)
    daily = build_daily_metrics(otb_df, cfg, as_of=as_of)

    # Event multipliers for the horizon window
    event_mult = build_event_multiplier_series(events, start_date=start_date, end_date=end_date, cfg=cfg)

    print("[INFO] Loading pricing template...")
    template = load_pricing_template(template_path)

    print("[INFO] Computing multipliers...")
    mult_df = compute_multipliers(daily, event_mult, cfg, as_of=as_of)

    print("[INFO] Generating pricing rows...")
    new_rows_df = generate_pricing_rows(
        template_df=template,
        multipliers_df=mult_df,
        cfg=cfg,
        start_date=start_date,
        end_date=end_date,
    )

    # For change tracking, compare against existing output if present; else compare to template for overlap window.
    baseline_path = pricing_out_path if os.path.exists(pricing_out_path) else template_path

    print("[INFO] Writing outputs...")
    write_pricing_csv(
        baseline_csv_path=baseline_path,
        appended_rows_df=new_rows_df,
        output_csv_path=pricing_out_path,
    )

    write_changes_csv(
        baseline_csv_path=baseline_path,
        appended_rows_df=new_rows_df,
        changes_csv_path=changes_out_path,
        cfg=cfg,
        as_of=as_of,
    )

    print(f"[OK] Wrote pricing: {os.path.relpath(pricing_out_path, repo)}")
    print(f"[OK] Wrote changes: {os.path.relpath(changes_out_path, repo)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
