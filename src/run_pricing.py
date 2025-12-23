# /src/run_pricing.py
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from typing import Dict, Any

import yaml


def repo_root() -> str:
    here = os.path.abspath(os.path.dirname(__file__))
    return os.path.abspath(os.path.join(here, ".."))


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="src/config.yml", help="Path to config.yml (relative to repo root)")
    p.add_argument("--as_of", default=None, help="Override run date YYYY-MM-DD (default: today)")
    return p.parse_args()


def rpath(repo: str, maybe_rel: str) -> str:
    return maybe_rel if os.path.isabs(maybe_rel) else os.path.join(repo, maybe_rel)


def main() -> int:
    repo = repo_root()
    args = parse_args()

    cfg_path = rpath(repo, args.config)
    if not os.path.exists(cfg_path):
        print(f"[ERROR] Missing config: {cfg_path}", file=sys.stderr)
        return 2

    cfg = load_yaml(cfg_path)

    as_of = dt.date.fromisoformat(args.as_of) if args.as_of else dt.date.today()
    horizon_days = int(cfg.get("pricing_horizon_days", 90))
    start_date = as_of
    end_date = as_of + dt.timedelta(days=horizon_days)

    # Resolve paths from config (matches your /data layout)
    otb_path = rpath(repo, cfg["otb"]["file"])
    events_path = rpath(repo, cfg["events"]["file"])
    template_path = rpath(repo, cfg["template"]["file"])
    out_pricing_path = rpath(repo, cfg["output"]["pricing_file"])
    out_changes_path = rpath(repo, cfg["output"]["changes_file"])

    for pth, label in [
        (otb_path, "OTB"),
        (events_path, "events"),
        (template_path, "template"),
    ]:
        if not os.path.exists(pth):
            print(f"[ERROR] Missing {label} file: {pth}", file=sys.stderr)
            return 2

    # Imports from src/lib
    from lib.otb import load_otb_dataframe, build_daily_metrics
    from lib.events import load_events, build_event_multiplier_series
    from lib.pricing import (
        load_pricing_template,
        compute_multipliers,
        generate_pricing_rows,
        write_pricing_csv,
        write_changes_csv,
    )

    print("[INFO] Loading events...")
    events = load_events(events_path)

    print("[INFO] Loading OTB...")
    otb_df = load_otb_dataframe(otb_path, cfg)

    print("[INFO] Building daily metrics...")
    daily = build_daily_metrics(otb_df, cfg, as_of=as_of)

    print("[INFO] Building event multipliers...")
    event_mult = build_event_multiplier_series(events, start_date=start_date, end_date=end_date, cfg=cfg)

    print("[INFO] Loading template...")
    template_df = load_pricing_template(template_path)

    print("[INFO] Computing final multipliers...")
    mult_df = compute_multipliers(daily, event_mult, cfg, as_of=as_of)

    print("[INFO] Generating pricing rows...")
    new_rows = generate_pricing_rows(
        template_df=template_df,
        multipliers_df=mult_df,
        cfg=cfg,
        start_date=start_date,
        end_date=end_date,
    )

    # Overwrite output file with: template + new rows
    print("[INFO] Writing pricing output (overwrite)...")
    write_pricing_csv(
        baseline_csv_path=template_path,
        appended_rows_df=new_rows,
        output_csv_path=out_pricing_path,
    )

    print("[INFO] Writing change log...")
    write_changes_csv(
        baseline_csv_path=template_path,
        appended_rows_df=new_rows,
        changes_csv_path=out_changes_path,
        cfg=cfg,
        as_of=as_of,
    )

    print(f"[OK] Wrote: {os.path.relpath(out_pricing_path, repo)}")
    print(f"[OK] Wrote: {os.path.relpath(out_changes_path, repo)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
