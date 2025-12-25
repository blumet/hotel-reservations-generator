def write_pricing_csv(
    baseline_csv_path: str,
    appended_rows_df: pd.DataFrame,
    output_csv_path: str,
) -> None:
    base = pd.read_csv(baseline_csv_path, dtype=str).fillna("")
    base = base[SCHEMA].copy()

    # If no new rows, output baseline
    if appended_rows_df is None or appended_rows_df.empty:
        base.to_csv(output_csv_path, index=False, quoting=csv.QUOTE_MINIMAL)
        return

    # Parse dates
    base_dt = base.copy()
    base_dt["StartDate"] = pd.to_datetime(base_dt["StartDate"], errors="coerce")
    base_dt["EndDate"] = pd.to_datetime(base_dt["EndDate"], errors="coerce")

    new_dt = appended_rows_df.copy()
    new_dt["StartDate"] = pd.to_datetime(new_dt["StartDate"], errors="coerce")
    new_dt["EndDate"] = pd.to_datetime(new_dt["EndDate"], errors="coerce")

    # Horizon covered by new rows
    horizon_start = new_dt["StartDate"].min()
    horizon_end = new_dt["EndDate"].max()

    # Keys that we are generating
    key_cols = ["Currency", "RoomTypeCode", "RatePlanCode", "AgeBucketCode"]
    gen_keys = set(map(tuple, new_dt[key_cols].drop_duplicates().values.tolist()))

    # Remove baseline rows that overlap horizon for those same keys
    def should_drop(row) -> bool:
        key = (row["Currency"], row["RoomTypeCode"], row["RatePlanCode"], row["AgeBucketCode"])
        if key not in gen_keys:
            return False
        if pd.isna(row["StartDate"]) or pd.isna(row["EndDate"]):
            return False
        return (row["StartDate"] < horizon_end) and (row["EndDate"] > horizon_start)

    drop_mask = base_dt.apply(should_drop, axis=1)
    base_kept = base.loc[~drop_mask].copy()

    out = pd.concat([base_kept, appended_rows_df], ignore_index=True)
    out.to_csv(output_csv_path, index=False, quoting=csv.QUOTE_MINIMAL)
