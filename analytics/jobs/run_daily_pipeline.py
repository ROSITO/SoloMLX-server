#!/usr/bin/env python3
import argparse
import logging

from analytics.src.alerts.alert_engine import build_alert_records
from analytics.src.extract.mysql_reader import read_gps_last_days, read_wellness_last_days
from analytics.src.load.alerts_writer import upsert_alerts
from analytics.src.load.feature_store_writer import write_daily_features
from analytics.src.load.reports_writer import upsert_reports
from analytics.src.reports.player_report_builder import build_report_records
from analytics.src.transform.features import build_daily_features
from analytics.src.transform.normalize import normalize_wellness
from analytics.src.transform.quality_checks import validate_input


def main():
    parser = argparse.ArgumentParser(description="Run SafePerform daily analytics pipeline.")
    parser.add_argument("--days", type=int, default=30, help="Lookback window in days.")
    parser.add_argument("--dry-run", action="store_true", help="Do not write to database.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.info("Starting daily pipeline: lookback=%s days dry_run=%s", args.days, args.dry_run)

    raw_df = read_wellness_last_days(args.days)
    gps_raw_df = read_gps_last_days(args.days)
    validate_input(raw_df)
    normalized_df = normalize_wellness(raw_df)
    features_df = build_daily_features(normalized_df, gps_raw_df)

    logging.info(
        "Raw wellness rows=%s raw gps rows=%s normalized rows=%s feature rows=%s",
        len(raw_df),
        len(gps_raw_df),
        len(normalized_df),
        len(features_df),
    )

    if args.dry_run:
        logging.info("Dry-run mode: no data written.")
        return

    inserted = write_daily_features(features_df)
    alerts = build_alert_records(normalized_df, features_df, gps_raw_df)
    reports = build_report_records(normalized_df, features_df, gps_raw_df)
    n_alerts = upsert_alerts(alerts)
    n_reports = upsert_reports(reports)
    logging.info(
        "Pipeline completed. features_upserted=%s alerts_upserted=%s reports_upserted=%s",
        inserted,
        n_alerts,
        n_reports,
    )


if __name__ == "__main__":
    main()
