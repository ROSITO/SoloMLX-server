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
    parser = argparse.ArgumentParser(description="Rebuild analytics pipeline for one player.")
    parser.add_argument("--player-id", type=int, required=True)
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.info("Rebuild player pipeline player_id=%s days=%s", args.player_id, args.days)

    raw_df = read_wellness_last_days(args.days)
    gps_raw_df = read_gps_last_days(args.days)
    player_df = raw_df[raw_df["player_id"] == args.player_id]
    gps_player_df = gps_raw_df[gps_raw_df["player_id"] == args.player_id]
    validate_input(player_df)
    normalized_df = normalize_wellness(player_df)
    features_df = build_daily_features(normalized_df, gps_player_df)

    if args.dry_run:
        logging.info("Dry-run mode: player feature rows=%s", len(features_df))
        return

    inserted = write_daily_features(features_df)
    alerts = build_alert_records(normalized_df, features_df, gps_player_df)
    reports = build_report_records(normalized_df, features_df, gps_player_df)
    upsert_alerts(alerts)
    upsert_reports(reports)
    logging.info("Player rebuild completed. features_upserted=%s", inserted)


if __name__ == "__main__":
    main()
