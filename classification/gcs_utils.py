"""
Google Cloud Storage helpers for reading trail data and writing predictions.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone

from google.cloud import storage

from .config import (
    GCS_BUCKET_OUTPUT,
    GCS_BUCKET_RAW,
    GCS_BUCKET_WEATHER,
    GCS_INPUT_PREFIX,
    GCS_PRED_PREFIX,
    GCS_WEATHER_PREFIX,
    REPORT_MAX_AGE_DAYS,
)


def parse_report_date(date_str: str | None) -> datetime | None:
    """Parse date_hiked from trip reports. Handles multiple WTA formats."""
    if not date_str:
        return None
    cleaned = re.sub(r"^[A-Za-z]+,\s*", "", date_str)
    cleaned = cleaned.replace(".", "")
    for fmt in ("%b %d, %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(cleaned.strip(), fmt)
        except ValueError:
            continue
    return None


def get_gcs_client() -> storage.Client:
    return storage.Client()


def list_hike_ids(client: storage.Client, bucket_name: str, prefix: str) -> list[str]:
    """List all hike IDs that have a reports.jsonl file in GCS."""
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix + "/")
    hike_ids = set()
    for blob in blobs:
        parts = blob.name.removeprefix(prefix + "/").split("/")
        if len(parts) == 2 and parts[1] == "reports.jsonl":
            hike_ids.add(parts[0])
    return sorted(hike_ids)


def load_reports_from_gcs(
    client: storage.Client,
    bucket_name: str,
    prefix: str,
    max_hikes: int | None = None,
    only_hike_ids: set[str] | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    Download reports.jsonl + metadata.json for each hike from GCS.

    Args:
        only_hike_ids: If provided, only load these hike IDs (for incremental runs).

    Returns:
        recent_reports: reports within the last REPORT_MAX_AGE_DAYS days
        stale_hikes: metadata dicts for hikes with no recent reports
    """
    bucket = client.bucket(bucket_name)
    hike_ids = list_hike_ids(client, bucket_name, prefix)
    if only_hike_ids:
        hike_ids = [h for h in hike_ids if h in only_hike_ids]
    if max_hikes:
        hike_ids = hike_ids[:max_hikes]

    print(f"Found {len(hike_ids)} hikes in gs://{bucket_name}/{prefix}/")

    cutoff = datetime.now() - timedelta(days=REPORT_MAX_AGE_DAYS)
    recent_reports = []
    stale_hikes = []

    for hike_id in hike_ids:
        # Load metadata
        meta_blob = bucket.blob(f"{prefix}/{hike_id}/metadata.json")
        if not meta_blob.exists():
            continue
        metadata = json.loads(meta_blob.download_as_text())

        # Shared metadata fields
        hike_meta = {
            "hike_id": hike_id,
            "hike_name": metadata.get("name", hike_id),
            "hike_region": metadata.get("region"),
            "elevation_gain": metadata.get("elevation_gain"),
            "highest_point": metadata.get("highest_point"),
        }

        # Load reports
        reports_blob = bucket.blob(f"{prefix}/{hike_id}/reports.jsonl")
        if not reports_blob.exists():
            stale_hikes.append({**hike_meta, "most_recent_report_date": None})
            continue

        content = reports_blob.download_as_text().strip()
        if not content:
            stale_hikes.append({**hike_meta, "most_recent_report_date": None})
            continue

        has_recent = False
        most_recent_date = None
        for line in content.split("\n"):
            line = line.strip()
            if not line:
                continue
            report = json.loads(line)
            report_date = parse_report_date(report.get("date_hiked"))

            if report_date and (most_recent_date is None or report_date > most_recent_date):
                most_recent_date = report_date

            if report_date and report_date >= cutoff:
                report.update(hike_meta)
                report["classification_source"] = "report+weather"
                recent_reports.append(report)
                has_recent = True

        if not has_recent:
            stale_hikes.append({
                **hike_meta,
                "most_recent_report_date": most_recent_date.strftime("%Y-%m-%d") if most_recent_date else None,
            })

    print(f"Recent reports (last {REPORT_MAX_AGE_DAYS} days): {len(recent_reports)}")
    print(f"Hikes with no recent reports: {len(stale_hikes)}")
    return recent_reports, stale_hikes


def upload_predictions_to_gcs(
    client: storage.Client,
    predictions: list[dict],
    mode: str = "incremental",
    batch_num: int | None = None,
) -> str:
    """Upload classified predictions to GCS as a JSON file."""
    bucket = client.bucket(GCS_BUCKET_OUTPUT)
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if batch_num is not None:
        filename = f"batch_{batch_num:03d}.json"
    else:
        filename = f"{mode}.json"
    blob_path = f"{GCS_PRED_PREFIX}/{date_str}/{filename}"
    blob = bucket.blob(blob_path)
    blob.upload_from_string(
        json.dumps(predictions, indent=2, ensure_ascii=False),
        content_type="application/json",
    )
    print(f"Uploaded predictions -> gs://{GCS_BUCKET_OUTPUT}/{blob_path}")
    return f"gs://{GCS_BUCKET_OUTPUT}/{blob_path}"


def get_last_run_timestamp(client: storage.Client) -> datetime | None:
    """
    Check the most recent prediction file in GCS to determine
    when the last classification run happened.
    """
    bucket = client.bucket(GCS_BUCKET_OUTPUT)
    blobs = list(bucket.list_blobs(prefix=f"{GCS_PRED_PREFIX}/"))
    if not blobs:
        return None
    most_recent = max(blobs, key=lambda b: b.updated)
    return most_recent.updated.replace(tzinfo=None)


def find_changed_hike_ids(client: storage.Client, since: datetime) -> set[str]:
    """
    Find hike IDs that have new or updated data since `since`.
    Checks both the reports bucket and the weather bucket.
    """
    changed = set()

    # Check for new/updated reports
    raw_bucket = client.bucket(GCS_BUCKET_RAW)
    for blob in raw_bucket.list_blobs(prefix=GCS_INPUT_PREFIX + "/"):
        if blob.name.endswith("reports.jsonl") and blob.updated.replace(tzinfo=None) > since:
            parts = blob.name.removeprefix(GCS_INPUT_PREFIX + "/").split("/")
            if len(parts) == 2:
                changed.add(parts[0])

    # Check for new weather data
    weather_bucket = client.bucket(GCS_BUCKET_WEATHER)
    for blob in weather_bucket.list_blobs(prefix=GCS_WEATHER_PREFIX + "/"):
        if blob.name.endswith("weather.json") and blob.updated.replace(tzinfo=None) > since:
            parts = blob.name.split("/")
            for i, part in enumerate(parts):
                if part == "weather.json" and i >= 1:
                    changed.add(parts[i - 1])

    return changed
