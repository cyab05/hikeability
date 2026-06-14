"""
Google Cloud Storage helpers for reading trail data and writing predictions.
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
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
        # Load metadata (optional — missing metadata won't drop the trail)
        meta_blob = bucket.blob(f"{prefix}/{hike_id}/metadata.json")
        metadata = {}
        if meta_blob.exists():
            metadata = json.loads(meta_blob.download_as_text())

        # Shared metadata fields
        hike_meta = {
            "hike_id": hike_id,
            "hike_name": metadata.get("name", hike_id),
            "hike_region": metadata.get("region"),
            "elevation_gain": metadata.get("elevation_gain"),
            "highest_point": metadata.get("highest_point"),
            "latitude": metadata.get("latitude"),
            "longitude": metadata.get("longitude"),
            "url": metadata.get("url"),
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

        recent_for_trail = []
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
                recent_for_trail.append(report)

        # Sort reports newest-first
        recent_for_trail.sort(
            key=lambda r: parse_report_date(r.get("date_hiked")) or datetime.min,
            reverse=True,
        )

        if recent_for_trail:
            recent_reports.append({
                **hike_meta,
                "reports": recent_for_trail,
                "most_recent_report_date": most_recent_date.strftime("%Y-%m-%d") if most_recent_date else None,
                "classification_source": "report+weather",
            })
        else:
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


# Earliest daily-run date the snapshot includes. Mirrors MIN_DATE in app/gcs.py —
# runs before this had incomplete coverage.
_SNAPSHOT_MIN_DATE = "2026-05-01"
_SNAPSHOT_PATH = f"{GCS_PRED_PREFIX}/latest.json"


def write_latest_snapshot(client: storage.Client) -> str:
    """
    Merge every daily prediction folder (>= _SNAPSHOT_MIN_DATE) into a single
    predictions/latest.json blob, enriched with metadata from the raw bucket
    so the web app can serve cold-start traffic with one GCS read.
    """
    output = client.bucket(GCS_BUCKET_OUTPUT)

    # 1. Find date folders to include.
    blobs = output.list_blobs(prefix=f"{GCS_PRED_PREFIX}/", delimiter="/")
    list(blobs)  # populate .prefixes
    date_prefixes = sorted(
        p for p in blobs.prefixes
        if p.rstrip("/").split("/")[-1] >= _SNAPSHOT_MIN_DATE
    )
    if not date_prefixes:
        print("No prediction folders to snapshot.")
        return ""

    # 2. Collect every prediction JSON across all dates.
    pred_blobs: list[tuple[str, storage.Blob]] = []
    for prefix in date_prefixes:
        for blob in output.list_blobs(prefix=prefix):
            if blob.name.endswith(".json"):
                pred_blobs.append((prefix, blob))

    def _download(item):
        prefix, blob = item
        try:
            return prefix, json.loads(blob.download_as_text())
        except Exception:
            return prefix, None

    with ThreadPoolExecutor(max_workers=32) as ex:
        results = list(ex.map(_download, pred_blobs))

    # 3. Merge by hike_id, newer date wins.
    results.sort(key=lambda r: r[0])
    merged: dict[str, dict] = {}
    for _prefix, data in results:
        if isinstance(data, list):
            for p in data:
                if p.get("hike_id"):
                    merged[p["hike_id"]] = p

    predictions = list(merged.values())
    print(f"Merged {len(predictions)} unique hikes across {len(date_prefixes)} date folders.")

    # 4. Enrich from raw metadata so the web app never has to.
    raw_bucket = client.bucket(GCS_BUCKET_RAW)

    def _needs_meta(p: dict) -> bool:
        return not (
            p.get("latitude") is not None and p.get("longitude") is not None
            and p.get("distance") and p.get("rating") and p.get("url")
            and p.get("elevation_gain") and p.get("highest_point") and p.get("hike_name")
            and p.get("image_url") and p.get("difficulty")
            and "parking_pass" in p and "closure_warning" in p
        )

    needs_fetch = [p for p in predictions if _needs_meta(p)]

    def _fetch_meta(p: dict):
        blob = raw_bucket.blob(f"{GCS_INPUT_PREFIX}/{p['hike_id']}/metadata.json")
        try:
            return p, json.loads(blob.download_as_text())
        except Exception:
            return p, None

    with ThreadPoolExecutor(max_workers=64) as ex:
        for p, meta in ex.map(_fetch_meta, needs_fetch):
            if not meta:
                continue
            for src, dst in (
                ("latitude", "latitude"), ("longitude", "longitude"),
                ("name", "hike_name"), ("url", "url"),
                ("elevation_gain", "elevation_gain"), ("highest_point", "highest_point"),
                ("distance", "distance"), ("rating", "rating"),
                ("image_url", "image_url"), ("difficulty", "difficulty"),
            ):
                if not p.get(dst):
                    val = meta.get(src)
                    if val is not None:
                        p[dst] = val
            if p.get("parking_pass") is None:
                p["parking_pass"] = meta.get("parking_pass")
            if p.get("closure_warning") is None:
                p["closure_warning"] = meta.get("closure_warning") or []

    # 5. Upload as predictions/latest.json (no indentation to keep payload small).
    blob = output.blob(_SNAPSHOT_PATH)
    blob.upload_from_string(
        json.dumps(predictions, ensure_ascii=False),
        content_type="application/json",
    )
    uri = f"gs://{GCS_BUCKET_OUTPUT}/{_SNAPSHOT_PATH}"
    print(f"Wrote snapshot -> {uri} ({len(predictions)} hikes)")
    return uri


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
        if blob.name.endswith("open_meteo_24h.json") and blob.updated.replace(tzinfo=None) > since:
            parts = blob.name.split("/")
            for i, part in enumerate(parts):
                if part == "open_meteo_24h.json" and i >= 1:
                    changed.add(parts[i - 1])

    return changed
