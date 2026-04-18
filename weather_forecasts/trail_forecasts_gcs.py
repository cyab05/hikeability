
"""
Read trail metadata from GCS (wta-hikes), grab open-meteo hourly forecast
per trail, and writes back to weather bucket in GCS.

Order of operations:
1. parse args and build the destination prefix (weather-scraped/YYYY-MM-DD/)
2. create the GCS client
3. list metadata blobs under output/hikes/
4. process trails concurrently: load metadata, fetch Open-Meteo, write to GCS

Reads from:
  gs://{source_bucket}/output/hikes/{trail_slug}/metadata.json

Writes to:
  gs://{dest_bucket}/{dest_root}/{scrape_date}/{trail_slug}/open_meteo_24h.json
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any

from google.cloud import storage
from google.oauth2 import service_account

import open_meteo_fetch as omf

DEFAULT_SOURCE_BUCKET = "wta-hikes"
HIKES_PREFIX = "output/hikes/"
DEFAULT_DEST_BUCKET = "weather-conditions"
DEFAULT_DEST_ROOT = "weather-scraped"
DEFAULT_MAX_TRAILS: int | None = None
DEFAULT_WORKERS = 20
USER_AGENT_NOTE = "trail_forecasts_gcs/1.0"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Read trail lat/lon from GCS metadata, write Open-Meteo 24h forecast to GCS.",
    )
    p.add_argument("--source-bucket", default=DEFAULT_SOURCE_BUCKET)
    p.add_argument("--dest-bucket", default=DEFAULT_DEST_BUCKET)
    p.add_argument(
        "--dest-root",
        default=DEFAULT_DEST_ROOT,
        help="Path under dest bucket before the dated folder (default: weather-scraped).",
    )
    p.add_argument(
        "--max-trails",
        type=int,
        default=DEFAULT_MAX_TRAILS,
        metavar="N",
        help="Process at most N trails (sorted by slug). Omit to process every trail.",
    )
    p.add_argument(
        "--credentials",
        default=None,
        help="Optional path to service account JSON. Otherwise uses Application Default Credentials.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        metavar="N",
        help=f"Concurrent trail workers (default: {DEFAULT_WORKERS}).",
    )
    return p.parse_args()


def gcs_client(credentials_path: str | None) -> storage.Client:
    if credentials_path:
        creds = service_account.Credentials.from_service_account_file(credentials_path)
        return storage.Client(credentials=creds, project=creds.project_id)
    return storage.Client()


def list_metadata_blob_names(
    bucket: storage.Bucket,
    *,
    max_trails: int | None,
) -> list[tuple[str, str]]:
    """
    Return (slug, metadata_blob_name) pairs under HIKES_PREFIX, sorted by trail slug.
    If max_trails is None, return every trail. Uses delimiter listing.
    """
    iterator = bucket.list_blobs(prefix=HIKES_PREFIX, delimiter="/")
    slugs: list[str] = []
    for page in iterator.pages:
        for p in page.prefixes:
            slug = p.rstrip("/").rsplit("/", 1)[-1]
            if slug:
                slugs.append(slug)
    slugs.sort()
    if max_trails is not None:
        slugs = slugs[:max_trails]
    out: list[tuple[str, str]] = []
    for slug in slugs:
        blob_name = f"{HIKES_PREFIX}{slug}/metadata.json"
        out.append((slug, blob_name))
    return out


def load_metadata_json(bucket: storage.Bucket, blob_name: str) -> dict[str, Any]:
    blob = bucket.blob(blob_name)
    return json.loads(blob.download_as_text(encoding="utf-8"))


def coords_from_metadata(metadata: dict[str, Any]) -> tuple[float, float]:
    """
    Resolve WGS84 lat/lon from trail metadata.json.
    Tries top-level latitude/longitude, lat/lon, nested location objects, and GeoJSON-style coordinates.
    """
    pairs = (
        ("latitude", "longitude"),
        ("lat", "lon"),
        ("lat", "lng"),
    )

    def try_dict(d: dict[str, Any]) -> tuple[float, float] | None:
        for la_k, lo_k in pairs:
            if la_k in d and lo_k in d and d[la_k] is not None and d[lo_k] is not None:
                return float(d[la_k]), float(d[lo_k])
        return None

    if hit := try_dict(metadata):
        return hit
    for key in ("location", "trail", "hike", "geometry", "coords", "coordinate"):
        sub = metadata.get(key)
        if isinstance(sub, dict) and (hit := try_dict(sub)):
            return hit
    coords = metadata.get("coordinates")
    if isinstance(coords, (list, tuple)) and len(coords) >= 2:
        lon, lat = float(coords[0]), float(coords[1])
        return lat, lon

    raise KeyError(
        "no coordinates found (tried latitude/longitude, lat/lon, nested location.*, coordinates[])",
    )


def forecast_payload(
    *,
    trail_slug: str,
    source_gs_uri: str,
    metadata: dict[str, Any],
    df,
) -> dict[str, Any]:
    fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    records = json.loads(df.to_json(orient="records", date_format="iso"))
    return {
        "trail_slug": trail_slug,
        "source_metadata_gs_uri": source_gs_uri,
        "fetched_at_utc": fetched_at,
        "open_meteo_timezone": df.attrs.get("timezone", omf.TIMEZONE),
        "forecast_hours": 24,
        "weather_hourly_units": df.attrs.get("weather_units", {}),
        "air_quality_hourly_units": df.attrs.get("air_quality_units", {}),
        "trail_metadata_excerpt": {
            k: metadata.get(k)
            for k in ("latitude", "longitude", "name", "title", "id", "url")
            if k in metadata
        },
        "hourly_forecast": records,
        "integration": {
            "weather_api": omf.FORECAST_URL,
            "air_quality_api": omf.AIR_QUALITY_URL,
            "user_agent_note": USER_AGENT_NOTE,
        },
    }


def write_forecast_json(
    bucket: storage.Bucket,
    dest_prefix: str,
    trail_slug: str,
    payload: dict[str, Any],
) -> str:
    if not dest_prefix.endswith("/"):
        dest_prefix = dest_prefix + "/"
    dest_name = f"{dest_prefix}{trail_slug}/open_meteo_24h.json"
    blob = bucket.blob(dest_name)
    blob.upload_from_string(
        json.dumps(payload, indent=2),
        content_type="application/json",
    )
    return f"gs://{bucket.name}/{dest_name}"


def process_trail(
    trail_slug: str,
    blob_name: str,
    src_bucket: storage.Bucket,
    dst_bucket: storage.Bucket,
    dest_prefix: str,
    source_bucket_name: str,
) -> str:
    """Load metadata, fetch Open-Meteo, write JSON. Returns gs:// URI. Raises on failure."""
    source_uri = f"gs://{source_bucket_name}/{blob_name}"
    try:
        metadata = load_metadata_json(src_bucket, blob_name)
        lat, lon = coords_from_metadata(metadata)
    except (KeyError, TypeError, ValueError) as e:
        raise RuntimeError(f"bad metadata / coordinates ({e})") from e

    try:
        df = omf.fetch_open_meteo(lat, lon)
    except Exception as e:
        raise RuntimeError(f"Open-Meteo failed: {e}") from e

    payload = forecast_payload(
        trail_slug=trail_slug,
        source_gs_uri=source_uri,
        metadata=metadata,
        df=df,
    )
    try:
        return write_forecast_json(dst_bucket, dest_prefix, trail_slug, payload)
    except Exception as e:
        raise RuntimeError(f"GCS write failed: {e}") from e


def main() -> int:
    args = parse_args()
    scrape_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    root = args.dest_root.strip("/")
    dest_prefix = f"{root}/{scrape_date}/"

    try:
        client = gcs_client(args.credentials)
    except Exception as e:
        print(f"error: could not create GCS client: {e}", file=sys.stderr)
        return 1

    src_bucket = client.bucket(args.source_bucket)
    dst_bucket = client.bucket(args.dest_bucket)

    try:
        pairs = list_metadata_blob_names(src_bucket, max_trails=args.max_trails)
    except Exception as e:
        print(f"error: listing metadata blobs: {e}", file=sys.stderr)
        return 1

    if not pairs:
        print("No metadata.json objects found under prefix.", file=sys.stderr)
        return 1

    cap = "all" if args.max_trails is None else args.max_trails
    workers = max(1, args.workers)
    print(
        f"Found {len(pairs)} trail(s) to process (cap={cap}, workers={workers}); "
        f"dest gs://{args.dest_bucket}/{dest_prefix}",
    )

    errors = 0
    n_ok = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                process_trail,
                trail_slug,
                blob_name,
                src_bucket,
                dst_bucket,
                dest_prefix,
                args.source_bucket,
            ): trail_slug
            for trail_slug, blob_name in pairs
        }
        for future in as_completed(futures):
            trail_slug = futures[future]
            try:
                out_uri = future.result()
            except Exception as e:
                errors += 1
                print(f"skip {trail_slug}: {e}", file=sys.stderr)
            else:
                n_ok += 1
                print(f"wrote {out_uri}")

    print(
        f"Summary: {n_ok} wrote successfully, {errors} skipped or failed "
        f"(out of {len(pairs)} trails listed).",
    )
    if errors:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
