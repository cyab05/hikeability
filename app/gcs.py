"""
GCS data access layer for the Flask web app.
Reads predictions and weather data; builds GeoJSON for Mapbox.
"""
from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from zoneinfo import ZoneInfo

from google.cloud import storage

_PACIFIC_TZ = ZoneInfo("America/Los_Angeles")


def _format_pacific(dt: datetime | None) -> str | None:
    """Convert a UTC datetime to '05-17-2026 9:04am PDT'. Returns None on bad input."""
    if dt is None:
        return None
    try:
        local = dt.astimezone(_PACIFIC_TZ)
        # %-I = non-padded hour (9 not 09). The .lower() on AM/PM matches the requested style.
        return local.strftime("%m-%d-%Y %-I:%M%p %Z").replace("AM", "am").replace("PM", "pm")
    except Exception:
        return None

# Validators for scraped stat fields. The WTA scraper sometimes grabs paragraph
# text instead of the structured stat — we drop anything that doesn't look like
# a clean numeric value at read time.
_VALID_FEET     = re.compile(r"^[\d,]+(\s*(feet|ft))?\.?$", re.IGNORECASE)
_VALID_DISTANCE = re.compile(r"^[\d.,]+\s*miles?(\s*,?\s*(roundtrip|one-way|of trails))?\.?$", re.IGNORECASE)

# WTA reuses red severity for both real closures ("road closed", "trailhead
# inaccessible") and year-round safety advisories ("in winter the trail
# crosses an avalanche chute"). We only treat the first kind as a forced
# unhikeable. Same regex is mirrored in classification.classifier.is_closure_alert
# — kept inline here so the Vercel deploy doesn't need the classification package.
_CLOSURE_TERMS = re.compile(
    r"\b(closed|closure|inaccessible|washed[\s-]*out|impassable|blocked|"
    r"do\s+not\s+(go|hike|enter|attempt))\b",
    re.IGNORECASE,
)


def _clean_stat(value, pattern: re.Pattern) -> str | None:
    """Return value if it matches the pattern, else None. Treats blank/None as None."""
    if value in (None, ""):
        return None
    s = str(value).strip()
    return s if pattern.match(s) else None

# Bucket / prefix constants (mirrors classification/config.py)
_BUCKET_OUTPUT  = "hikes-model-output"
_PRED_PREFIX    = "predictions"
_BUCKET_RAW     = "wta-hikes"
_RAW_PREFIX     = "output/hikes"

LABEL_COLORS = {
    "hikeable":   "#54B393",
    "modest":     "#F7A745",
    "unhikeable": "#DC4848",
}


def get_client() -> storage.Client:
    creds_json = os.environ.get("GCS_CREDENTIALS_JSON")
    if creds_json:
        from google.oauth2 import service_account
        # raw_decode parses the first complete JSON object and ignores anything after,
        # so it survives Vercel-textarea quirks where the value gets duplicated/appended.
        info, _ = json.JSONDecoder().raw_decode(creds_json.strip())
        return storage.Client(credentials=service_account.Credentials.from_service_account_info(info))
    return storage.Client()  # falls back to file-based auth locally


def load_latest_predictions(client: storage.Client, date: str | None = None) -> list[dict]:
    """
    Load predictions from GCS.
    If `date` is given, only that date folder is read.
    Otherwise prefers the consolidated `predictions/latest.json` snapshot
    written by the classifier; falls back to merging per-date folders if the
    snapshot doesn't exist yet (first deploy / older buckets).
    """
    bucket = client.bucket(_BUCKET_OUTPUT)

    # Only merge runs from when the classification pipeline was finalized (2026-05-01).
    # Earlier daily runs had incomplete data and should be ignored.
    MIN_DATE = "2026-05-01"

    if date is None:
        # Fast path: single-blob snapshot the classifier rewrites after each run.
        snapshot = bucket.blob(f"{_PRED_PREFIX}/latest.json")
        try:
            text = snapshot.download_as_text()
            snapshot.reload()  # populate .updated for prediction_written_at
            predictions = json.loads(text)
            stamp = snapshot.updated
            for p in predictions:
                p["prediction_written_at"] = stamp
            return _finalize_predictions(client, predictions)
        except Exception:
            pass  # fall through to per-date merge

    if date:
        date_prefixes = [f"{_PRED_PREFIX}/{date}/"]
    else:
        blobs = bucket.list_blobs(prefix=f"{_PRED_PREFIX}/", delimiter="/")
        list(blobs)  # force iteration to populate prefixes
        all_prefixes = sorted(blobs.prefixes)  # oldest → newest
        date_prefixes = [p for p in all_prefixes if p.rstrip("/").split("/")[-1] >= MIN_DATE]
        if not date_prefixes:
            return []

    # Collect all blobs first (cheap), then download them in parallel
    all_blobs: list[tuple[str, storage.Blob]] = []
    for prefix in date_prefixes:
        for blob in bucket.list_blobs(prefix=prefix):
            if blob.name.endswith(".json"):
                all_blobs.append((prefix, blob))

    def _download(item):
        prefix, blob = item
        try:
            text = blob.download_as_text()
        except Exception:
            return prefix, None, None
        # blob.updated is populated when the blob was fetched via list_blobs above
        return prefix, json.loads(text), blob.updated

    # Parallel downloads, then merge in date-prefix order so newer overwrites
    seen: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=32) as ex:
        results = list(ex.map(_download, all_blobs))

    # Sort by prefix (= date) ascending so newer dates overwrite older
    results.sort(key=lambda r: r[0])
    for prefix, data, updated in results:
        if isinstance(data, list):
            for p in data:
                if p.get("hike_id"):
                    # Exact write time of the prediction blob — used to label the
                    # weather snapshot on the hike page with a precise local timestamp.
                    p["prediction_written_at"] = updated
                    seen[p["hike_id"]] = p

    return _finalize_predictions(client, list(seen.values()))


def _finalize_predictions(client: storage.Client, predictions: list[dict]) -> list[dict]:
    """Enrich, sanitize, and apply the closure-override pass shared by both load paths."""
    _enrich_coordinates(client, predictions)

    # Sanitize scraped stat fields — drop anything that doesn't look like a real
    # numeric value (the WTA scraper occasionally grabs paragraph prose instead).
    for p in predictions:
        p["elevation_gain"] = _clean_stat(p.get("elevation_gain"), _VALID_FEET)
        p["highest_point"]  = _clean_stat(p.get("highest_point"),  _VALID_FEET)
        p["distance"]       = _clean_stat(p.get("distance"),       _VALID_DISTANCE)

    # Hard-override the model's label when WTA has flagged a serious closure.
    # A red wta-note means the trail is inaccessible regardless of what recent
    # trip reports or weather suggest.
    for p in predictions:
        notes = p.get("closure_warning") or []
        if any(n.get("severity") == "red" for n in notes):
            p["predicted_label"] = "unhikeable"

    return predictions


def _enrich_coordinates(client: storage.Client, predictions: list[dict]) -> None:
    """Backfill missing fields (lat/lng, distance, rating, etc.) from raw metadata.json. Parallelized."""
    bucket = client.bucket(_BUCKET_RAW)

    def _has_everything(p: dict) -> bool:
        return (p.get("latitude") is not None and p.get("longitude") is not None
                and p.get("distance") and p.get("rating") and p.get("url")
                and p.get("elevation_gain") and p.get("highest_point") and p.get("hike_name")
                and p.get("image_url") and p.get("difficulty")
                # parking_pass and closure_warning can legitimately be falsy
                # (None / []) after enrichment, so check key presence instead
                # of truthiness to avoid re-fetching on every app boot.
                and "parking_pass" in p and "closure_warning" in p)

    needs_fetch = [p for p in predictions if not _has_everything(p)]

    def _fetch_meta(p: dict):
        blob = bucket.blob(f"{_RAW_PREFIX}/{p['hike_id']}/metadata.json")
        try:
            text = blob.download_as_text()
        except Exception:
            return p, None
        try:
            return p, json.loads(text)
        except Exception:
            return p, None

    with ThreadPoolExecutor(max_workers=64) as ex:
        for p, meta in ex.map(_fetch_meta, needs_fetch):
            if not meta:
                continue
            if p.get("latitude") is None:
                p["latitude"] = _to_float(meta.get("latitude"))
            if p.get("longitude") is None:
                p["longitude"] = _to_float(meta.get("longitude"))
            if not p.get("hike_name"):
                p["hike_name"] = meta.get("name", p["hike_id"])
            if not p.get("url"):
                p["url"] = meta.get("url")
            if not p.get("elevation_gain"):
                p["elevation_gain"] = meta.get("elevation_gain")
            if not p.get("highest_point"):
                p["highest_point"] = meta.get("highest_point")
            if not p.get("distance"):
                p["distance"] = meta.get("distance")
            if not p.get("rating"):
                p["rating"] = meta.get("rating")
            if not p.get("image_url"):
                p["image_url"] = meta.get("image_url")
            if not p.get("difficulty"):
                p["difficulty"] = meta.get("difficulty")
            if p.get("parking_pass") is None:
                p["parking_pass"] = meta.get("parking_pass")
            if p.get("closure_warning") is None:
                p["closure_warning"] = meta.get("closure_warning") or []


def get_hike(hike_id: str, all_predictions: list[dict], client: storage.Client) -> dict | None:
    """
    Return a single hike's prediction dict enriched with a parsed weather summary.
    Returns None if the hike_id is not found.
    """
    hike = next((p for p in all_predictions if p["hike_id"] == hike_id), None)
    if not hike:
        return None

    hike = dict(hike)  # don't mutate the cache
    # Surface the weather snapshot bundled into the prediction blob at
    # classification time, so the displayed values always match the
    # label_explanation. Live re-fetching from GCS drifts as soon as the
    # weather scraper writes a newer file between classification runs.
    desc = hike.get("weather_description")
    hike["weather"] = {
        "temp_f":            hike.get("current_temp_f"),
        "aqi":               hike.get("current_aqi"),
        "snow_depth_in":     hike.get("current_snow_depth_in"),
        "wind_gusts_mph":    hike.get("max_wind_gusts_mph"),
        "precip_chance_pct": hike.get("precip_chance_pct"),
        "description":       desc.capitalize() if desc else None,
        # Exact moment the prediction blob was written, so the timestamp lines
        # up with the weather snapshot the LLM saw — not whatever the live
        # scraper last wrote.
        "fetched_at":        _format_pacific(hike.get("prediction_written_at")),
    }
    hike["label_color"] = LABEL_COLORS.get(hike.get("predicted_label", ""), "#888888")

    # Flatten the most recent trip report's road/snow/bugs fields onto the hike
    # dict so the template (hike.html) can reference hike.road_conditions etc.
    # without iterating into reports[]. Reports are ordered most-recent first
    # by the scraper.
    reports = hike.get("reports") or []
    if reports:
        latest = reports[0] or {}
        for field in ("road_conditions", "snow", "bugs"):
            if not hike.get(field) and latest.get(field):
                hike[field] = latest[field]

    return hike


def build_geojson(predictions: list[dict]) -> dict:
    """Convert predictions list into a GeoJSON FeatureCollection for Mapbox."""
    features = []
    for p in predictions:
        lat = p.get("latitude")
        lng = p.get("longitude")
        if lat is None or lng is None:
            continue
        label = p.get("predicted_label", "unknown")
        notes = p.get("closure_warning") or []
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lng, lat]},
            "properties": {
                "hike_id":        p["hike_id"],
                "name":           p.get("hike_name", p["hike_id"]),
                "label":          label,
                "color":          LABEL_COLORS.get(label, "#888888"),
                "explanation":    p.get("label_explanation", p.get("explanation", "")),
                "region":         p.get("hike_region", ""),
                "distance":       p.get("distance"),
                "elevation_gain": p.get("elevation_gain"),
                "classification_source": p.get("classification_source", ""),
                "image_url":      p.get("image_url"),
                "difficulty":     p.get("difficulty"),
                "parking_pass_name": (p.get("parking_pass") or {}).get("name"),
                # Flat flags for the hover popup; full closure_warning list is
                # available via /api/hike/<id>/json for side panel + detail page.
                # is_closed = ANY red severity (includes year-round safety advisories
                # like "in winter the trail crosses an avalanche chute").
                "is_closed":      any(n.get("severity") == "red" for n in notes),
                # is_real_closure = red severity WITH a closure verb in the message.
                # Mirrors the post-classification override and is what actually flips
                # a trail to unhikeable in the UI. Lets the stats dashboard separate
                # true closures from advisory-only red alerts.
                "is_real_closure": any(
                    (n.get("severity") or "").lower() == "red"
                    and _CLOSURE_TERMS.search(n.get("message") or "")
                    for n in notes
                ),
                # True for any actionable alert (red closure OR orange warning).
                # Drives the "Trails with alerts" stat card and /trails?has_alert=1.
                # Yellow/blue/green are excluded as routine info, not alerts.
                "has_alert":      any(
                    (n.get("severity") or "").lower() in ("red", "orange") for n in notes
                ),
                "warning_short":  notes[0]["message"][:80] if notes else None,
                # Full alert text(s) for the trails-table hover tooltip. Joins all
                # actionable alerts on the trail (red + orange) prefixed with severity.
                "warnings_text":  "\n".join(
                    f"[{(n.get('severity') or '').upper()}] {(n.get('message') or '').strip()}"
                    for n in notes
                    if (n.get("severity") or "").lower() in ("red", "orange")
                    and (n.get("message") or "").strip()
                ) or None,
            },
        })
    return {"type": "FeatureCollection", "features": features}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _to_float(val) -> float | None:
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None

