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


def _format_pacific(iso_utc: str | None) -> str | None:
    """Convert an ISO-8601 UTC timestamp ('2026-05-04T09:19:06Z') to '05-04-2026 2:19am PDT'."""
    if not iso_utc:
        return None
    try:
        dt = datetime.fromisoformat(iso_utc.replace("Z", "+00:00"))
        local = dt.astimezone(_PACIFIC_TZ)
        # %-I = non-padded hour (2 not 02). The .lower() on AM/PM matches the requested style.
        return local.strftime("%m-%d-%Y %-I:%M%p %Z").replace("AM", "am").replace("PM", "pm")
    except Exception:
        return iso_utc  # fall back to raw if parsing fails

# Validators for scraped stat fields. The WTA scraper sometimes grabs paragraph
# text instead of the structured stat — we drop anything that doesn't look like
# a clean numeric value at read time.
_VALID_FEET     = re.compile(r"^[\d,]+(\s*(feet|ft))?\.?$", re.IGNORECASE)
_VALID_DISTANCE = re.compile(r"^[\d.,]+\s*miles?(\s*,?\s*(roundtrip|one-way|of trails))?\.?$", re.IGNORECASE)


def _clean_stat(value, pattern: re.Pattern) -> str | None:
    """Return value if it matches the pattern, else None. Treats blank/None as None."""
    if value in (None, ""):
        return None
    s = str(value).strip()
    return s if pattern.match(s) else None

# Bucket / prefix constants (mirrors classification/config.py)
_BUCKET_OUTPUT  = "hikes-model-output"
_PRED_PREFIX    = "predictions"
_BUCKET_WEATHER = "weather-conditions"
_WEATHER_PREFIX = "weather-scraped"
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
    Otherwise all date folders are merged in chronological order so each hike
    ends up with its most recent prediction (daily runs are incremental).
    """
    bucket = client.bucket(_BUCKET_OUTPUT)

    # Only merge runs from when the classification pipeline was finalized (2026-05-01).
    # Earlier daily runs had incomplete data and should be ignored.
    MIN_DATE = "2026-05-01"

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
            return prefix, json.loads(blob.download_as_text())
        except Exception:
            return prefix, None

    # Parallel downloads, then merge in date-prefix order so newer overwrites
    seen: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=32) as ex:
        results = list(ex.map(_download, all_blobs))

    # Sort by prefix (= date) ascending so newer dates overwrite older
    results.sort(key=lambda r: r[0])
    for _prefix, data in results:
        if isinstance(data, list):
            for p in data:
                if p.get("hike_id"):
                    seen[p["hike_id"]] = p

    predictions = list(seen.values())
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
    hike["weather"] = _fetch_weather_summary(client, hike_id)
    hike["label_color"] = LABEL_COLORS.get(hike.get("predicted_label", ""), "#888888")
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
                "is_closed":      any(n.get("severity") == "red" for n in notes),
                "warning_short":  notes[0]["message"][:80] if notes else None,
            },
        })
    return {"type": "FeatureCollection", "features": features}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _to_float(val) -> float | None:
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None

_WMO_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    61: "Light rain", 63: "Moderate rain", 65: "Heavy rain",
    71: "Light snow", 73: "Moderate snow", 75: "Heavy snow",
    80: "Rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
    85: "Snow showers", 95: "Thunderstorm",
}


def _fetch_weather_summary(client: storage.Client, hike_id: str) -> dict:
    """Fetch and parse weather JSON for a hike. Returns empty dict on miss."""
    bucket = client.bucket(_BUCKET_WEATHER)
    blobs = bucket.list_blobs(prefix=f"{_WEATHER_PREFIX}/", delimiter="/")
    list(blobs)
    date_prefixes = sorted(blobs.prefixes, reverse=True)
    if not date_prefixes:
        return {}

    weather_date = date_prefixes[0].rstrip("/").split("/")[-1]
    blob = bucket.blob(f"{_WEATHER_PREFIX}/{weather_date}/{hike_id}/open_meteo_24h.json")
    if not blob.exists():
        return {}

    raw = json.loads(blob.download_as_text())
    return _parse_weather(raw)


def _parse_weather(raw: dict) -> dict:
    out = {
        "temp_f": None,
        "aqi": None,
        "snow_depth_in": None,
        "wind_gusts_mph": None,
        "precip_chance_pct": None,
        "description": None,
        "fetched_at": _format_pacific(raw.get("fetched_at_utc")),
    }
    hourly = raw.get("hourly_forecast") or []
    if hourly:
        h = hourly[0]
        out["temp_f"] = _to_float(h.get("apparent_temperature"))
        out["aqi"] = _to_float(h.get("us_aqi"))
        snow_m = _to_float(h.get("snow_depth"))
        if snow_m is not None:
            out["snow_depth_in"] = round(snow_m * 39.37, 1)

    daily = raw.get("daily_summary") or {}
    gusts_kmh = _to_float(daily.get("wind_gusts_10m_max"))
    if gusts_kmh is not None:
        out["wind_gusts_mph"] = round(gusts_kmh * 0.6214, 1)
    out["precip_chance_pct"] = _to_float(daily.get("precipitation_probability_max"))
    code = daily.get("weather_code")
    if code is not None:
        out["description"] = _WMO_CODES.get(int(float(code)), f"Code {code}")

    return out
