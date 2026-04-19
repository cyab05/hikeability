"""
Weather data loading from GCS and formatting for LLM prompts.
"""
from __future__ import annotations

import json

from google.cloud import storage

from .config import GCS_BUCKET_WEATHER, GCS_WEATHER_PREFIX


def load_weather_for_hikes(
    client: storage.Client,
    hike_ids: list[str],
    weather_date: str | None = None,
) -> dict[str, dict]:
    """
    Load weather data from GCS for a list of hike IDs.

    Looks for the most recent date folder if weather_date is None.
    Path pattern: weather-scraped/{date}/output/hikes/{parent}/{hike_id}/weather.json

    Returns a dict mapping hike_id -> weather JSON.
    """
    bucket = client.bucket(GCS_BUCKET_WEATHER)

    # Find the most recent date folder if not specified
    if not weather_date:
        blobs = bucket.list_blobs(prefix=f"{GCS_WEATHER_PREFIX}/", delimiter="/")
        list(blobs)  # Force iteration to populate prefixes
        date_prefixes = sorted(blobs.prefixes, reverse=True)
        if not date_prefixes:
            print("No weather data found in GCS.")
            return {}
        weather_date = date_prefixes[0].rstrip("/").split("/")[-1]

    print(f"Loading weather data for date: {weather_date}")

    weather_map = {}
    prefix = f"{GCS_WEATHER_PREFIX}/{weather_date}/output/hikes"
    weather_blobs = bucket.list_blobs(prefix=prefix + "/")

    hike_id_set = set(hike_ids)
    for blob in weather_blobs:
        if not blob.name.endswith("/weather.json"):
            continue
        parts = blob.name.removeprefix(prefix + "/").split("/")
        if len(parts) == 3:
            hike_id = parts[1]
        elif len(parts) == 2:
            hike_id = parts[0]
        else:
            continue

        if hike_id in hike_id_set:
            weather_map[hike_id] = json.loads(blob.download_as_text())

    print(f"Loaded weather for {len(weather_map)}/{len(hike_ids)} hikes")
    return weather_map


def format_weather_context(weather: dict) -> str:
    """Format weather JSON into a concise text block for the LLM prompt."""
    current = weather.get("current_period", {})
    grid = weather.get("grid_summary", {})

    parts = ["Current weather at trailhead:"]

    if current.get("temperature"):
        parts.append(f"  Temperature: {current['temperature']}\u00b0{current.get('temperature_unit', 'F')}")
    if current.get("wind_speed"):
        parts.append(f"  Wind: {current['wind_speed']} {current.get('wind_direction', '')}")
    if current.get("short_forecast"):
        parts.append(f"  Forecast: {current['short_forecast']}")
    if current.get("detailed_forecast"):
        parts.append(f"  Details: {current['detailed_forecast']}")

    precip = current.get("probability_of_precipitation", {})
    if precip.get("value") is not None:
        parts.append(f"  Precipitation chance: {precip['value']}%")

    if grid.get("snow_expected"):
        snow_sum = grid.get("snowfall_amount", {}).get("sum", 0)
        parts.append(f"  Snow expected: Yes ({snow_sum}mm in next {weather.get('next_hours', 24)}h)")
    elif grid.get("snow_expected") is False:
        parts.append("  Snow expected: No")

    fetched = weather.get("fetched_at_utc", "")
    if fetched:
        parts.append(f"  Weather as of: {fetched}")

    return "\n".join(parts)
