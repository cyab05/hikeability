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


# WMO weather code → plain English (subset relevant to hiking)
WMO_CODES = {
    0: "clear sky",
    1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "depositing rime fog",
    51: "light drizzle", 53: "moderate drizzle", 55: "dense drizzle",
    56: "light freezing drizzle", 57: "dense freezing drizzle",
    61: "light rain", 63: "moderate rain", 65: "heavy rain",
    66: "light freezing rain", 67: "heavy freezing rain",
    71: "light snow", 73: "moderate snow", 75: "heavy snow",
    77: "snow grains",
    80: "light rain showers", 81: "moderate rain showers", 82: "violent rain showers",
    85: "light snow showers", 86: "heavy snow showers",
    95: "thunderstorm", 96: "thunderstorm with light hail", 99: "thunderstorm with heavy hail",
}


def _aqi_category(aqi: float) -> str:
    if aqi <= 50:   return "Good"
    if aqi <= 100:  return "Moderate"
    if aqi <= 150:  return "Unhealthy for Sensitive Groups"
    if aqi <= 200:  return "Unhealthy"
    if aqi <= 300:  return "Very Unhealthy"
    return "Hazardous"


def format_weather_context(weather: dict) -> str:
    """Format Open-Meteo weather JSON into a concise text block for the LLM prompt."""
    hourly = weather.get("hourly_forecast") or []
    daily = weather.get("daily_summary") or {}

    parts = ["Current weather at trailhead:"]

    if hourly:
        current = hourly[0]

        # Current snapshot
        temp = current.get("apparent_temperature")
        if temp is not None:
            parts.append(f"  Feels-like temperature: {temp:.0f}\u00b0F")

        aqi = current.get("us_aqi")
        if aqi is not None:
            parts.append(f"  Air quality index: {aqi:.0f} ({_aqi_category(aqi)})")

        snow_depth_m = current.get("snow_depth")
        if snow_depth_m is not None and snow_depth_m > 0:
            snow_depth_in = snow_depth_m * 39.37
            parts.append(f"  Snow on ground: {snow_depth_in:.1f} inches")
        elif snow_depth_m == 0:
            parts.append("  Snow on ground: none")

        # 24h summary
        temps = [h["apparent_temperature"] for h in hourly if h.get("apparent_temperature") is not None]
        aqis = [h["us_aqi"] for h in hourly if h.get("us_aqi") is not None]
        snowfalls_cm = [h["snowfall"] for h in hourly if h.get("snowfall") is not None]

        if temps:
            parts.append(f"  Next 24h temp range: {min(temps):.0f}\u00b0F to {max(temps):.0f}\u00b0F")
        if aqis and max(aqis) > 100:
            parts.append(f"  Max AQI next 24h: {max(aqis):.0f} ({_aqi_category(max(aqis))})")
        if snowfalls_cm:
            total_snow_cm = sum(snowfalls_cm)
            if total_snow_cm > 0:
                total_snow_in = total_snow_cm / 2.54
                parts.append(f"  Snowfall expected next 24h: {total_snow_in:.1f} inches")
            else:
                parts.append("  Snowfall expected next 24h: none")

    if daily:
        code = daily.get("weather_code")
        if code is not None:
            desc = WMO_CODES.get(int(code), f"code {code}")
            parts.append(f"  Conditions today: {desc}")

        gusts_kmh = daily.get("wind_gusts_10m_max")
        if gusts_kmh is not None:
            gusts_mph = gusts_kmh * 0.6214
            parts.append(f"  Max wind gusts today: {gusts_mph:.0f} mph")

        precip_prob = daily.get("precipitation_probability_max")
        if precip_prob is not None:
            parts.append(f"  Chance of precipitation today: {precip_prob}%")

    fetched = weather.get("fetched_at_utc", "")
    if fetched:
        parts.append(f"  Weather as of: {fetched}")

    return "\n".join(parts)
