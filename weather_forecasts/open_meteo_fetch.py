
"""
This file grabs the current conditions and hourly forecast for next 24 hours from open-meteo.

Weather variables
https://open-meteo.com/en/docs

Air quality vars
https://open-meteo.com/en/docs/air-quality-api
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from urllib.parse import urlsplit

import pandas as pd
import urllib3

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

WEATHER_VARS = "apparent_temperature,snowfall,snow_depth"
DAILY_VARS = "precipitation_probability_max,wind_gusts_10m_max,weather_code"
AIR_VARS = "us_aqi"

TIMEZONE = "America/Los_Angeles"

# open-meteo's allows around 600 requests/min (~10/sec) per endpoint
# cap slightly below for headroom
DEFAULT_MAX_REQ_PER_SEC = 8.0


class _RateLimiter:
    """Thread-safe token-bucket style limiter. Blocks until a slot is available."""

    def __init__(self, max_per_sec: float) -> None:
        self._min_interval = 1.0 / max_per_sec if max_per_sec > 0 else 0.0
        self._lock = threading.Lock()
        self._next_at = 0.0

    def acquire(self) -> None:
        if self._min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            slot = max(self._next_at, now)
            self._next_at = slot + self._min_interval
        delay = slot - time.monotonic()
        if delay > 0:
            time.sleep(delay)


# one limiter per host, since weather and air-quality are separate endpoints
# with independent quotas on open-meteo's side.
_LIMITERS: dict[str, _RateLimiter] = {
    urlsplit(FORECAST_URL).netloc: _RateLimiter(DEFAULT_MAX_REQ_PER_SEC),
    urlsplit(AIR_QUALITY_URL).netloc: _RateLimiter(DEFAULT_MAX_REQ_PER_SEC),
}


def set_rate_limit(max_per_sec: float) -> None:
    """Override the per-endpoint requests/sec cap (affects future calls)."""
    for host in _LIMITERS:
        _LIMITERS[host] = _RateLimiter(max_per_sec)


# shared connection pool. thread-safe, reuses TCP+TLS across calls so we don't
# pay the ~100ms handshake cost per request. longer backoff handles stray 429s.
_HTTP = urllib3.PoolManager(
    num_pools=4,
    maxsize=50,
    retries=urllib3.Retry(
        total=4,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        respect_retry_after_header=True,
    ),
    headers={"User-Agent": "open-meteo-fetch/1.0"},
)


# builds the url, makes the request, and returns the json response
# only place that actually touches the network
def _get_json(url: str, params: dict[str, Any]) -> dict[str, Any]:
    limiter = _LIMITERS.get(urlsplit(url).netloc)
    if limiter is not None:
        limiter.acquire()
    resp = _HTTP.request("GET", url, fields=params, timeout=60.0)
    if resp.status >= 400:
        raise RuntimeError(f"HTTP {resp.status}: {resp.data[:500].decode(errors='replace')}")
    return json.loads(resp.data.decode())

# grabs the weather data (hourly for next 24h + today's daily summary variables)
def fetch_weather(latitude: float, longitude: float) -> dict[str, Any]:
    return _get_json(
        FORECAST_URL,
        {
            "latitude": latitude,
            "longitude": longitude,
            "timezone": TIMEZONE,
            "current": WEATHER_VARS,
            "hourly": WEATHER_VARS,
            "daily": DAILY_VARS,
            "temperature_unit": "fahrenheit",
            "forecast_hours": 24,
            "forecast_days": 1,
        },
    )

# grabs the air quality data
def fetch_air_quality(latitude: float, longitude: float) -> dict[str, Any]:
    return _get_json(
        AIR_QUALITY_URL,
        {
            "latitude": latitude,
            "longitude": longitude,
            "timezone": TIMEZONE,
            "current": AIR_VARS,
            "hourly": AIR_VARS,
            "forecast_hours": 24,
        },
    )


def fetch_open_meteo(
    latitude: float,
    longitude: float,
) -> pd.DataFrame:
    """
    Return a DataFrame of the 24-hour hourly forecast merged from the
    weather and air quality APIs. Timezone is always America/Los_Angeles.
    Fires fetch_weather and fetch_air_quality in parallel.

    Columns: time, apparent_temperature, snowfall, snow_depth, us_aqi.

    ``df.attrs['daily']`` holds today's single-day summary:
    precipitation_probability_max, wind_gusts_10m_max, weather_code.
    """
    def _run() -> tuple[dict[str, Any], dict[str, Any]]:
        with ThreadPoolExecutor(max_workers=2) as ex:
            f_w = ex.submit(fetch_weather, latitude, longitude)
            f_a = ex.submit(fetch_air_quality, latitude, longitude)
            try:
                weather = f_w.result()
                air = f_a.result()
            except urllib3.exceptions.HTTPError as e:
                raise RuntimeError(f"open-meteo network error: {e}") from e
        return weather, air

    weather, air = _run()

    # make a df from hourly weather data
    hourly_w = weather.get("hourly", {})
    hourly_a = air.get("hourly", {})

    df = pd.DataFrame({
        "time": pd.to_datetime(hourly_w.get("time", [])),
        "apparent_temperature": hourly_w.get("apparent_temperature"),
        "snowfall":             hourly_w.get("snowfall"),
        "snow_depth":           hourly_w.get("snow_depth"),
        "us_aqi":               hourly_a.get("us_aqi"),
    })

    # today's daily summary (one value per variable)
    daily = weather.get("daily", {}) or {}
    daily_summary: dict[str, Any] = {}
    for key in ("time", "precipitation_probability_max", "wind_gusts_10m_max", "weather_code"):
        values = daily.get(key)
        daily_summary[key] = values[0] if isinstance(values, list) and values else None

    # add units as DataFrame attrs for reference
    df.attrs["weather_units"]     = weather.get("hourly_units", {})
    df.attrs["air_quality_units"] = air.get("hourly_units", {})
    df.attrs["daily"]             = daily_summary
    df.attrs["daily_units"]       = weather.get("daily_units", {})
    df.attrs["timezone"]          = TIMEZONE

    return df


def main() -> None:
    # makes a cli parser
    p = argparse.ArgumentParser(
        description="Open-Meteo: apparent temp, snowfall, snow depth, air quality (24h forecast). Timezone: America/Los_Angeles.",
    )
    # adds the latitude and longitude arguments
    p.add_argument("latitude", type=float, help="WGS84 latitude")
    p.add_argument("longitude", type=float, help="WGS84 longitude")
    # reads the terminal input
    args = p.parse_args()

    try: # tries to fetch the data
        df = fetch_open_meteo(args.latitude, args.longitude)
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)

    print(df.to_string(index=False))
    print()
    print("Daily summary (today):")
    for k, v in df.attrs.get("daily", {}).items():
        unit = df.attrs.get("daily_units", {}).get(k, "")
        print(f"  {k}: {v} {unit}".rstrip())


if __name__ == "__main__":
    main()
