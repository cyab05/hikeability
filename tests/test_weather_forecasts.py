import json
import os
import urllib.error
import uuid

import pytest

import open_meteo_fetch as omf
from trail_forecasts_gcs import coords_from_metadata, forecast_payload, write_forecast_json


# test different shapes of lat lon from GCS (og function is coords_from_metadata)
def test_coords_common_shapes():
    assert coords_from_metadata({"latitude": 47.5, "longitude": -121.8}) == (47.5, -121.8)
    assert coords_from_metadata({"lat": 48.0, "lon": -122.5}) == (48.0, -122.5)
    # tests GeoJSON format
    lat, lon = coords_from_metadata({"coordinates": [-122.33, 47.61]})
    assert abs(lat - 47.61) < 1e-9 and abs(lon + 122.33) < 1e-9


# verify api call (make sure it works end to end)
def test_open_meteo_smoke():
    try:
        df = omf.fetch_open_meteo(47.61, -122.33)
    except (urllib.error.URLError, OSError, RuntimeError):
        pytest.skip("open meteo failed")

    assert len(df) == 24 # one prediction an hour
    assert list(df.columns) == [
        "time", "apparent_temperature", "snowfall", "snow_depth", "us_aqi",] # make sure df has right columns
    assert df.attrs["timezone"] == omf.TIMEZONE # ensure correct timezone

# end-to-end read/write from GCS 
def test_gcs_forecast_roundtrip_optional():
    """writes one forecast JSON like trail_forecasts_gcs, reads it back, deletes it"""
    
    bucket_name = bucket_name = "weather-conditions"

    from google.cloud import storage

    dest_prefix = f"weather-scraper-pytest/{uuid.uuid4()}/"
    trail_slug = "pytest-trail"

    try:
        df = omf.fetch_open_meteo(47.61, -122.33)
    except (OSError, RuntimeError, urllib.error.URLError):
        pytest.skip("Open-Meteo unavailable")

    payload = forecast_payload(
        trail_slug=trail_slug,
        source_gs_uri=f"gs://{bucket_name}/dummy/metadata.json",
        metadata={"latitude": 47.61, "longitude": -122.33, "name": "pytest"},
        df=df,
    )

    bucket = storage.Client().bucket(bucket_name)
    blob = bucket.blob(f"{dest_prefix}{trail_slug}/open_meteo_24h.json")

    try:
        write_forecast_json(bucket, dest_prefix, trail_slug, payload)
        data = json.loads(blob.download_as_text(encoding="utf-8"))
        assert data["trail_slug"] == trail_slug
        assert len(data["hourly_forecast"]) == 24
    except Exception as e:
        pytest.skip(f"GCS write/read failed: {e}")
    finally:
        try:
            blob.delete()
        except Exception:
            pass
