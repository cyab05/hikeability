import json
import pytest
import sys
sys.path.insert(0, ".")
from classification.gcs_utils import parse_report_date
from classification.weather import format_weather_context
from classification.classifier import (
    build_input_text,
    build_weather_only_text,
    format_examples_block,
)
from classification.run_classification import (
    load_progress,
    save_progress,
)


def test_build_input_text_fields():
    report = {
        "trail_conditions": "Muddy",
        "road_conditions": "Clear",
        "snow": "None",
        "report_text": "Great hike overall.",
    }
    result = build_input_text(report)
    assert "Trail conditions: Muddy" in result
    assert "Road conditions: Clear" in result
    assert "Snow: None" in result
    assert "Great hike overall." in result

def test_build_input_text_with_weather():
    report = {"report_text": "Good conditions."}
    weather = {
        "hourly_forecast": [
            {
                "time": "2026-04-22T13:00:00.000",
                "apparent_temperature": 55.0,
                "snowfall": 0.0,
                "snow_depth": 0.0,
                "us_aqi": 30,
            }
        ],
        "daily_summary": {
            "weather_code": 0,  # 0 = clear sky per WMO
            "wind_gusts_10m_max": 10.0,
            "precipitation_probability_max": 5,
        },
    }
    result = build_input_text(report, weather=weather)
    assert "Good conditions." in result
    assert "clear sky" in result  # mapped from weather_code=0
    
# ── build_weather_only_text ──────────────────────────────────────────────────

def test_build_weather_only_text_stale_date():
    hike = {
        "hike_id": "h1",
        "hike_name": "Blue Lake",
        "hike_region": "North Cascades",
        "most_recent_report_date": "2024-01-01",
    }
    weather = {"current_period": {}, "grid_summary": {}}
    result = build_weather_only_text(hike, weather)
    assert "2024-01-01" in result
    assert "Blue Lake" in result

def test_build_weather_only_text_no_reports():
    hike = {
        "hike_id": "h2",
        "hike_name": "Mystery Peak",
        "hike_region": "Olympics",
        "most_recent_report_date": None,
    }
    weather = {"current_period": {}, "grid_summary": {}}
    result = build_weather_only_text(hike, weather)
    assert "No trip reports have ever been submitted" in result