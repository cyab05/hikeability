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
    format_trail_alerts,
    is_closure_alert,
)
from classification.run_classification import (
    load_progress,
    save_progress,
)


def test_build_input_text_fields():
    trail = {
        "hike_name": "Test Trail",
        "reports": [{
            "trail_conditions": "Muddy",
            "road_conditions": "Clear",
            "snow": "None",
            "report_text": "Great hike overall.",
        }],
    }
    result = build_input_text(trail)
    assert "Trail conditions: Muddy" in result
    assert "Road conditions: Clear" in result
    assert "Snow: None" in result
    assert "Great hike overall." in result

def test_build_input_text_with_weather():
    trail = {"reports": [{"report_text": "Good conditions."}]}
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
    result = build_input_text(trail, weather=weather)
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


# ── is_closure_alert ────────────────────────────────────────────────────────

@pytest.mark.parametrize("message", [
    "The road to Mount Pilchuck is closed at the Heather Lake trailhead for the winter.",
    "The road to this trailhead (NF 37) is washed out 9.6 miles before the trailhead.",
    "Cascade River Road is officially closed to vehicle traffic at milepost 21.5.",
    "Trailhead inaccessible due to Beckler River Road being closed.",
    "Trail is impassable due to flood damage.",
    "Bridge blocked; do not attempt the crossing.",
    "DO NOT GO past the river crossing — washout active.",
])
def test_is_closure_alert_detects_real_closures(message):
    assert is_closure_alert({"severity": "red", "message": message}) is True


def test_is_closure_alert_rejects_seasonal_advisory():
    # WTA's granite-mountain alert: red severity, but year-round seasonal advisory
    note = {
        "severity": "red",
        "message": "In winter the trail up Granite Mountain crosses a large avalanche chute.",
    }
    assert is_closure_alert(note) is False


def test_is_closure_alert_only_fires_on_red():
    # Closure verbs in non-red severities should not trip the closure flag
    for sev in ("orange", "yellow", "blue", "green"):
        assert is_closure_alert({
            "severity": sev,
            "message": "Road closed; trailhead inaccessible.",
        }) is False


def test_is_closure_alert_handles_missing_fields():
    assert is_closure_alert({}) is False
    assert is_closure_alert({"severity": "red"}) is False
    assert is_closure_alert({"severity": "red", "message": ""}) is False
    assert is_closure_alert({"severity": "RED", "message": "Trail closed."}) is True


# ── format_trail_alerts ─────────────────────────────────────────────────────

def test_format_trail_alerts_empty():
    assert format_trail_alerts(None) is None
    assert format_trail_alerts([]) is None


def test_format_trail_alerts_red_closure_vs_advisory():
    # Same severity ("red"), different tag depending on closure verbs
    block = format_trail_alerts([
        {"severity": "red", "message": "Road is closed for the season."},
        {"severity": "red", "message": "In winter the trail crosses an avalanche chute."},
    ])
    assert "[CLOSURE / DO NOT GO]" in block
    assert "[SERIOUS WARNING]" in block
    # Closure tag should sit on the closure message, not the advisory
    closure_line = [l for l in block.splitlines() if "CLOSURE" in l][0]
    assert "Road is closed" in closure_line
    advisory_line = [l for l in block.splitlines() if "SERIOUS WARNING" in l][0]
    assert "avalanche chute" in advisory_line


def test_format_trail_alerts_severity_mapping():
    block = format_trail_alerts([
        {"severity": "orange", "message": "Permit required."},
        {"severity": "yellow", "message": "Heads up."},
        {"severity": "blue",   "message": "Notice at trailhead."},
        {"severity": "green",  "message": "Volunteer day."},
    ])
    assert "[SERIOUS WARNING]" in block  # orange
    assert "[CAUTION]" in block          # yellow
    # blue and green both map to INFORMATIONAL
    assert block.count("[INFORMATIONAL]") == 2


def test_format_trail_alerts_skips_empty_messages():
    block = format_trail_alerts([
        {"severity": "red", "message": ""},
        {"severity": "red", "message": "Real closure."},
    ])
    # Only the message-bearing alert should render as a bullet
    assert block.count("\n  [") == 1


# ── alert plumbing into prompts ─────────────────────────────────────────────

def test_build_input_text_includes_closure_alert():
    trail = {
        "hike_name": "Mount Pilchuck",
        "closure_warning": [
            {"severity": "red", "message": "The road to Mount Pilchuck is closed for the winter."},
        ],
        "reports": [{"report_text": "Snow free at lower elevations."}],
    }
    result = build_input_text(trail)
    assert "Official trail alerts" in result
    assert "[CLOSURE / DO NOT GO]" in result
    assert "closed for the winter" in result


def test_build_input_text_no_alerts_no_section():
    trail = {
        "hike_name": "Mount Si",
        "closure_warning": [],
        "reports": [{"report_text": "All good."}],
    }
    result = build_input_text(trail)
    assert "Official trail alerts" not in result


def test_build_weather_only_text_includes_alert():
    hike = {
        "hike_id": "skyline-divide",
        "hike_name": "Skyline Divide",
        "closure_warning": [
            {"severity": "red", "message": "NF 37 is washed out 9.6 miles before the trailhead."},
        ],
        "most_recent_report_date": "2025-07-03",
    }
    weather = {"hourly_forecast": [], "daily_summary": {}}
    result = build_weather_only_text(hike, weather)
    assert "[CLOSURE / DO NOT GO]" in result
    assert "washed out" in result