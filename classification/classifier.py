"""
Nemotron LLM client and classification logic.
Hardcoded to one-shot strategy (winner from eval).
"""
from __future__ import annotations

import csv
import json
import os
import re
import time

from dotenv import load_dotenv
from openai import OpenAI

from .config import (
    API_DELAY_SECONDS,
    CSV_LABEL_MAP,
    LABELS,
    MODEL_ID,
    NVIDIA_API_BASE,
    PER_LABEL,
    REPORT_MAX_AGE_DAYS,
    SYSTEM_PROMPT,
    TRAINING_CSV_PATH,
)
from .weather import format_weather_context

load_dotenv()


# ── NVIDIA client ────────────────────────────────────────────────────────────

def get_nvidia_client() -> OpenAI:
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise EnvironmentError("NVIDIA_API_KEY not set")
    return OpenAI(base_url=NVIDIA_API_BASE, api_key=api_key)


# ── Few-shot examples ────────────────────────────────────────────────────────

def load_few_shot_examples(
    csv_path: str = TRAINING_CSV_PATH,
    per_label: int = PER_LABEL,
) -> list[dict]:
    """
    Load labeled examples from the training CSV for one-shot prompting.
    Returns `per_label` examples for each of the 3 categories.
    """
    examples = {label: [] for label in LABELS}

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapped = CSV_LABEL_MAP.get(row["label"])
            if mapped and len(examples[mapped]) < per_label:
                examples[mapped].append({
                    "text": row["comment_text"],
                    "label": mapped,
                })
            if all(len(v) >= per_label for v in examples.values()):
                break

    flat = []
    for label_examples in examples.values():
        flat.extend(label_examples)
    return flat


EXAMPLE_EXPLANATIONS = {
    "hikeable":   "Sunny and dry. Trail is clear and safe for all hikers.",
    "modest":     "Partly cloudy. Patchy snow at higher elevations — microspikes recommended.",
    "unhikeable": "Heavy snow and closed road. Trail is buried and inaccessible.",
}


def format_examples_block(examples: list[dict]) -> str:
    """Format few-shot examples into a text block for the prompt."""
    lines = ["Here are labeled examples:\n"]
    for i, ex in enumerate(examples, 1):
        explanation = EXAMPLE_EXPLANATIONS.get(ex["label"], "")
        lines.append(f"Example {i}:")
        lines.append(f"Report: {ex['text'][:300]}")
        lines.append(f'Classification: {{"label": "{ex["label"]}", "explanation": "{explanation}"}}')
        lines.append("")
    return "\n".join(lines)


# ── Input text builders ──────────────────────────────────────────────────────

def _format_elevation(data: dict) -> str | None:
    """Format elevation info from report or hike metadata."""
    lines = []
    if data.get("elevation_gain"):
        lines.append(f"  Elevation gain: {data['elevation_gain']}")
    if data.get("highest_point"):
        lines.append(f"  Highest point: {data['highest_point']}")
    if lines:
        return "Trail elevation:\n" + "\n".join(lines)
    return None


def build_input_text(report: dict, weather: dict | None = None) -> str:
    """Combine structured fields, free text, elevation, and optional weather data."""
    parts = []
    elevation = _format_elevation(report)
    if elevation:
        parts.append(elevation)
    else:
        parts.append("Elevation data not available.")
    if report.get("trail_conditions"):
        parts.append(f"Trail conditions: {report['trail_conditions']}")
    if report.get("road_conditions"):
        parts.append(f"Road conditions: {report['road_conditions']}")
    if report.get("snow"):
        parts.append(f"Snow: {report['snow']}")
    if report.get("bugs"):
        parts.append(f"Bugs: {report['bugs']}")
    if report.get("report_text"):
        parts.append(report["report_text"])
    if weather:
        parts.append("")
        parts.append(format_weather_context(weather))
    return "\n".join(parts).strip()


def build_weather_only_text(hike: dict, weather: dict) -> str:
    """Build prompt text for hikes with no recent reports — weather only."""
    last_report = hike.get("most_recent_report_date")
    if last_report:
        staleness_note = f"Most recent trip report was on {last_report} (over 30 days ago)."
    else:
        staleness_note = "No trip reports have ever been submitted for this trail."

    parts = [
        f"Hike: {hike.get('hike_name', hike.get('hike_id', 'Unknown'))}",
        f"Region: {hike.get('hike_region', 'Unknown')}",
    ]
    elevation = _format_elevation(hike)
    if elevation:
        parts.append(elevation)
    else:
        parts.append("Elevation data is not available for this trail.")
    parts += [
        "",
        staleness_note,
        "Classify based on current weather conditions and trail elevation.",
        "",
        format_weather_context(weather),
    ]
    return "\n".join(parts)


# ── Single classification ────────────────────────────────────────────────────

def classify_one(
    text: str,
    client: OpenAI,
    examples: list[dict] | None = None,
    retries: int = 3,
) -> dict:
    """
    Classify a single report using one-shot prompting.
    """
    user_content = text.strip()

    if examples:
        examples_block = format_examples_block(examples)
        user_content = f"{examples_block}\nNow classify this report:\n{user_content}"

    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_ID,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_content},
                ],
                temperature=0.0,
                top_p=1.0,
                max_tokens=256,
            )
            raw = response.choices[0].message.content.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            result = json.loads(raw)

            if result.get("label") not in LABELS:
                raise ValueError(f"Unexpected label: {result.get('label')!r}")

            return {
                "label": result["label"],
                "explanation": result.get("explanation", "").strip(),
            }

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            if attempt == retries:
                return {"label": "unknown", "explanation": f"Parse error: {e}"}
            time.sleep(1.5 ** attempt)

        except Exception as e:
            if attempt == retries:
                return {"label": "unknown", "explanation": f"API error: {e}"}
            time.sleep(2 ** attempt)

    return {"label": "unknown", "explanation": "Classification failed."}


# ── Batch classification ─────────────────────────────────────────────────────

def classify_batch(
    reports: list[dict],
    stale_hikes: list[dict],
    client: OpenAI,
    examples: list[dict] | None = None,
    weather_map: dict[str, dict] | None = None,
) -> list[dict]:
    """Classify reports + weather-only hikes. Returns enriched predictions."""
    weather_map = weather_map or {}
    results = []

    total = len(reports) + sum(1 for h in stale_hikes if h["hike_id"] in weather_map)
    idx = 0

    # Classify recent reports (report + weather)
    for report in reports:
        idx += 1
        weather = weather_map.get(report.get("hike_id"))
        text = build_input_text(report, weather=weather)
        print(f"  [{idx}/{total}] {report.get('hike_name', '?')[:30]}...", end=" ", flush=True)

        pred = classify_one(text, client, examples=examples)
        print(pred["label"].upper())
        print(f"    {pred['explanation']}")

        enriched = {**report}
        enriched["predicted_label"] = pred["label"]
        enriched["label_explanation"] = pred["explanation"]
        enriched["classification_source"] = "report+weather"
        results.append(enriched)
        time.sleep(API_DELAY_SECONDS)

    # Classify stale hikes (weather only)
    for hike in stale_hikes:
        weather = weather_map.get(hike["hike_id"])
        if not weather:
            continue
        idx += 1
        text = build_weather_only_text(hike, weather)
        print(f"  [{idx}/{total}] {hike.get('hike_name', '?')[:30]} (weather only)...", end=" ", flush=True)

        pred = classify_one(text, client, examples=examples)
        print(pred["label"].upper())
        print(f"    {pred['explanation']}")

        enriched = {**hike}
        enriched["predicted_label"] = pred["label"]
        last_report = hike.get("most_recent_report_date")
        if last_report:
            staleness = f"Most recent report: {last_report}."
        else:
            staleness = "No reports on record."
        enriched["label_explanation"] = (
            f"{staleness} Based on weather only: {pred['explanation']}"
        )
        enriched["classification_source"] = "weather_only"
        results.append(enriched)
        time.sleep(API_DELAY_SECONDS)

    return results
