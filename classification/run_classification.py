"""
Main entry point for trail condition classification.

Modes:
    full [batch_size]   — Classify all trails in batches (default 500).
                          Saves progress and resumes where it left off.
    incremental         — Only classify trails with new data since last run.
    eval                — Test accuracy against human-labeled CSV.

Usage:
    python -m model_sandbox.classification.run_classification full
    python -m model_sandbox.classification.run_classification full 1000
    python -m model_sandbox.classification.run_classification incremental
    python -m model_sandbox.classification.run_classification eval
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

import mlflow
from google.cloud import storage

from .classifier import (
    classify_batch,
    get_nvidia_client,
    load_few_shot_examples,
)
from .config import (
    DEFAULT_BATCH_SIZE,
    GCS_ARTIF_PREFIX,
    GCS_BUCKET_OUTPUT,
    GCS_BUCKET_RAW,
    GCS_INPUT_PREFIX,
    GCS_PRED_PREFIX,
    LABELS,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING,
    MODEL_ID,
    REPORT_MAX_AGE_DAYS,
    STRATEGY,
)
from .eval import main_eval
from .gcs_utils import (
    find_changed_hike_ids,
    get_gcs_client,
    get_last_run_timestamp,
    list_hike_ids,
    load_reports_from_gcs,
    upload_predictions_to_gcs,
)
from .weather import load_weather_for_hikes


# ── Progress tracking ────────────────────────────────────────────────────────

PROGRESS_FILE = "data/classification_progress.json"


def load_progress() -> set[str]:
    """Load set of hike IDs that have already been classified in the current full run."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, encoding="utf-8") as f:
            data = json.load(f)
            return set(data.get("classified_hike_ids", []))
    return set()


def save_progress(classified_ids: set[str]):
    """Save classified hike IDs so the next batch can skip them."""
    os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "classified_hike_ids": sorted(classified_ids),
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }, f, indent=2)


def clear_progress():
    """Remove progress file (called when full run completes)."""
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
        print("Progress file cleared.")


# ── MLflow + GCS logging ────────────────────────────────────────────────────

def log_run(
    predictions: list[dict],
    mode: str,
    gcs_client: storage.Client,
    batch_num: int | None = None,
):
    """Log a classification run to MLflow and upload to GCS."""
    total = len(predictions)
    if not total:
        print("No predictions to log.")
        return

    label_counts = {label: 0 for label in LABELS + ["unknown"]}
    for p in predictions:
        label_counts[p["predicted_label"]] = label_counts.get(p["predicted_label"], 0) + 1

    unknown_count = label_counts.pop("unknown", 0)

    mlflow.set_tracking_uri(MLFLOW_TRACKING)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    suffix = f"-batch{batch_num}" if batch_num else ""
    run_name = f"nemotron-{STRATEGY}-{mode}{suffix}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M')}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model", MODEL_ID)
        mlflow.log_param("strategy", STRATEGY)
        mlflow.log_param("mode", mode)
        mlflow.log_param("num_predictions", total)
        mlflow.log_param("report_max_age_days", REPORT_MAX_AGE_DAYS)
        if batch_num:
            mlflow.log_param("batch_num", batch_num)

        report_count = sum(1 for p in predictions if p.get("classification_source") == "report+weather")
        weather_only = sum(1 for p in predictions if p.get("classification_source") == "weather_only")
        mlflow.log_metric("count_report_based", report_count)
        mlflow.log_metric("count_weather_only", weather_only)

        for label, count in label_counts.items():
            mlflow.log_metric(f"count_{label}", count)
            mlflow.log_metric(f"pct_{label}", round(count / total * 100, 1))

        mlflow.log_metric("count_unknown", unknown_count)

        # Save locally
        local_path = f"data/{STRATEGY}_predictions.json"
        os.makedirs("data", exist_ok=True)
        with open(local_path, "w", encoding="utf-8") as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)

        # Upload predictions to GCS
        gcs_uri = upload_predictions_to_gcs(
            gcs_client, predictions, mode=mode, batch_num=batch_num,
        )
        mlflow.log_param("gcs_output", gcs_uri)

        # Upload artifacts to GCS
        run_id = mlflow.active_run().info.run_id
        bucket = gcs_client.bucket(GCS_BUCKET_OUTPUT)
        blob = bucket.blob(
            f"{GCS_ARTIF_PREFIX}/{MLFLOW_EXPERIMENT}/{run_id}/predictions.json"
        )
        blob.upload_from_string(
            json.dumps(predictions, indent=2, ensure_ascii=False),
            content_type="application/json",
        )

    # Print summary
    print(f"\n--- {mode.upper()} Results ---")
    for label, count in label_counts.items():
        print(f"  {label:<14}: {count} ({count/total*100:.1f}%)")
    if unknown_count:
        print(f"  {'unknown':<14}: {unknown_count}")
    print(f"MLflow run logged: {run_name}\n")


# ── Full mode (batched) ─────────────────────────────────────────────────────

def main_full(batch_size: int = DEFAULT_BATCH_SIZE):
    """
    Classify all trails in batches. Saves progress after each batch
    so it can resume if interrupted.
    """
    gcs_client = get_gcs_client()
    examples = load_few_shot_examples()

    # Get all hike IDs
    all_hike_ids = list_hike_ids(gcs_client, GCS_BUCKET_RAW, GCS_INPUT_PREFIX)
    print(f"Total hikes in GCS: {len(all_hike_ids)}")

    # Load progress from previous batches
    already_done = load_progress()
    remaining = [h for h in all_hike_ids if h not in already_done]
    print(f"Already classified: {len(already_done)}")
    print(f"Remaining: {len(remaining)}")

    if not remaining:
        print("All hikes classified. Clearing progress file.")
        clear_progress()
        return

    batch_num = len(already_done) // batch_size + 1
    total_batches = (len(remaining) + batch_size - 1) // batch_size

    while remaining:
        batch_ids = set(remaining[:batch_size])
        remaining = remaining[batch_size:]

        print(f"\n{'='*65}")
        print(f"BATCH {batch_num}/{batch_num + len(remaining) // batch_size}  |  {len(batch_ids)} hikes")
        print(f"{'='*65}")

        # Load reports for this batch only
        reports, stale_hikes = load_reports_from_gcs(
            gcs_client, GCS_BUCKET_RAW, GCS_INPUT_PREFIX,
            only_hike_ids=batch_ids,
        )

        # Load weather
        weather_hike_ids = list(
            {r["hike_id"] for r in reports} | {h["hike_id"] for h in stale_hikes}
        )
        weather_map = load_weather_for_hikes(gcs_client, weather_hike_ids)

        # Classify
        client = get_nvidia_client()
        predictions = classify_batch(
            reports, stale_hikes, client,
            examples=examples, weather_map=weather_map,
        )

        # Log results
        if predictions:
            log_run(predictions, "full", gcs_client, batch_num=batch_num)

        # Save progress
        already_done.update(batch_ids)
        save_progress(already_done)
        print(f"Progress saved: {len(already_done)}/{len(all_hike_ids)} hikes done")

        batch_num += 1

    print(f"\n{'='*65}")
    print(f"FULL RUN COMPLETE — {len(already_done)} hikes classified")
    print(f"{'='*65}")
    clear_progress()


# ── Incremental mode ─────────────────────────────────────────────────────────

def main_incremental():
    """
    Only classify trails with new data since the last run.
    Falls back to full run if no previous run exists.
    """
    gcs_client = get_gcs_client()
    examples = load_few_shot_examples()

    # Determine what changed
    last_run = get_last_run_timestamp(gcs_client)
    if last_run:
        changed_ids = find_changed_hike_ids(gcs_client, last_run)
        print(f"Last run: {last_run.strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"Trails with new data since then: {len(changed_ids)}")
        if not changed_ids:
            print("Nothing changed. Skipping.")
            return
    else:
        print("No previous run found. Use 'full' mode for initial classification.")
        return

    # Load only changed trails
    reports, stale_hikes = load_reports_from_gcs(
        gcs_client, GCS_BUCKET_RAW, GCS_INPUT_PREFIX,
        only_hike_ids=changed_ids,
    )
    if not reports and not stale_hikes:
        print("No data to classify. Exiting.")
        return

    # Load weather
    weather_hike_ids = list(
        {r["hike_id"] for r in reports} | {h["hike_id"] for h in stale_hikes}
    )
    weather_map = load_weather_for_hikes(gcs_client, weather_hike_ids)

    # Classify
    client = get_nvidia_client()
    predictions = classify_batch(
        reports, stale_hikes, client,
        examples=examples, weather_map=weather_map,
    )

    if predictions:
        log_run(predictions, "incremental", gcs_client)
    print(f"\nClassified {len(predictions)} trails (incremental)")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"

    if mode == "full":
        batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_BATCH_SIZE
        main_full(batch_size=batch_size)
    elif mode == "incremental":
        main_incremental()
    elif mode == "eval":
        main_eval()
    else:
        print("Usage:")
        print(f"  python -m model_sandbox.classification.run_classification full [batch_size]")
        print(f"  python -m model_sandbox.classification.run_classification incremental")
        print(f"  python -m model_sandbox.classification.run_classification eval")


if __name__ == "__main__":
    main()
