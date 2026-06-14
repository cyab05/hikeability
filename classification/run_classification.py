"""
Main entry point for trail condition classification.

Modes:
    full [batch_size]   — Classify all trails in batches (default 500).
                          Saves progress and resumes where it left off.
    incremental         — Only classify trails with new data since last run.
    eval                — Test accuracy against human-labeled CSV.

Usage:
    python -m classification.run_classification full
    python -m classification.run_classification full 1000
    python -m classification.run_classification incremental
    python -m classification.run_classification eval
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

from contextlib import contextmanager

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
    write_latest_snapshot,
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

@contextmanager
def _maybe_mlflow_parent(run_name: str):
    """
    Open an MLflow parent run if the tracking server is reachable; otherwise
    yield None so the classification loop still runs and writes to GCS.
    The MLflow VM has gone down repeatedly — never let it fail the daily job.
    """
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        with mlflow.start_run(run_name=run_name) as run:
            yield run
        return
    except Exception as e:
        print(f"[warn] MLflow parent run skipped ({e.__class__.__name__}): {e}")
    yield None


def _label_counts(predictions: list[dict]) -> tuple[dict, int]:
    """Return (label_counts, unknown_count) for a list of predictions."""
    counts = {label: 0 for label in LABELS}
    unknown = 0
    for p in predictions:
        lbl = p["predicted_label"]
        if lbl in counts:
            counts[lbl] += 1
        else:
            unknown += 1
    return counts, unknown


def _log_batch_metrics(predictions: list[dict], gcs_client: storage.Client, batch_num: int | None = None):
    """Upload one batch to GCS and (best-effort) log it as a nested MLflow run."""
    total = len(predictions)
    label_counts, unknown_count = _label_counts(predictions)

    # GCS upload runs first and unconditionally — the snapshot depends on it.
    gcs_uri = upload_predictions_to_gcs(gcs_client, predictions, mode="full", batch_num=batch_num)

    suffix = f"-batch{batch_num}" if batch_num else ""
    run_name = f"batch{suffix}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M')}"

    try:
        with mlflow.start_run(run_name=run_name, nested=True):
            if batch_num:
                mlflow.log_param("batch_num", batch_num)
            mlflow.log_param("num_predictions", total)

            report_count = sum(1 for p in predictions if p.get("classification_source") == "report+weather")
            weather_only = sum(1 for p in predictions if p.get("classification_source") == "weather_only")
            mlflow.log_metric("count_report_based", report_count)
            mlflow.log_metric("count_weather_only", weather_only)

            for label, count in label_counts.items():
                mlflow.log_metric(f"count_{label}", count)
                mlflow.log_metric(f"pct_{label}", round(count / total * 100, 1))
            mlflow.log_metric("count_unknown", unknown_count)
            mlflow.log_param("gcs_output", gcs_uri)
    except Exception as e:
        print(f"  [warn] MLflow batch log skipped ({e.__class__.__name__}): {e}")


def log_run(
    predictions: list[dict],
    mode: str,
    gcs_client: storage.Client,
    batch_num: int | None = None,
    parent_run_id: str | None = None,
):
    """Log a classification run to MLflow and upload to GCS.

    For full-mode batches, pass parent_run_id to attach as a nested run.
    For incremental runs (no parent), creates a standalone run as before.
    """
    total = len(predictions)
    if not total:
        print("No predictions to log.")
        return

    label_counts, unknown_count = _label_counts(predictions)

    if batch_num is not None:
        # Batch within a full run — nested MLflow + per-batch GCS upload.
        # Branch on batch_num (not parent_run_id) so the batch path still runs
        # when MLflow is unreachable and parent_run_id ends up None.
        _log_batch_metrics(predictions, gcs_client, batch_num=batch_num)
    else:
        # Standalone run (incremental). GCS upload runs unconditionally; MLflow
        # logging is best-effort so a downed tracking server can't fail the job.
        local_path = f"data/{STRATEGY}_predictions.json"
        os.makedirs("data", exist_ok=True)
        with open(local_path, "w", encoding="utf-8") as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)
        gcs_uri = upload_predictions_to_gcs(gcs_client, predictions, mode=mode)

        run_name = f"nemotron-{STRATEGY}-{mode}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M')}"
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING)
            mlflow.set_experiment(MLFLOW_EXPERIMENT)
            with mlflow.start_run(run_name=run_name):
                mlflow.log_param("model", MODEL_ID)
                mlflow.log_param("strategy", STRATEGY)
                mlflow.log_param("mode", mode)
                mlflow.log_param("num_predictions", total)
                mlflow.log_param("report_max_age_days", REPORT_MAX_AGE_DAYS)

                report_count = sum(1 for p in predictions if p.get("classification_source") == "report+weather")
                weather_only = sum(1 for p in predictions if p.get("classification_source") == "weather_only")
                mlflow.log_metric("count_report_based", report_count)
                mlflow.log_metric("count_weather_only", weather_only)

                for label, count in label_counts.items():
                    mlflow.log_metric(f"count_{label}", count)
                    mlflow.log_metric(f"pct_{label}", round(count / total * 100, 1))
                mlflow.log_metric("count_unknown", unknown_count)
                mlflow.log_param("gcs_output", gcs_uri)

                run_id = mlflow.active_run().info.run_id
                bucket = gcs_client.bucket(GCS_BUCKET_OUTPUT)
                blob = bucket.blob(
                    f"{GCS_ARTIF_PREFIX}/{MLFLOW_EXPERIMENT}/{run_id}/predictions.json"
                )
                blob.upload_from_string(
                    json.dumps(predictions, indent=2, ensure_ascii=False),
                    content_type="application/json",
                )
            print(f"MLflow run logged: {run_name}")
        except Exception as e:
            print(f"[warn] MLflow logging skipped ({e.__class__.__name__}): {e}")

    # Print summary
    print(f"\n--- {mode.upper()} Results ---")
    for label, count in label_counts.items():
        print(f"  {label:<14}: {count} ({count/total*100:.1f}%)")
    if unknown_count:
        print(f"  {'unknown':<14}: {unknown_count}")


# ── Full mode (batched) ─────────────────────────────────────────────────────

def main_full(batch_size: int = DEFAULT_BATCH_SIZE):
    """
    Classify all trails in batches. Saves progress after each batch
    so it can resume if interrupted.

    All batches are logged as nested runs under one parent MLflow run so the
    UI shows a single entry per full classification instead of one per batch.
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
    all_predictions: list[dict] = []

    parent_run_name = f"nemotron-{STRATEGY}-full-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M')}"

    with _maybe_mlflow_parent(parent_run_name) as parent_run:
        if parent_run is not None:
            try:
                mlflow.log_param("model", MODEL_ID)
                mlflow.log_param("strategy", STRATEGY)
                mlflow.log_param("mode", "full")
                mlflow.log_param("total_hikes", len(all_hike_ids))
                mlflow.log_param("report_max_age_days", REPORT_MAX_AGE_DAYS)
            except Exception as e:
                print(f"[warn] MLflow parent params skipped ({e.__class__.__name__}): {e}")

        client = get_nvidia_client()

        while remaining:
            batch_ids = set(remaining[:batch_size])
            remaining = remaining[batch_size:]

            print(f"\n{'='*65}")
            print(f"BATCH {batch_num}  |  {len(batch_ids)} hikes")
            print(f"{'='*65}")

            reports, stale_hikes = load_reports_from_gcs(
                gcs_client, GCS_BUCKET_RAW, GCS_INPUT_PREFIX,
                only_hike_ids=batch_ids,
            )

            weather_hike_ids = list(
                {r["hike_id"] for r in reports} | {h["hike_id"] for h in stale_hikes}
            )
            weather_map = load_weather_for_hikes(gcs_client, weather_hike_ids)

            predictions = classify_batch(
                reports, stale_hikes, client,
                examples=examples, weather_map=weather_map,
            )

            if predictions:
                log_run(
                    predictions, "full", gcs_client,
                    batch_num=batch_num,
                    parent_run_id=parent_run.info.run_id if parent_run else None,
                )
                all_predictions.extend(predictions)

            already_done.update(batch_ids)
            save_progress(already_done)
            print(f"Progress saved: {len(already_done)}/{len(all_hike_ids)} hikes done")
            batch_num += 1

        # Aggregate totals + final artifact. Local snapshot always runs;
        # MLflow metrics + artifact upload are best-effort.
        total = len(all_predictions)
        if total:
            local_path = f"data/{STRATEGY}_predictions.json"
            os.makedirs("data", exist_ok=True)
            with open(local_path, "w", encoding="utf-8") as f:
                json.dump(all_predictions, f, indent=2, ensure_ascii=False)

            if parent_run is not None:
                try:
                    label_counts, unknown_count = _label_counts(all_predictions)
                    mlflow.log_metric("total_classified", total)
                    for label, count in label_counts.items():
                        mlflow.log_metric(f"total_count_{label}", count)
                        mlflow.log_metric(f"total_pct_{label}", round(count / total * 100, 1))
                    mlflow.log_metric("total_count_unknown", unknown_count)

                    run_id = parent_run.info.run_id
                    bucket = gcs_client.bucket(GCS_BUCKET_OUTPUT)
                    blob = bucket.blob(
                        f"{GCS_ARTIF_PREFIX}/{MLFLOW_EXPERIMENT}/{run_id}/predictions.json"
                    )
                    blob.upload_from_string(
                        json.dumps(all_predictions, indent=2, ensure_ascii=False),
                        content_type="application/json",
                    )
                except Exception as e:
                    print(f"[warn] MLflow aggregate logging skipped ({e.__class__.__name__}): {e}")

    print(f"\n{'='*65}")
    print(f"FULL RUN COMPLETE — {len(already_done)} hikes classified")
    print(f"{'='*65}")
    clear_progress()

    # Consolidate every daily run into one snapshot for the web app cold-start.
    write_latest_snapshot(gcs_client)


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
        print("No data to classify.")
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

    # Refresh the merged snapshot the web app reads on cold start.
    write_latest_snapshot(gcs_client)


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
        print(f"  python -m classification.run_classification full [batch_size]")
        print(f"  python -m classification.run_classification incremental")
        print(f"  python -m classification.run_classification eval")


if __name__ == "__main__":
    main()
