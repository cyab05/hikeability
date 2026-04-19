"""
Evaluate one-shot classification accuracy against human-labeled CSV data.
Logs results to MLflow.

Usage:
    python -m model_sandbox.classification.eval
"""
from __future__ import annotations

import csv
import time
from datetime import datetime, timezone

import mlflow

from .classifier import classify_one, get_nvidia_client, load_few_shot_examples
from .config import (
    API_DELAY_SECONDS,
    CSV_LABEL_MAP,
    LABELS,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING,
    TRAINING_CSV_PATH,
)


def main_eval(csv_path: str = TRAINING_CSV_PATH):
    """
    Run one-shot strategy against human-labeled data.
    Prints accuracy, per-label precision/recall, and misclassifications.
    """
    # 1. Load human-labeled data
    labeled = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            true_label = CSV_LABEL_MAP.get(row["label"])
            if true_label:
                labeled.append({
                    "text": row["comment_text"],
                    "true_label": true_label,
                    "trail_name": row["trail_name"],
                })

    print(f"Loaded {len(labeled)} labeled examples from {csv_path}")
    print(f"Label distribution: { {l: sum(1 for x in labeled if x['true_label'] == l) for l in LABELS} }\n")

    # 2. Load one-shot examples
    examples = load_few_shot_examples()
    print(f"Loaded {len(examples)} one-shot examples\n")

    client = get_nvidia_client()

    # 3. Classify all labeled examples
    print("=" * 65)
    print("EVALUATING: ONE_SHOT")
    print("=" * 65)

    preds = []
    for i, item in enumerate(labeled, 1):
        print(f"  [{i}/{len(labeled)}] {item['trail_name'][:30]}...", end=" ", flush=True)
        pred = classify_one(item["text"], client, examples=examples)
        print(f"predicted={pred['label']:<12} actual={item['true_label']}")
        preds.append(pred["label"])
        time.sleep(API_DELAY_SECONDS)

    # 4. Calculate results
    true_labels = [x["true_label"] for x in labeled]
    correct = sum(1 for t, p in zip(true_labels, preds) if t == p)
    accuracy = correct / len(true_labels) * 100

    print(f"\n{'='*65}")
    print("EVALUATION RESULTS")
    print(f"{'='*65}")
    print(f"\n  Accuracy: {correct}/{len(true_labels)} ({accuracy:.1f}%)")

    for label in LABELS:
        true_pos = sum(1 for t, p in zip(true_labels, preds) if t == label and p == label)
        false_pos = sum(1 for t, p in zip(true_labels, preds) if t != label and p == label)
        false_neg = sum(1 for t, p in zip(true_labels, preds) if t == label and p != label)
        precision = true_pos / (true_pos + false_pos) * 100 if (true_pos + false_pos) else 0
        recall = true_pos / (true_pos + false_neg) * 100 if (true_pos + false_neg) else 0
        print(f"  {label:<12}: precision={precision:.0f}%  recall={recall:.0f}%")

    misses = [(labeled[i]["trail_name"], true_labels[i], preds[i])
              for i in range(len(true_labels)) if true_labels[i] != preds[i]]
    if misses:
        print(f"\n  Misclassified ({len(misses)}):")
        for trail, actual, predicted in misses:
            print(f"    {trail[:30]}: actual={actual}, predicted={predicted}")

    # 5. Log to MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=f"eval-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M')}"):
        mlflow.log_param("mode", "eval")
        mlflow.log_param("strategy", "one_shot")
        mlflow.log_param("eval_dataset", csv_path)
        mlflow.log_param("eval_size", len(labeled))
        mlflow.log_metric("accuracy", round(accuracy, 1))

    print(f"\n{'='*65}")
    print(f"ACCURACY: {accuracy:.1f}%")
    print(f"{'='*65}")


if __name__ == "__main__":
    main_eval()
