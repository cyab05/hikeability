"""
Evaluate zero-shot, one-shot, and few-shot classification strategies
against human-labeled CSV data. Each strategy is logged as a separate
MLflow run so they can be compared in the MLflow UI.

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


STRATEGIES = [
    # (name, per_label examples — 0 means zero-shot)
    ("zero_shot", 0),
    ("one_shot",  1),
    ("few_shot",  3),
]


def _load_labeled(csv_path: str) -> list[dict]:
    labeled = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            true_label = CSV_LABEL_MAP.get(row["label"])
            if true_label:
                labeled.append({
                    "text":       row["comment_text"],
                    "true_label": true_label,
                    "trail_name": row["trail_name"],
                })
    return labeled


def _metrics(true_labels: list[str], preds: list[str]) -> dict:
    correct = sum(1 for t, p in zip(true_labels, preds) if t == p)
    accuracy = correct / len(true_labels) * 100
    out = {"accuracy": round(accuracy, 1)}
    for label in LABELS:
        tp = sum(1 for t, p in zip(true_labels, preds) if t == label and p == label)
        fp = sum(1 for t, p in zip(true_labels, preds) if t != label and p == label)
        fn = sum(1 for t, p in zip(true_labels, preds) if t == label and p != label)
        precision = tp / (tp + fp) * 100 if (tp + fp) else 0
        recall    = tp / (tp + fn) * 100 if (tp + fn) else 0
        out[f"precision_{label}"] = round(precision, 1)
        out[f"recall_{label}"]    = round(recall, 1)
    return out


def _run_strategy(strategy: str, per_label: int, labeled: list[dict], client) -> dict:
    """Run one strategy end-to-end. Returns metrics dict."""
    examples = load_few_shot_examples(per_label=per_label) if per_label > 0 else None

    print("=" * 65)
    print(f"EVALUATING: {strategy.upper()}  (examples={per_label * len(LABELS) if per_label else 0})")
    print("=" * 65)

    preds = []
    for i, item in enumerate(labeled, 1):
        print(f"  [{i}/{len(labeled)}] {item['trail_name'][:30]}...", end=" ", flush=True)
        pred = classify_one(item["text"], client, examples=examples)
        print(f"predicted={pred['label']:<12} actual={item['true_label']}")
        preds.append(pred["label"])
        time.sleep(API_DELAY_SECONDS)

    true_labels = [x["true_label"] for x in labeled]
    metrics = _metrics(true_labels, preds)

    print(f"\n  Accuracy: {metrics['accuracy']}%")
    for label in LABELS:
        print(f"  {label:<12}: precision={metrics[f'precision_{label}']}%  "
              f"recall={metrics[f'recall_{label}']}%")

    misses = [(labeled[i]["trail_name"], true_labels[i], preds[i])
              for i in range(len(true_labels)) if true_labels[i] != preds[i]]
    if misses:
        print(f"\n  Misclassified ({len(misses)}):")
        for trail, actual, predicted in misses:
            print(f"    {trail[:30]}: actual={actual}, predicted={predicted}")

    return metrics


def main_eval(csv_path: str = TRAINING_CSV_PATH):
    labeled = _load_labeled(csv_path)
    print(f"Loaded {len(labeled)} labeled examples from {csv_path}")
    dist = {l: sum(1 for x in labeled if x['true_label'] == l) for l in LABELS}
    print(f"Label distribution: {dist}\n")

    client = get_nvidia_client()

    mlflow.set_tracking_uri(MLFLOW_TRACKING)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    summary = {}

    for strategy, per_label in STRATEGIES:
        metrics = _run_strategy(strategy, per_label, labeled, client)
        summary[strategy] = metrics

        with mlflow.start_run(run_name=f"eval-{strategy}-{ts}"):
            mlflow.log_param("mode", "eval")
            mlflow.log_param("strategy", strategy)
            mlflow.log_param("examples_per_label", per_label)
            mlflow.log_param("total_examples", per_label * len(LABELS))
            mlflow.log_param("eval_dataset", csv_path)
            mlflow.log_param("eval_size", len(labeled))
            for key, value in metrics.items():
                mlflow.log_metric(key, value)

    # Final comparison
    print(f"\n{'='*65}")
    print("STRATEGY COMPARISON")
    print(f"{'='*65}")
    print(f"{'strategy':<12}  {'accuracy':<10}  "
          f"{'prec(H)':<8} {'prec(M)':<8} {'prec(U)':<8}")
    for strategy, _ in STRATEGIES:
        m = summary[strategy]
        print(f"{strategy:<12}  {m['accuracy']:<10}  "
              f"{m['precision_hikeable']:<8} {m['precision_modest']:<8} "
              f"{m['precision_unhikeable']:<8}")

    best = max(summary.items(), key=lambda kv: kv[1]["accuracy"])
    print(f"\nBest strategy: {best[0]} ({best[1]['accuracy']}%)")


if __name__ == "__main__":
    main_eval()
