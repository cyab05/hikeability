from __future__ import annotations

import os
import warnings
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.pytorch
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

warnings.filterwarnings("ignore")


DATA_PATH           = "wta_training_data.csv"
TARGET_COL          = "label"
LABEL_ORDER         = ["hikeable", "modest_conditions", "not_hikeable"]
PRETRAINED_MODEL    = "distilbert-base-uncased"
MLFLOW_EXPERIMENT   = "trail-condition-nlp"
RANDOM_STATE        = 42

HPARAMS = {
    "max_length":    128,
    "batch_size":    16,
    "epochs":        4,
    "learning_rate": 2e-5,
    "warmup_ratio":  0.1,
    "weight_decay":  0.01,
    "dropout":       0.1,
    "test_size":     0.2,
    "pretrained":    PRETRAINED_MODEL,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class TrailDataset(Dataset):
    """Tokenises comment text and returns tensors for DistilBERT."""

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: DistilBertTokenizerFast,
        max_length: int,
    ):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }



def train_epoch(model, loader, optimizer, scheduler, device) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)
        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[list[int], list[int]]:
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = outputs.logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    return all_labels, all_preds


def save_confusion_matrix(y_true, y_pred, label_names, path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_names, yticklabels=label_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)



def run_experiment(
    X_train_texts, X_test_texts,
    y_train_ids,   y_test_ids,
    label2id: dict[str, int],
    id2label: dict[int, str],
    hparams: dict,
    run_name: str,
) -> tuple[str, float]:
    """Fine-tune DistilBERT and log everything to MLflow."""

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(hparams)
        mlflow.log_param("device", str(DEVICE))
        mlflow.log_param("train_samples", len(X_train_texts))
        mlflow.log_param("test_samples",  len(X_test_texts))

        tokenizer = DistilBertTokenizerFast.from_pretrained(hparams["pretrained"])

        train_dataset = TrailDataset(
            X_train_texts, y_train_ids, tokenizer, hparams["max_length"]
        )
        test_dataset = TrailDataset(
            X_test_texts, y_test_ids, tokenizer, hparams["max_length"]
        )
        train_loader = DataLoader(
            train_dataset, batch_size=hparams["batch_size"], shuffle=True
        )
        test_loader  = DataLoader(
            test_dataset,  batch_size=hparams["batch_size"], shuffle=False
        )

        model = DistilBertForSequenceClassification.from_pretrained(
            hparams["pretrained"],
            num_labels=len(label2id),
            id2label=id2label,
            label2id=label2id,
            seq_classif_dropout=hparams["dropout"],
        ).to(DEVICE)

        optimizer = AdamW(
            model.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"],
        )
        total_steps   = len(train_loader) * hparams["epochs"]
        warmup_steps  = int(total_steps * hparams["warmup_ratio"])
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        best_f1    = 0.0
        best_epoch = 0
        for epoch in range(1, hparams["epochs"] + 1):
            train_loss = train_epoch(model, train_loader, optimizer, scheduler, DEVICE)
            y_true, y_pred = evaluate(model, test_loader, DEVICE)
            acc      = accuracy_score(y_true, y_pred)
            f1_macro = f1_score(y_true, y_pred, average="macro")

            # Log per-epoch metrics
            mlflow.log_metric("train_loss",  train_loss, step=epoch)
            mlflow.log_metric("eval_accuracy", acc,      step=epoch)
            mlflow.log_metric("eval_f1_macro", f1_macro, step=epoch)

            print(f"  Epoch {epoch}/{hparams['epochs']}  "
                  f"loss={train_loss:.4f}  acc={acc:.3f}  f1={f1_macro:.3f}")

            if f1_macro > best_f1:
                best_f1    = f1_macro
                best_epoch = epoch

        y_true, y_pred = evaluate(model, test_loader, DEVICE)
        label_names    = [id2label[i] for i in sorted(id2label)]
        report         = classification_report(
            y_true, y_pred,
            target_names=label_names,
            output_dict=True,
        )

        final_acc      = accuracy_score(y_true, y_pred)
        final_f1_macro = f1_score(y_true, y_pred, average="macro")
        final_f1_wt    = f1_score(y_true, y_pred, average="weighted")

        mlflow.log_metric("final_accuracy",   final_acc)
        mlflow.log_metric("final_f1_macro",   final_f1_macro)
        mlflow.log_metric("final_f1_weighted", final_f1_wt)
        mlflow.log_metric("best_epoch",       best_epoch)

        for label in label_names:
            if label in report:
                mlflow.log_metric(f"f1_{label}",        report[label]["f1-score"])
                mlflow.log_metric(f"precision_{label}", report[label]["precision"])
                mlflow.log_metric(f"recall_{label}",    report[label]["recall"])

        # ── Confusion matrix artifact ──────────────────────────────────────
        cm_path = f"/tmp/confusion_matrix_{run_name}.png"
        save_confusion_matrix(y_true, y_pred, label_names, cm_path)
        try:
            mlflow.log_artifact(cm_path, artifact_path="plots")
        except Exception as e:
            print(f"  ⚠ Could not log confusion matrix artifact: {e}")

        # ── Log classification report as JSON artifact ─────────────────────
        report_path = f"/tmp/classification_report_{run_name}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        try:
            mlflow.log_artifact(report_path, artifact_path="reports")
        except Exception as e:
            print(f"  ⚠ Could not log classification report artifact: {e}")


        input_schema  = Schema([ColSpec("string", "comment_text")])
        output_schema = Schema([ColSpec("string", "label")])
        signature     = ModelSignature(inputs=input_schema, outputs=output_schema)

        try:
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path="model",
                signature=signature,
                registered_model_name=f"trail-condition-{run_name}",
            )
        except Exception as e:
            print(f"Could not log model artifact: {e}")

        tokenizer_path = f"/tmp/tokenizer_{run_name}"
        tokenizer.save_pretrained(tokenizer_path)
        try:
            mlflow.log_artifacts(tokenizer_path, artifact_path="tokenizer")
        except Exception as e:
            print(f"Could not log tokenizer artifact: {e}")

        print(f"\n  ── {run_name} final results ──")
        print(f"  Accuracy  : {final_acc:.3f}")
        print(f"  F1 macro  : {final_f1_macro:.3f}")
        print(f"  F1 wtd    : {final_f1_wt:.3f}")
        print(f"  Best epoch: {best_epoch}")
        print(classification_report(y_true, y_pred, target_names=label_names))

        run_id = mlflow.active_run().info.run_id
        return run_id, final_f1_macro



def predict(texts: list[str], model_uri: str) -> list[str]:
    """
    Load a logged model and return predicted labels.

    Example
    -------
    labels = predict(
        ["Trail is clear and dry all the way!"],
        model_uri="runs:/<run_id>/model",
    )
    """
    # Reload tokenizer from the artifact store
    tokenizer = DistilBertTokenizerFast.from_pretrained(PRETRAINED_MODEL)
    model     = mlflow.pytorch.load_model(model_uri).to(DEVICE)
    model.eval()

    encodings = tokenizer(
        texts, truncation=True, padding="max_length",
        max_length=HPARAMS["max_length"], return_tensors="pt"
    )
    with torch.no_grad():
        logits = model(
            input_ids=encodings["input_ids"].to(DEVICE),
            attention_mask=encodings["attention_mask"].to(DEVICE),
        ).logits
    pred_ids = logits.argmax(dim=-1).cpu().tolist()
    return [model.config.id2label[i] for i in pred_ids]



def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows.")
    print(f"Label distribution:\n{df[TARGET_COL].value_counts()}\n")
    print(f"Running on: {DEVICE}\n")

    label2id = {lbl: i for i, lbl in enumerate(LABEL_ORDER)}
    id2label = {i: lbl for lbl, i in label2id.items()}

    texts  = df["comment_text"].fillna("").tolist()
    labels = df[TARGET_COL].map(label2id).tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels,
        test_size=HPARAMS["test_size"],
        stratify=labels,
        random_state=RANDOM_STATE,
    )

    mlflow.set_tracking_uri("http://35.202.68.106:5000/")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    results = []

    # ── Run 1: Default hyper-parameters ───────────────────────────────────
    print("=" * 60)
    print("Run 1: distilbert — default hyper-parameters")
    print("=" * 60)
    run_id, f1 = run_experiment(
        X_train, X_test, y_train, y_test,
        label2id, id2label,
        hparams=HPARAMS,
        run_name="distilbert-default",
    )
    results.append(("distilbert-default", run_id, f1))

    # ── Run 2: Higher learning rate ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Run 2: distilbert — higher LR (3e-5)")
    print("=" * 60)
    hparams_v2 = {**HPARAMS, "learning_rate": 3e-5}
    run_id, f1 = run_experiment(
        X_train, X_test, y_train, y_test,
        label2id, id2label,
        hparams=hparams_v2,
        run_name="distilbert-lr3e5",
    )
    results.append(("distilbert-lr3e5", run_id, f1))

    # ── Run 3: Longer sequences ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Run 3: distilbert — longer token sequences (256)")
    print("=" * 60)
    hparams_v3 = {**HPARAMS, "max_length": 256}
    run_id, f1 = run_experiment(
        X_train, X_test, y_train, y_test,
        label2id, id2label,
        hparams=hparams_v3,
        run_name="distilbert-maxlen256",
    )
    results.append(("distilbert-maxlen256", run_id, f1))

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY  (ranked by final F1 macro)")
    print("=" * 60)
    results.sort(key=lambda x: x[2], reverse=True)
    for rank, (name, run_id, f1) in enumerate(results, 1):
        print(f"  {rank}. {name:<35}  F1={f1:.3f}  run_id={run_id[:8]}...")

    best_name, best_run_id, best_f1 = results[0]
    print(f"\n✓ Best run : {best_name}  (F1={best_f1:.3f})")
    print(f"  Run ID   : {best_run_id}")
    print("\nTo browse all runs:")
    print("  mlflow ui  →  http://35.202.68.106:5000\n")

    # ── Example inference with the best model ─────────────────────────────
    sample_comments = [
        "Trail is completely snow-free and dry. Parking lot was open. Great views!",
        "Some patchy snow above mile 3. Microspikes recommended for upper section.",
        "Road is closed. Trail buried under 4 feet of snow. Do not attempt.",
    ]
    print("Sample predictions (best model):")
    preds = predict(sample_comments, model_uri=f"runs:/{best_run_id}/model")
    for text, pred in zip(sample_comments, preds):
        print(f"  [{pred}]  {text[:70]}...")


if __name__ == "__main__":
    main()