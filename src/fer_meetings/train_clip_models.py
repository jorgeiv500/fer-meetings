import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from fer_meetings.clip_models import (
    fit_attention_pooler,
    fit_hist_gradient_probe,
    fit_logistic_probe,
    labels_to_indices,
    metric_bundle,
    predict_attention_pooler,
    predict_probe,
)
from fer_meetings.constants import LABEL_ORDER
from fer_meetings.labels import resolve_gold_label
from fer_meetings.utils import read_csv_rows, write_csv_rows


def parse_args():
    parser = argparse.ArgumentParser(description="Train clip-level classifiers from exported frame embeddings.")
    parser.add_argument("--clip-features", required=True, help="CSV written by fer-run-pilot --clip-features-output.")
    parser.add_argument("--labels", required=True, help="Labels CSV with gold_label column.")
    parser.add_argument("--output-dir", required=True, help="Directory where metrics, predictions and summaries are written.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or mps.")
    return parser.parse_args()


def select_device(requested_device):
    if requested_device != "auto":
        return requested_device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def merge_features_and_labels(features, labels):
    labels_by_clip = {row["clip_id"]: row for row in labels}
    merged_rows = []
    for feature_row in features:
        label_row = labels_by_clip.get(feature_row["clip_id"])
        if not label_row:
            continue
        gold_label, label_source = resolve_gold_label(label_row)
        if not gold_label:
            continue
        merged_row = dict(feature_row)
        merged_row["gold_label"] = gold_label
        merged_row["label_source"] = label_source
        merged_rows.append(merged_row)
    return merged_rows


def rows_for_split(rows, split_name):
    return [row for row in rows if row["split"] == split_name]


def group_rows_by_model(rows):
    grouped = {}
    for row in rows:
        key = (
            row.get("model_name", "default"),
            row.get("model_family", "unknown"),
            row.get("hf_model_id", ""),
        )
        grouped.setdefault(key, []).append(row)
    return grouped


def build_prediction_rows(model_key, rows, method_name, predicted_indices, probabilities=None, attention_weights=None):
    model_name, model_family, model_id = model_key
    prediction_rows = []
    for index, (row, predicted_index) in enumerate(zip(rows, predicted_indices)):
        prediction_row = {
            "clip_id": row["clip_id"],
            "split": row["split"],
            "model_name": model_name,
            "model_family": model_family,
            "hf_model_id": model_id,
            "method": method_name,
            "gold_label": row["gold_label"],
            "predicted_label": LABEL_ORDER[int(predicted_index)],
        }
        if probabilities is not None:
            for label_index, label in enumerate(LABEL_ORDER):
                prediction_row[f"{label}_prob"] = f"{float(probabilities[index][label_index]):.6f}"
        if attention_weights is not None:
            prediction_row["attention_weights_json"] = json.dumps(
                [float(weight) for weight in attention_weights[index].tolist()],
                separators=(",", ":"),
            )
        prediction_rows.append(prediction_row)
    return prediction_rows


def per_class_metric_rows(model_key, method_name, rows, predicted_indices):
    model_name, model_family, model_id = model_key
    y_true = [row["gold_label"] for row in rows]
    y_pred = [LABEL_ORDER[int(index)] for index in predicted_indices]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABEL_ORDER,
        zero_division=0,
    )
    metric_rows = []
    for index, label in enumerate(LABEL_ORDER):
        metric_rows.append(
            {
                "model_name": model_name,
                "model_family": model_family,
                "hf_model_id": model_id,
                "method": method_name,
                "label": label,
                "precision": precision[index],
                "recall": recall[index],
                "f1": f1[index],
                "support": int(support[index]),
            }
        )
    return metric_rows


def confusion_rows(model_key, method_name, rows, predicted_indices):
    model_name, model_family, model_id = model_key
    y_true = [row["gold_label"] for row in rows]
    y_pred = [LABEL_ORDER[int(index)] for index in predicted_indices]
    matrix = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)
    result = []
    for gold_index, gold_label in enumerate(LABEL_ORDER):
        result.append(
            {
                "model_name": model_name,
                "model_family": model_family,
                "hf_model_id": model_id,
                "method": method_name,
                "gold_label": gold_label,
                "negative": int(matrix[gold_index][0]),
                "neutral": int(matrix[gold_index][1]),
                "positive": int(matrix[gold_index][2]),
            }
        )
    return result


def write_summary(output_dir, metrics_rows, notes):
    summary_path = Path(output_dir) / "clip_model_summary.md"
    lines = [
        "# Clip-Level Model Summary",
        "",
        "| model | family | method | n_clips | accuracy | balanced_accuracy | macro_f1 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in metrics_rows:
        lines.append(
            f"| {row['model_name']} | {row['model_family']} | {row['method']} | {row['n_clips']} | "
            f"{row['accuracy']:.4f} | {row['balanced_accuracy']:.4f} | {row['macro_f1']:.4f} |"
        )

    if notes:
        lines.extend(["", "## Notes", ""])
        for note in notes:
            lines.append(f"- {note}")

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    features = read_csv_rows(args.clip_features)
    labels = read_csv_rows(args.labels)
    merged_rows = merge_features_and_labels(features, labels)
    if not merged_rows:
        raise RuntimeError("No labeled clip features were available after merging embeddings and labels.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)

    metrics_rows = []
    prediction_rows = []
    per_class_rows = []
    confusion_matrix_rows = []
    notes = []

    for model_key, model_rows in group_rows_by_model(merged_rows).items():
        model_name, model_family, model_id = model_key
        train_rows = rows_for_split(model_rows, "dev")
        test_rows = rows_for_split(model_rows, "test")

        if not train_rows or not test_rows:
            notes.append(f"Skipping {model_name}: both split=dev and split=test are required.")
            continue

        train_labels = labels_to_indices(train_rows)
        if len(set(train_labels.tolist())) < 2:
            notes.append(f"Skipping {model_name}: split=dev needs at least two classes for supervised clip training.")
            continue

        logistic_model = fit_logistic_probe(train_rows)
        logistic_predictions = predict_probe(logistic_model, test_rows)
        logistic_metrics = metric_bundle(labels_to_indices(test_rows), logistic_predictions)
        metrics_rows.append(
            {
                "model_name": model_name,
                "model_family": model_family,
                "hf_model_id": model_id,
                "method": "mean_embedding_logreg",
                **logistic_metrics,
            }
        )
        prediction_rows.extend(build_prediction_rows(model_key, test_rows, "mean_embedding_logreg", logistic_predictions))
        per_class_rows.extend(per_class_metric_rows(model_key, "mean_embedding_logreg", test_rows, logistic_predictions))
        confusion_matrix_rows.extend(confusion_rows(model_key, "mean_embedding_logreg", test_rows, logistic_predictions))

        hist_model = fit_hist_gradient_probe(train_rows)
        hist_predictions = predict_probe(hist_model, test_rows)
        hist_metrics = metric_bundle(labels_to_indices(test_rows), hist_predictions)
        metrics_rows.append(
            {
                "model_name": model_name,
                "model_family": model_family,
                "hf_model_id": model_id,
                "method": "mean_embedding_hgb",
                **hist_metrics,
            }
        )
        prediction_rows.extend(build_prediction_rows(model_key, test_rows, "mean_embedding_hgb", hist_predictions))
        per_class_rows.extend(per_class_metric_rows(model_key, "mean_embedding_hgb", test_rows, hist_predictions))
        confusion_matrix_rows.extend(confusion_rows(model_key, "mean_embedding_hgb", test_rows, hist_predictions))

        try:
            attention_model = fit_attention_pooler(train_rows, device=device)
            attention_predictions, attention_probabilities, attention_weights = predict_attention_pooler(
                attention_model,
                test_rows,
                device=device,
            )
            attention_metrics = metric_bundle(labels_to_indices(test_rows), attention_predictions)
            metrics_rows.append(
                {
                    "model_name": model_name,
                    "model_family": model_family,
                    "hf_model_id": model_id,
                    "method": "attention_pooling",
                    **attention_metrics,
                }
            )
            prediction_rows.extend(
                build_prediction_rows(
                    model_key,
                    test_rows,
                    "attention_pooling",
                    attention_predictions,
                    probabilities=attention_probabilities,
                    attention_weights=attention_weights,
                )
            )
            per_class_rows.extend(per_class_metric_rows(model_key, "attention_pooling", test_rows, attention_predictions))
            confusion_matrix_rows.extend(confusion_rows(model_key, "attention_pooling", test_rows, attention_predictions))
        except ValueError as error:
            notes.append(f"Attention pooling skipped for {model_name}: {error}")

    metrics_rows = sorted(metrics_rows, key=lambda row: (row["model_name"], row["method"]))
    per_class_rows = sorted(per_class_rows, key=lambda row: (row["model_name"], row["method"], row["label"]))
    metrics_path = output_dir / "clip_model_metrics.json"
    metrics_path.write_text(json.dumps(metrics_rows, indent=2), encoding="utf-8")
    write_csv_rows(
        output_dir / "clip_model_metrics.csv",
        metrics_rows,
        [
            "model_name",
            "model_family",
            "hf_model_id",
            "method",
            "n_clips",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
        ],
    )
    write_csv_rows(
        output_dir / "clip_model_per_class_metrics.csv",
        per_class_rows,
        [
            "model_name",
            "model_family",
            "hf_model_id",
            "method",
            "label",
            "precision",
            "recall",
            "f1",
            "support",
        ],
    )
    write_csv_rows(
        output_dir / "clip_model_confusion_matrices.csv",
        confusion_matrix_rows,
        ["model_name", "model_family", "hf_model_id", "method", "gold_label"] + LABEL_ORDER,
    )

    fieldnames = [
        "clip_id",
        "split",
        "model_name",
        "model_family",
        "hf_model_id",
        "method",
        "gold_label",
        "predicted_label",
    ]
    probability_fields = [f"{label}_prob" for label in LABEL_ORDER]
    has_probabilities = any(any(field in row for field in probability_fields) for row in prediction_rows)
    has_attention = any("attention_weights_json" in row for row in prediction_rows)
    if has_probabilities:
        fieldnames.extend(probability_fields)
    if has_attention:
        fieldnames.append("attention_weights_json")
    write_csv_rows(output_dir / "clip_model_predictions.csv", prediction_rows, fieldnames)
    write_summary(output_dir, metrics_rows, notes)
    print(f"Wrote clip-level metrics to {metrics_path}")
    print(f"Wrote clip-level predictions to {output_dir / 'clip_model_predictions.csv'}")


if __name__ == "__main__":
    main()
