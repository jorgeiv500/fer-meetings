import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from fer_meetings.constants import LABEL_ORDER
from fer_meetings.labels import resolve_gold_label
from fer_meetings.utils import ensure_parent, read_csv_rows, write_csv_rows


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate clip-level FER predictions against manual labels.")
    parser.add_argument("--predictions", required=True, help="Predictions CSV from fer-run-pilot.")
    parser.add_argument("--labels", required=True, help="Labels CSV with gold_label column.")
    parser.add_argument("--output-dir", required=True, help="Directory where metrics and tables will be written.")
    parser.add_argument(
        "--fit-calibrator",
        action="store_true",
        help="Fit a multinomial logistic regression on split=dev and apply it to split=test.",
    )
    return parser.parse_args()


def merge_predictions_and_labels(predictions, labels):
    labels_by_clip = {row["clip_id"]: row for row in labels}
    merged_rows = []
    for prediction in predictions:
        label_row = labels_by_clip.get(prediction["clip_id"])
        if label_row is None:
            continue
        gold_label, label_source = resolve_gold_label(label_row)
        if not gold_label:
            continue
        merged_row = dict(prediction)
        merged_row["gold_label"] = gold_label
        merged_row["label_source"] = label_source
        merged_rows.append(merged_row)
    return merged_rows


def rows_for_scope(rows, split_name):
    if split_name == "all":
        return rows
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


def metric_bundle(rows, prediction_column):
    y_true = [row["gold_label"] for row in rows]
    y_pred = [row[prediction_column] for row in rows]
    matrix = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)

    return {
        "n_clips": len(rows),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=LABEL_ORDER, average="macro", zero_division=0),
        "confusion_matrix": matrix.tolist(),
    }


def per_class_metric_rows(model_name, model_family, model_id, scope, method, rows, prediction_column):
    y_true = [row["gold_label"] for row in rows]
    y_pred = [row[prediction_column] for row in rows]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABEL_ORDER,
        zero_division=0,
    )
    per_class_rows = []
    for index, label in enumerate(LABEL_ORDER):
        per_class_rows.append(
            {
                "model_name": model_name,
                "model_family": model_family,
                "hf_model_id": model_id,
                "scope": scope,
                "method": method,
                "label": label,
                "precision": precision[index],
                "recall": recall[index],
                "f1": f1[index],
                "support": int(support[index]),
            }
        )
    return per_class_rows


def probability_matrix(rows, prefix):
    return np.array(
        [
            [float(row[f"{prefix}_{label}_prob"]) for label in LABEL_ORDER]
            for row in rows
        ],
        dtype=float,
    )


def maybe_fit_calibrator(rows):
    dev_rows = rows_for_scope(rows, "dev")
    test_rows = rows_for_scope(rows, "test")
    if not dev_rows or not test_rows:
        return None

    dev_labels = [LABEL_ORDER.index(row["gold_label"]) for row in dev_rows]
    if len(set(dev_labels)) < 2:
        return None

    model = LogisticRegression(max_iter=1000, multi_class="multinomial")
    model.fit(probability_matrix(dev_rows, "smoothed"), dev_labels)
    calibrated_indices = model.predict(probability_matrix(test_rows, "smoothed"))

    calibrated_rows = []
    for row, predicted_index in zip(test_rows, calibrated_indices):
        calibrated_row = dict(row)
        calibrated_row["smoothed_calibrated"] = LABEL_ORDER[int(predicted_index)]
        calibrated_rows.append(calibrated_row)
    return calibrated_rows


def matrix_to_rows(model_name, model_family, model_id, scope, method, matrix):
    rows = []
    for gold_index, gold_label in enumerate(LABEL_ORDER):
        row = {
            "model_name": model_name,
            "model_family": model_family,
            "hf_model_id": model_id,
            "scope": scope,
            "method": method,
            "gold_label": gold_label,
        }
        for predicted_index, predicted_label in enumerate(LABEL_ORDER):
            row[predicted_label] = matrix[gold_index][predicted_index]
        rows.append(row)
    return rows


def write_summary(output_dir, metrics_rows, notes):
    summary_path = Path(output_dir) / "summary.md"
    lines = [
        "# Pilot Summary",
        "",
        "| model | family | scope | method | n_clips | accuracy | balanced_accuracy | macro_f1 |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in metrics_rows:
        lines.append(
            f"| {row['model_name']} | {row['model_family']} | {row['scope']} | {row['method']} | {row['n_clips']} | "
            f"{row['accuracy']:.4f} | {row['balanced_accuracy']:.4f} | {row['macro_f1']:.4f} |"
        )

    if notes:
        lines.extend(["", "## Notes", ""])
        for note in notes:
            lines.append(f"- {note}")

    ensure_parent(summary_path)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_labeled_predictions(path, rows):
    if not rows:
        return

    preferred_order = [
        "clip_id",
        "split",
        "meeting_id",
        "speaker_id",
        "camera",
        "video_file",
        "video_path",
        "clip_start_s",
        "clip_end_s",
        "video_duration_s",
        "source_dataset",
        "model_name",
        "model_family",
        "hf_model_id",
        "gold_label",
        "single_frame_label",
        "smoothed_label",
        "vote_label",
    ]
    trailing_fields = sorted(
        {
            key
            for row in rows
            for key in row.keys()
            if key not in preferred_order
        }
    )
    fieldnames = preferred_order + trailing_fields
    write_csv_rows(path, rows, fieldnames)


def main():
    args = parse_args()
    predictions = read_csv_rows(args.predictions)
    labels = read_csv_rows(args.labels)
    merged_rows = merge_predictions_and_labels(predictions, labels)
    if not merged_rows:
        raise RuntimeError("No labeled rows were available after merging predictions and labels.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scopes = ["all"] + sorted({row["split"] for row in merged_rows})
    methods = [
        ("single_frame", "single_frame_label"),
        ("smoothed", "smoothed_label"),
        ("vote", "vote_label"),
    ]

    metrics_rows = []
    confusion_rows = []
    per_class_rows = []
    notes = []
    calibrated_prediction_rows = []

    for (model_name, model_family, model_id), model_rows in group_rows_by_model(merged_rows).items():
        for scope in scopes:
            scoped_rows = rows_for_scope(model_rows, scope)
            if not scoped_rows:
                continue
            for method_name, prediction_column in methods:
                bundle = metric_bundle(scoped_rows, prediction_column)
                metrics_rows.append(
                    {
                        "model_name": model_name,
                        "model_family": model_family,
                        "hf_model_id": model_id,
                        "scope": scope,
                        "method": method_name,
                        "n_clips": bundle["n_clips"],
                        "accuracy": bundle["accuracy"],
                        "balanced_accuracy": bundle["balanced_accuracy"],
                        "macro_f1": bundle["macro_f1"],
                    }
                )
                confusion_rows.extend(
                    matrix_to_rows(
                        model_name,
                        model_family,
                        model_id,
                        scope,
                        method_name,
                        bundle["confusion_matrix"],
                    )
                )
                per_class_rows.extend(
                    per_class_metric_rows(
                        model_name,
                        model_family,
                        model_id,
                        scope,
                        method_name,
                        scoped_rows,
                        prediction_column,
                    )
                )

        if args.fit_calibrator:
            calibrated_rows = maybe_fit_calibrator(model_rows)
            if calibrated_rows is None:
                notes.append(
                    f"Calibration skipped for {model_name}: split=dev and split=test are required with at least two classes in dev."
                )
            else:
                bundle = metric_bundle(calibrated_rows, "smoothed_calibrated")
                metrics_rows.append(
                    {
                        "model_name": model_name,
                        "model_family": model_family,
                        "hf_model_id": model_id,
                        "scope": "test",
                        "method": "smoothed_calibrated",
                        "n_clips": bundle["n_clips"],
                        "accuracy": bundle["accuracy"],
                        "balanced_accuracy": bundle["balanced_accuracy"],
                        "macro_f1": bundle["macro_f1"],
                    }
                )
                confusion_rows.extend(
                    matrix_to_rows(
                        model_name,
                        model_family,
                        model_id,
                        "test",
                        "smoothed_calibrated",
                        bundle["confusion_matrix"],
                    )
                )
                per_class_rows.extend(
                    per_class_metric_rows(
                        model_name,
                        model_family,
                        model_id,
                        "test",
                        "smoothed_calibrated",
                        calibrated_rows,
                        "smoothed_calibrated",
                    )
                )
                calibrated_prediction_rows.extend(calibrated_rows)
                notes.append(
                    f"Calibration for {model_name} used multinomial logistic regression over smoothed 3-class probabilities from split=dev."
                )

    metrics_rows = sorted(
        metrics_rows,
        key=lambda row: (row["model_name"], row["scope"], row["method"]),
    )
    per_class_rows = sorted(
        per_class_rows,
        key=lambda row: (row["model_name"], row["scope"], row["method"], row["label"]),
    )
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_rows, indent=2), encoding="utf-8")
    write_csv_rows(
        output_dir / "metrics.csv",
        metrics_rows,
        [
            "model_name",
            "model_family",
            "hf_model_id",
            "scope",
            "method",
            "n_clips",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
        ],
    )
    write_csv_rows(
        output_dir / "confusion_matrices.csv",
        confusion_rows,
        ["model_name", "model_family", "hf_model_id", "scope", "method", "gold_label"] + LABEL_ORDER,
    )
    write_csv_rows(
        output_dir / "per_class_metrics.csv",
        per_class_rows,
        [
            "model_name",
            "model_family",
            "hf_model_id",
            "scope",
            "method",
            "label",
            "precision",
            "recall",
            "f1",
            "support",
        ],
    )
    write_labeled_predictions(output_dir / "labeled_predictions.csv", merged_rows)
    if calibrated_prediction_rows:
        write_labeled_predictions(output_dir / "calibrated_test_predictions.csv", calibrated_prediction_rows)
    write_summary(output_dir, metrics_rows, notes)
    print(f"Wrote metrics to {metrics_path}")
    print(f"Wrote summary to {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
