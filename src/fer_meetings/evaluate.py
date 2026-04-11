import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)

from fer_meetings.constants import LABEL_ORDER
from fer_meetings.fusion import (
    entropy_weighted_probability_fusion,
    label_from_probabilities,
    mean_probability_fusion,
    probability_vector,
)
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


def rows_have_prediction(rows, prediction_column):
    return bool(rows) and all(str(row.get(prediction_column, "")).strip() for row in rows)


def rows_have_probabilities(rows, probability_prefix):
    if not probability_prefix:
        return False
    return bool(rows) and all(
        str(row.get(f"{probability_prefix}_{label}_prob", "")).strip()
        for row in rows
        for label in LABEL_ORDER
    )


def group_rows_by_clip(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["clip_id"], []).append(row)
    return grouped


def build_probability_ensemble_rows(rows):
    grouped = group_rows_by_clip(rows)
    ensemble_rows = []
    for clip_id, clip_rows in grouped.items():
        families = {row.get("model_family", "") for row in clip_rows}
        if not {"cnn", "vit"}.issubset(families):
            continue
        ordered_rows = sorted(
            clip_rows,
            key=lambda row: (row.get("model_family", ""), row.get("model_name", ""), row.get("hf_model_id", "")),
        )
        base_row = dict(ordered_rows[0])
        component_names = [row.get("model_name", "") for row in ordered_rows]
        component_ids = [row.get("hf_model_id", "") for row in ordered_rows if row.get("hf_model_id", "")]
        shared_metadata = {
            "model_family": "hybrid",
            "hf_model_id": " + ".join(component_ids),
            "ensemble_components": " + ".join(component_names),
        }

        for ensemble_name, fusion_fn in [
            ("cnn_vit_mean_ensemble", mean_probability_fusion),
            ("cnn_vit_entropy_ensemble", entropy_weighted_probability_fusion),
        ]:
            ensemble_row = dict(base_row)
            ensemble_row.update(shared_metadata)
            ensemble_row["model_name"] = ensemble_name
            ensemble_row["single_frame_raw_label"] = ""
            ensemble_row["vote_label"] = ""
            for probability_prefix, label_field in [
                ("single_frame", "single_frame_label"),
                ("smoothed", "smoothed_label"),
            ]:
                if not rows_have_probabilities(ordered_rows, probability_prefix):
                    continue
                fused_probabilities, weights = fusion_fn(
                    [probability_vector(row, probability_prefix) for row in ordered_rows]
                )
                ensemble_row[label_field] = label_from_probabilities(fused_probabilities)
                ensemble_row[f"{probability_prefix}_weights_json"] = json.dumps(
                    {name: float(weight) for name, weight in zip(component_names, weights)},
                    separators=(",", ":"),
                )
                for label_index, label in enumerate(LABEL_ORDER):
                    ensemble_row[f"{probability_prefix}_{label}_prob"] = f"{float(fused_probabilities[label_index]):.6f}"
            ensemble_rows.append(ensemble_row)
    return ensemble_rows


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


def one_hot_labels(rows):
    gold_indices = [LABEL_ORDER.index(row["gold_label"]) for row in rows]
    matrix = np.zeros((len(rows), len(LABEL_ORDER)), dtype=float)
    for row_index, label_index in enumerate(gold_indices):
        matrix[row_index, label_index] = 1.0
    return matrix


def probability_metric_bundle(rows, probability_prefix):
    probabilities = probability_matrix(rows, probability_prefix)
    if probabilities.size == 0:
        return {}
    y_true = one_hot_labels(rows)
    metrics = {}
    try:
        metrics["auroc_ovr"] = roc_auc_score(y_true, probabilities, average="macro", multi_class="ovr")
    except ValueError:
        metrics["auroc_ovr"] = ""
    try:
        metrics["auprc_macro"] = average_precision_score(y_true, probabilities, average="macro")
    except ValueError:
        metrics["auprc_macro"] = ""
    brier_scores = []
    for label_index in range(len(LABEL_ORDER)):
        try:
            brier_scores.append(brier_score_loss(y_true[:, label_index], probabilities[:, label_index]))
        except ValueError:
            continue
    metrics["brier_macro"] = float(np.mean(brier_scores)) if brier_scores else ""
    return metrics


def curve_rows(model_name, model_family, model_id, scope, method, rows, probability_prefix):
    probabilities = probability_matrix(rows, probability_prefix)
    y_true = one_hot_labels(rows)
    roc_rows = []
    pr_rows = []
    for label_index, label in enumerate(LABEL_ORDER):
        positives = int(y_true[:, label_index].sum())
        negatives = int((1.0 - y_true[:, label_index]).sum())
        if positives == 0 or negatives == 0:
            continue
        fpr, tpr, _ = roc_curve(y_true[:, label_index], probabilities[:, label_index])
        precision, recall, _ = precision_recall_curve(y_true[:, label_index], probabilities[:, label_index])
        for x_value, y_value in zip(fpr, tpr):
            roc_rows.append(
                {
                    "model_name": model_name,
                    "model_family": model_family,
                    "hf_model_id": model_id,
                    "scope": scope,
                    "method": method,
                    "label": label,
                    "x": float(x_value),
                    "y": float(y_value),
                }
            )
        for recall_value, precision_value in zip(recall, precision):
            pr_rows.append(
                {
                    "model_name": model_name,
                    "model_family": model_family,
                    "hf_model_id": model_id,
                    "scope": scope,
                    "method": method,
                    "label": label,
                    "x": float(recall_value),
                    "y": float(precision_value),
                }
            )
    return roc_rows, pr_rows


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
    calibrated_probabilities = model.predict_proba(probability_matrix(test_rows, "smoothed"))
    calibrated_indices = calibrated_probabilities.argmax(axis=1)

    calibrated_rows = []
    for row, predicted_index, probability_row in zip(test_rows, calibrated_indices, calibrated_probabilities):
        calibrated_row = dict(row)
        calibrated_row["smoothed_calibrated"] = LABEL_ORDER[int(predicted_index)]
        for label_index, label in enumerate(LABEL_ORDER):
            calibrated_row[f"smoothed_calibrated_{label}_prob"] = f"{float(probability_row[label_index]):.6f}"
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
        ("single_frame", "single_frame_label", "single_frame"),
        ("smoothed", "smoothed_label", "smoothed"),
        ("vote", "vote_label", ""),
    ]

    metrics_rows = []
    confusion_rows = []
    per_class_rows = []
    probability_metric_rows = []
    roc_curve_rows = []
    pr_curve_rows = []
    notes = []
    calibrated_prediction_rows = []

    for (model_name, model_family, model_id), model_rows in group_rows_by_model(merged_rows).items():
        for scope in scopes:
            scoped_rows = rows_for_scope(model_rows, scope)
            if not scoped_rows:
                continue
            for method_name, prediction_column, probability_prefix in methods:
                if not rows_have_prediction(scoped_rows, prediction_column):
                    continue
                bundle = metric_bundle(scoped_rows, prediction_column)
                probability_metrics = (
                    probability_metric_bundle(scoped_rows, probability_prefix)
                    if rows_have_probabilities(scoped_rows, probability_prefix)
                    else {}
                )
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
                        "auroc_ovr": probability_metrics.get("auroc_ovr", ""),
                        "auprc_macro": probability_metrics.get("auprc_macro", ""),
                        "brier_macro": probability_metrics.get("brier_macro", ""),
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
                if rows_have_probabilities(scoped_rows, probability_prefix):
                    method_roc_rows, method_pr_rows = curve_rows(
                        model_name,
                        model_family,
                        model_id,
                        scope,
                        method_name,
                        scoped_rows,
                        probability_prefix,
                    )
                    roc_curve_rows.extend(method_roc_rows)
                    pr_curve_rows.extend(method_pr_rows)

        if args.fit_calibrator:
            calibrated_rows = maybe_fit_calibrator(model_rows)
            if calibrated_rows is None:
                notes.append(
                    f"Calibration skipped for {model_name}: split=dev and split=test are required with at least two classes in dev."
                )
            else:
                bundle = metric_bundle(calibrated_rows, "smoothed_calibrated")
                probability_metrics = probability_metric_bundle(calibrated_rows, "smoothed_calibrated")
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
                        "auroc_ovr": probability_metrics.get("auroc_ovr", ""),
                        "auprc_macro": probability_metrics.get("auprc_macro", ""),
                        "brier_macro": probability_metrics.get("brier_macro", ""),
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
                method_roc_rows, method_pr_rows = curve_rows(
                    model_name,
                    model_family,
                    model_id,
                    "test",
                    "smoothed_calibrated",
                    calibrated_rows,
                    "smoothed_calibrated",
                )
                roc_curve_rows.extend(method_roc_rows)
                pr_curve_rows.extend(method_pr_rows)
                calibrated_prediction_rows.extend(calibrated_rows)
                notes.append(
                    f"Calibration for {model_name} used multinomial logistic regression over smoothed 3-class probabilities from split=dev."
                )

    ensemble_rows = build_probability_ensemble_rows(merged_rows)
    for (model_name, model_family, model_id), model_rows in group_rows_by_model(ensemble_rows).items():
        for scope in scopes:
            scoped_rows = rows_for_scope(model_rows, scope)
            if not scoped_rows:
                continue
            for method_name, prediction_column, probability_prefix in methods:
                if not rows_have_prediction(scoped_rows, prediction_column):
                    continue
                bundle = metric_bundle(scoped_rows, prediction_column)
                probability_metrics = (
                    probability_metric_bundle(scoped_rows, probability_prefix)
                    if rows_have_probabilities(scoped_rows, probability_prefix)
                    else {}
                )
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
                        "auroc_ovr": probability_metrics.get("auroc_ovr", ""),
                        "auprc_macro": probability_metrics.get("auprc_macro", ""),
                        "brier_macro": probability_metrics.get("brier_macro", ""),
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
                if rows_have_probabilities(scoped_rows, probability_prefix):
                    method_roc_rows, method_pr_rows = curve_rows(
                        model_name,
                        model_family,
                        model_id,
                        scope,
                        method_name,
                        scoped_rows,
                        probability_prefix,
                    )
                    roc_curve_rows.extend(method_roc_rows)
                    pr_curve_rows.extend(method_pr_rows)
    if ensemble_rows:
        notes.append(
            "Hybrid CNN+ViT ensemble rows were added with mean-probability fusion and entropy-weighted fusion over paired clip probabilities."
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
            "auroc_ovr",
            "auprc_macro",
            "brier_macro",
        ],
    )
    write_csv_rows(
        output_dir / "confusion_matrices.csv",
        confusion_rows,
        ["model_name", "model_family", "hf_model_id", "scope", "method", "gold_label"] + LABEL_ORDER,
    )
    write_csv_rows(
        output_dir / "roc_curves.csv",
        roc_curve_rows,
        ["model_name", "model_family", "hf_model_id", "scope", "method", "label", "x", "y"],
    )
    write_csv_rows(
        output_dir / "pr_curves.csv",
        pr_curve_rows,
        ["model_name", "model_family", "hf_model_id", "scope", "method", "label", "x", "y"],
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
