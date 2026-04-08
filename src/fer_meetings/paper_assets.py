import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from fer_meetings.labels import canonical_gold_label
from fer_meetings.utils import ensure_parent, read_csv_rows, write_csv_rows


def parse_args():
    parser = argparse.ArgumentParser(description="Generate paper-ready tables and figures from pilot outputs.")
    parser.add_argument("--pilot-dir", required=True, help="Directory containing fer-evaluate outputs.")
    parser.add_argument(
        "--clip-model-dir",
        default="",
        help="Optional directory containing fer-train-clip-models outputs.",
    )
    parser.add_argument("--manifest", default="", help="Optional manifest CSV for dataset summary tables.")
    parser.add_argument("--labels", default="", help="Optional labels CSV for label distribution tables.")
    parser.add_argument("--output-dir", required=True, help="Directory where tables and figures are written.")
    return parser.parse_args()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_markdown_table(path, rows, fieldnames):
    ensure_parent(path)
    lines = [
        "| " + " | ".join(fieldnames) + " |",
        "| " + " | ".join(["---"] * len(fieldnames)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fieldnames) + " |")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_metric(value):
    return f"{float(value):.4f}"


def load_main_metrics(pilot_dir):
    metrics_path = Path(pilot_dir) / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Could not find main metrics file: {metrics_path}")
    return read_json(metrics_path)


def load_clip_metrics(clip_model_dir):
    metrics_path = Path(clip_model_dir) / "clip_model_metrics.json"
    if not metrics_path.exists():
        return []
    return read_json(metrics_path)


def labeled_rows(label_rows):
    rows = []
    for row in label_rows:
        gold_label = canonical_gold_label(row.get("gold_label", ""))
        if not gold_label:
            continue
        merged = dict(row)
        merged["gold_label"] = gold_label
        rows.append(merged)
    return rows


def build_main_comparison_table(metrics_rows):
    rows = []
    for row in metrics_rows:
        rows.append(
            {
                "model": row["model_name"],
                "family": row["model_family"],
                "scope": row["scope"],
                "method": row["method"],
                "n_clips": row["n_clips"],
                "accuracy": format_metric(row["accuracy"]),
                "balanced_accuracy": format_metric(row["balanced_accuracy"]),
                "macro_f1": format_metric(row["macro_f1"]),
                "auroc_ovr": format_metric(row["auroc_ovr"]) if row.get("auroc_ovr", "") != "" else "",
                "auprc_macro": format_metric(row["auprc_macro"]) if row.get("auprc_macro", "") != "" else "",
                "brier_macro": format_metric(row["brier_macro"]) if row.get("brier_macro", "") != "" else "",
            }
        )
    return rows


def build_clip_comparison_table(metrics_rows):
    rows = []
    for row in metrics_rows:
        rows.append(
            {
                "model": row["model_name"],
                "family": row["model_family"],
                "method": row["method"],
                "n_clips": row["n_clips"],
                "accuracy": format_metric(row["accuracy"]),
                "balanced_accuracy": format_metric(row["balanced_accuracy"]),
                "macro_f1": format_metric(row["macro_f1"]),
            }
        )
    return rows


def curve_table_rows(curve_rows):
    rows = []
    for row in curve_rows:
        rows.append(
            {
                "model": row["model_name"],
                "family": row["model_family"],
                "scope": row["scope"],
                "method": row["method"],
                "label": row["label"],
                "x": format_metric(row["x"]),
                "y": format_metric(row["y"]),
            }
        )
    return rows


def history_table_rows(history_rows):
    rows = []
    for row in history_rows:
        rows.append(
            {
                "model": row["model_name"],
                "family": row["model_family"],
                "method": row["method"],
                "epoch": row["epoch"],
                "train_loss": format_metric(row["train_loss"]),
                "train_accuracy": format_metric(row["train_accuracy"]),
                "val_macro_f1": format_metric(row["val_macro_f1"]) if row.get("val_macro_f1", "") != "" else "",
                "val_accuracy": format_metric(row["val_accuracy"]) if row.get("val_accuracy", "") != "" else "",
            }
        )
    return rows


def build_dataset_summary_table(manifest_rows, label_rows):
    label_index = {row["clip_id"]: row for row in label_rows}
    scopes = sorted({row["split"] for row in manifest_rows})
    rows = []
    for scope in scopes + ["all"]:
        scoped_manifest = manifest_rows if scope == "all" else [row for row in manifest_rows if row["split"] == scope]
        scoped_labels = [
            label_index[row["clip_id"]]
            for row in scoped_manifest
            if row["clip_id"] in label_index
        ]
        rows.append(
            {
                "scope": scope,
                "manifest_clips": len(scoped_manifest),
                "labeled_clips": len(scoped_labels),
                "meetings": len({row.get("meeting_id", "") for row in scoped_manifest if row.get("meeting_id", "")}),
                "cameras": len({row.get("camera", "") for row in scoped_manifest if row.get("camera", "")}),
            }
        )
    return rows


def build_label_distribution_table(label_rows):
    rows = []
    scopes = sorted({row["split"] for row in label_rows})
    for scope in scopes + ["all"]:
        scoped_rows = label_rows if scope == "all" else [row for row in label_rows if row["split"] == scope]
        counts = Counter(row["gold_label"] for row in scoped_rows)
        rows.append(
            {
                "scope": scope,
                "n_labeled": len(scoped_rows),
                "negative": counts.get("negative", 0),
                "neutral": counts.get("neutral", 0),
                "positive": counts.get("positive", 0),
            }
        )
    return rows


def build_per_class_table(rows, scope_field=True):
    table_rows = []
    for row in rows:
        output = {
            "model": row["model_name"],
            "family": row["model_family"],
        }
        if scope_field and "scope" in row:
            output["scope"] = row["scope"]
        output["method"] = row["method"]
        output["label"] = row["label"]
        output["precision"] = format_metric(row["precision"])
        output["recall"] = format_metric(row["recall"])
        output["f1"] = format_metric(row["f1"])
        output["support"] = row["support"]
        table_rows.append(output)
    return table_rows


def build_confusion_tables(confusion_rows):
    tables = {}
    for row in confusion_rows:
        key = (row["model_name"], row["method"], row["scope"])
        tables.setdefault(key, []).append(row)
    return tables


def plot_method_bars(rows, output_path, title, metric_field="macro_f1", scope_filter=None):
    import matplotlib.pyplot as plt

    filtered = [
        row for row in rows
        if scope_filter is None or row.get("scope") == scope_filter
    ]
    if not filtered:
        return

    labels = [
        f"{row.get('model_name', row.get('model', 'model'))}\n{row['method']}"
        for row in filtered
    ]
    values = [float(row[metric_field]) for row in filtered]

    plt.figure(figsize=(max(8, len(labels) * 1.1), 4.8))
    plt.bar(labels, values, color="#4C6EF5")
    plt.ylim(0.0, 1.0)
    ylabel = metric_field.replace("_", " ").title()
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    ensure_parent(output_path)
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_label_distribution(rows, output_path):
    import matplotlib.pyplot as plt
    import numpy as np

    filtered = [row for row in rows if row["scope"] != "all"]
    if not filtered:
        return

    scopes = [row["scope"] for row in filtered]
    negatives = np.array([int(row["negative"]) for row in filtered], dtype=float)
    neutrals = np.array([int(row["neutral"]) for row in filtered], dtype=float)
    positives = np.array([int(row["positive"]) for row in filtered], dtype=float)
    x = np.arange(len(scopes))

    plt.figure(figsize=(max(6, len(scopes) * 1.4), 4.6))
    plt.bar(x, negatives, label="negative", color="#d94841")
    plt.bar(x, neutrals, bottom=negatives, label="neutral", color="#868e96")
    plt.bar(x, positives, bottom=negatives + neutrals, label="positive", color="#2f9e44")
    plt.xticks(x, scopes)
    plt.ylabel("Labeled clips")
    plt.title("Gold-label distribution by split")
    plt.legend()
    plt.tight_layout()
    ensure_parent(output_path)
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_confusion_heatmap(rows, output_path, title):
    import matplotlib.pyplot as plt
    import numpy as np

    labels = ["negative", "neutral", "positive"]
    matrix = np.array([[int(row[label]) for label in labels] for row in rows], dtype=float)

    plt.figure(figsize=(4.8, 4.2))
    plt.imshow(matrix, cmap="Blues")
    plt.xticks(range(len(labels)), labels)
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Predicted")
    plt.ylabel("Gold")
    plt.title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, int(matrix[i, j]), ha="center", va="center", color="black")
    plt.tight_layout()
    ensure_parent(output_path)
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_curve_families(rows, output_path, title, x_label, y_label):
    import matplotlib.pyplot as plt

    if not rows:
        return
    plt.figure(figsize=(7.2, 5.0))
    families = {"cnn": "#c0392b", "vit": "#1f5aa6"}
    methods = {"single_frame": "-", "smoothed": "--", "smoothed_calibrated": ":", "attention_pooling": "-."}
    grouped = defaultdict(list)
    for row in rows:
        key = (row["model_family"], row["model_name"], row["method"], row["label"])
        grouped[key].append(row)
    for key, series in grouped.items():
        family, model_name, method, label = key
        series = sorted(series, key=lambda item: float(item["x"]))
        x_values = [float(item["x"]) for item in series]
        y_values = [float(item["y"]) for item in series]
        plt.plot(
            x_values,
            y_values,
            label=f"{model_name} | {method} | {label}",
            color=families.get(family, "#495057"),
            linestyle=methods.get(method, "-"),
            linewidth=1.6,
            alpha=0.9,
        )
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(fontsize=8, ncol=1, loc="best")
    plt.tight_layout()
    ensure_parent(output_path)
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_attention_history(rows, output_path):
    import matplotlib.pyplot as plt

    if not rows:
        return
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.get("model_name", row.get("model", "model"))].append(row)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    colors = {"cnn": "#c0392b", "vit": "#1f5aa6"}
    for model_name, series in grouped.items():
        series = sorted(series, key=lambda item: int(item["epoch"]))
        family = series[0].get("model_family", series[0].get("family", "unknown"))
        epochs = [int(item["epoch"]) for item in series]
        train_loss = [float(item["train_loss"]) for item in series]
        train_accuracy = [float(item["train_accuracy"]) for item in series]
        val_accuracy = [float(item["val_accuracy"]) for item in series if item.get("val_accuracy", "") != ""]
        val_epochs = [int(item["epoch"]) for item in series if item.get("val_accuracy", "") != ""]
        axes[0].plot(epochs, train_loss, label=model_name, color=colors.get(family, "#495057"))
        axes[1].plot(epochs, train_accuracy, label=f"{model_name} train", color=colors.get(family, "#495057"))
        if val_accuracy:
            axes[1].plot(val_epochs, val_accuracy, label=f"{model_name} val", color=colors.get(family, "#495057"), linestyle="--")
    axes[0].set_title("Attention Pooling Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[1].set_title("Attention Pooling Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    for axis in axes:
        axis.legend(fontsize=8)
    plt.tight_layout()
    ensure_parent(output_path)
    plt.savefig(output_path, dpi=180)
    plt.close()


def maybe_generate_figures(main_rows, clip_rows, confusion_rows, output_dir, label_distribution_rows=None, roc_rows=None, pr_rows=None, history_rows=None):
    try:
        import matplotlib  # noqa: F401
    except Exception:
        note_path = Path(output_dir) / "figures" / "README.txt"
        ensure_parent(note_path)
        note_path.write_text(
            "matplotlib is not available in the current environment. Install dependencies and rerun fer-make-paper-assets.\n",
            encoding="utf-8",
        )
        return

    plot_method_bars(
        main_rows,
        Path(output_dir) / "figures" / "main_test_macro_f1.png",
        "Zero-shot and temporally pooled performance on test clips",
        metric_field="macro_f1",
        scope_filter="test",
    )
    plot_method_bars(
        main_rows,
        Path(output_dir) / "figures" / "main_test_balanced_accuracy.png",
        "Balanced accuracy on test clips",
        metric_field="balanced_accuracy",
        scope_filter="test",
    )
    plot_method_bars(
        main_rows,
        Path(output_dir) / "figures" / "main_test_accuracy.png",
        "Accuracy on test clips",
        metric_field="accuracy",
        scope_filter="test",
    )

    if clip_rows:
        plot_method_bars(
            clip_rows,
            Path(output_dir) / "figures" / "clip_models_macro_f1.png",
            "Clip-level representation models on test clips",
        )

    if label_distribution_rows:
        plot_label_distribution(
            label_distribution_rows,
            Path(output_dir) / "figures" / "label_distribution.png",
        )

    if roc_rows:
        plot_curve_families(
            [row for row in roc_rows if row["scope"] == "test" and row["method"] in {"single_frame", "smoothed", "smoothed_calibrated"}],
            Path(output_dir) / "figures" / "main_test_roc_ovr.png",
            "One-vs-rest ROC curves on test clips",
            "False positive rate",
            "True positive rate",
        )
    if pr_rows:
        plot_curve_families(
            [row for row in pr_rows if row["scope"] == "test" and row["method"] in {"single_frame", "smoothed", "smoothed_calibrated"}],
            Path(output_dir) / "figures" / "main_test_pr_ovr.png",
            "One-vs-rest Precision-Recall curves on test clips",
            "Recall",
            "Precision",
        )
    if history_rows:
        plot_attention_history(
            history_rows,
            Path(output_dir) / "figures" / "attention_pooling_training.png",
        )

    tables = build_confusion_tables(confusion_rows)
    for (model_name, method, scope), rows in tables.items():
        if scope != "test":
            continue
        safe_name = f"{model_name}_{method}_{scope}".replace("/", "_").replace(" ", "_")
        plot_confusion_heatmap(
            rows,
            Path(output_dir) / "figures" / f"confusion_{safe_name}.png",
            f"{model_name} | {method} | {scope}",
        )


def main():
    args = parse_args()
    pilot_dir = Path(args.pilot_dir)
    clip_model_dir = Path(args.clip_model_dir) if args.clip_model_dir else None
    output_dir = Path(args.output_dir)
    manifest_rows = read_csv_rows(args.manifest) if args.manifest else []
    raw_label_rows = read_csv_rows(args.labels) if args.labels else []
    gold_label_rows = labeled_rows(raw_label_rows)

    main_metrics = load_main_metrics(pilot_dir)
    main_table = build_main_comparison_table(main_metrics)
    write_csv_rows(
        output_dir / "tables" / "main_model_comparison.csv",
        main_table,
        ["model", "family", "scope", "method", "n_clips", "accuracy", "balanced_accuracy", "macro_f1", "auroc_ovr", "auprc_macro", "brier_macro"],
    )
    write_markdown_table(
        output_dir / "tables" / "main_model_comparison.md",
        main_table,
        ["model", "family", "scope", "method", "n_clips", "accuracy", "balanced_accuracy", "macro_f1", "auroc_ovr", "auprc_macro", "brier_macro"],
    )

    per_class_path = pilot_dir / "per_class_metrics.csv"
    if per_class_path.exists():
        main_per_class_rows = build_per_class_table(read_csv_rows(per_class_path), scope_field=True)
        write_csv_rows(
            output_dir / "tables" / "main_per_class_metrics.csv",
            main_per_class_rows,
            ["model", "family", "scope", "method", "label", "precision", "recall", "f1", "support"],
        )
        write_markdown_table(
            output_dir / "tables" / "main_per_class_metrics.md",
            main_per_class_rows,
            ["model", "family", "scope", "method", "label", "precision", "recall", "f1", "support"],
        )

    confusion_path = pilot_dir / "confusion_matrices.csv"
    confusion_rows = read_csv_rows(confusion_path) if confusion_path.exists() else []
    roc_path = pilot_dir / "roc_curves.csv"
    roc_rows = read_csv_rows(roc_path) if roc_path.exists() else []
    pr_path = pilot_dir / "pr_curves.csv"
    pr_rows = read_csv_rows(pr_path) if pr_path.exists() else []

    clip_table = []
    history_rows = []
    if clip_model_dir:
        clip_metrics = load_clip_metrics(clip_model_dir)
        clip_table = build_clip_comparison_table(clip_metrics)
        if clip_table:
            write_csv_rows(
                output_dir / "tables" / "clip_model_comparison.csv",
                clip_table,
                ["model", "family", "method", "n_clips", "accuracy", "balanced_accuracy", "macro_f1"],
            )
            write_markdown_table(
                output_dir / "tables" / "clip_model_comparison.md",
                clip_table,
                ["model", "family", "method", "n_clips", "accuracy", "balanced_accuracy", "macro_f1"],
            )

        clip_per_class_path = clip_model_dir / "clip_model_per_class_metrics.csv"
        if clip_per_class_path.exists():
            clip_per_class_rows = build_per_class_table(read_csv_rows(clip_per_class_path), scope_field=False)
            write_csv_rows(
                output_dir / "tables" / "clip_model_per_class_metrics.csv",
                clip_per_class_rows,
                ["model", "family", "method", "label", "precision", "recall", "f1", "support"],
            )
        history_path = clip_model_dir / "attention_pooling_history.csv"
        if history_path.exists():
            history_rows = history_table_rows(read_csv_rows(history_path))
            write_csv_rows(
                output_dir / "tables" / "attention_pooling_history.csv",
                history_rows,
                ["model", "family", "method", "epoch", "train_loss", "train_accuracy", "val_macro_f1", "val_accuracy"],
            )
            write_markdown_table(
                output_dir / "tables" / "attention_pooling_history.md",
                history_rows,
                ["model", "family", "method", "epoch", "train_loss", "train_accuracy", "val_macro_f1", "val_accuracy"],
            )
            write_markdown_table(
                output_dir / "tables" / "clip_model_per_class_metrics.md",
                clip_per_class_rows,
                ["model", "family", "method", "label", "precision", "recall", "f1", "support"],
            )

    label_distribution_rows = []
    if manifest_rows:
        dataset_summary = build_dataset_summary_table(manifest_rows, gold_label_rows)
        write_csv_rows(
            output_dir / "tables" / "dataset_summary.csv",
            dataset_summary,
            ["scope", "manifest_clips", "labeled_clips", "meetings", "cameras"],
        )
        write_markdown_table(
            output_dir / "tables" / "dataset_summary.md",
            dataset_summary,
            ["scope", "manifest_clips", "labeled_clips", "meetings", "cameras"],
        )
    if gold_label_rows:
        label_distribution_rows = build_label_distribution_table(gold_label_rows)
        write_csv_rows(
            output_dir / "tables" / "label_distribution.csv",
            label_distribution_rows,
            ["scope", "n_labeled", "negative", "neutral", "positive"],
        )

    if roc_rows:
        roc_table = curve_table_rows(roc_rows)
        write_csv_rows(
            output_dir / "tables" / "roc_curves.csv",
            roc_table,
            ["model", "family", "scope", "method", "label", "x", "y"],
        )
    if pr_rows:
        pr_table = curve_table_rows(pr_rows)
        write_csv_rows(
            output_dir / "tables" / "pr_curves.csv",
            pr_table,
            ["model", "family", "scope", "method", "label", "x", "y"],
        )
        write_markdown_table(
            output_dir / "tables" / "label_distribution.md",
            label_distribution_rows,
            ["scope", "n_labeled", "negative", "neutral", "positive"],
        )

    notes_path = output_dir / "tables" / "asset_manifest.txt"
    ensure_parent(notes_path)
    notes_path.write_text(
        "\n".join(
            [
                "Generated paper assets:",
                "- tables/main_model_comparison.csv",
                "- tables/main_model_comparison.md",
                "- tables/roc_curves.csv (if probability curves exist)",
                "- tables/pr_curves.csv (if probability curves exist)",
                "- tables/main_per_class_metrics.csv (if per-class metrics exist)",
                "- tables/main_per_class_metrics.md (if per-class metrics exist)",
                "- tables/clip_model_comparison.csv (if clip-model metrics exist)",
                "- tables/clip_model_comparison.md (if clip-model metrics exist)",
                "- tables/clip_model_per_class_metrics.csv (if clip-model per-class metrics exist)",
                "- tables/clip_model_per_class_metrics.md (if clip-model per-class metrics exist)",
                "- tables/attention_pooling_history.csv (if attention pooling history exists)",
                "- tables/attention_pooling_history.md (if attention pooling history exists)",
                "- tables/dataset_summary.csv (if manifest is provided)",
                "- tables/dataset_summary.md (if manifest is provided)",
                "- tables/label_distribution.csv (if labels are provided)",
                "- tables/label_distribution.md (if labels are provided)",
                "- figures/main_test_macro_f1.png",
                "- figures/main_test_accuracy.png",
                "- figures/main_test_balanced_accuracy.png",
                "- figures/main_test_roc_ovr.png (if probability curves exist)",
                "- figures/main_test_pr_ovr.png (if probability curves exist)",
                "- figures/clip_models_macro_f1.png (if clip-model metrics exist)",
                "- figures/attention_pooling_training.png (if training history exists)",
                "- figures/label_distribution.png (if labels are provided)",
                "- figures/confusion_*.png",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    maybe_generate_figures(main_metrics, clip_table, confusion_rows, output_dir, label_distribution_rows, roc_rows, pr_rows, history_rows)
    print(f"Wrote paper assets under {output_dir}")


if __name__ == "__main__":
    main()
