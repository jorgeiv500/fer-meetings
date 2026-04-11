import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from fer_meetings.labels import canonical_gold_label
from fer_meetings.utils import ensure_parent, read_csv_rows, write_csv_rows


MODEL_DISPLAY = {
    "convnext_tiny_emotion": "CNN",
    "vit_face_expression": "ViT",
    "cnn_vit_mean_ensemble": "Mean ensemble",
    "cnn_vit_entropy_ensemble": "Entropy ensemble",
    "cnn_vit_fusion": "Fusion",
}

METHOD_DISPLAY = {
    "single_frame": "Single frame",
    "smoothed": "Temporal mean",
    "smoothed_calibrated": "Calibrated mean",
    "vote": "Temporal vote",
    "attention_pooling": "Attention pooling",
    "mean_embedding_logreg": "Clip LogReg",
    "mean_embedding_hgb": "Clip HGB",
    "mean_embedding_logreg_coral": "Clip CORAL",
    "mean_embedding_mmd_adapter": "Clip MMD",
}

FAMILY_ORDER = {"cnn": 0, "vit": 1, "hybrid": 2}
METHOD_ORDER = {
    "single_frame": 0,
    "smoothed": 1,
    "smoothed_calibrated": 2,
    "vote": 3,
    "attention_pooling": 4,
    "mean_embedding_logreg": 5,
    "mean_embedding_hgb": 6,
    "mean_embedding_logreg_coral": 7,
    "mean_embedding_mmd_adapter": 8,
}
FAMILY_COLORS = {"cnn": "#c0392b", "vit": "#1f5aa6", "hybrid": "#2b8a3e"}


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


def display_model_name(name):
    return MODEL_DISPLAY.get(name, name.replace("_", " ").title())


def display_method_name(name):
    return METHOD_DISPLAY.get(name, name.replace("_", " ").title())


def display_combo(model_name, method_name):
    return f"{display_model_name(model_name)} | {display_method_name(method_name)}"


def prepare_output_dir(output_dir):
    for child in ("tables", "figures"):
        child_path = Path(output_dir) / child
        if child_path.exists():
            shutil.rmtree(child_path)
        child_path.mkdir(parents=True, exist_ok=True)


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


def build_interrater_table(summary_rows):
    summary = {row["metric"]: row.get("value", "") for row in summary_rows}
    observed = float(summary.get("observed_agreement", 0.0) or 0.0)
    kappa = float(summary.get("cohen_kappa", 0.0) or 0.0)
    double_rated = int(float(summary.get("double_rated_clips", 0) or 0))
    agree = int(round(double_rated * observed))
    disagree = max(0, double_rated - agree)
    return [
        {"metric": "double_rated_clips", "value": double_rated, "notes": "Clips with two independent human ratings."},
        {"metric": "agree", "value": agree, "notes": "Exact matches between human 1 and human 2."},
        {"metric": "disagree", "value": disagree, "notes": "Clips that still require adjudication if a single final label is needed."},
        {"metric": "observed_agreement", "value": format_metric(observed), "notes": "Fraction of exact matches across the 100 double-rated clips."},
        {"metric": "cohen_kappa", "value": format_metric(kappa), "notes": "Chance-corrected agreement across the three valence classes."},
    ]


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


def build_clip_confusion_tables(confusion_rows):
    tables = {}
    for row in confusion_rows:
        key = (row["model_name"], row["method"])
        tables.setdefault(key, []).append(row)
    return tables


def select_test_main_rows(metrics_rows):
    rows = [row for row in metrics_rows if row.get("scope") == "test"]
    return sorted(
        rows,
        key=lambda row: (
            FAMILY_ORDER.get(row.get("model_family", ""), 99),
            display_model_name(row.get("model_name", "")),
            METHOD_ORDER.get(row.get("method", ""), 99),
        ),
    )


def select_clip_rows(metrics_rows):
    return sorted(
        metrics_rows,
        key=lambda row: (
            FAMILY_ORDER.get(row.get("model_family", ""), 99),
            display_model_name(row.get("model_name", "")),
            METHOD_ORDER.get(row.get("method", ""), 99),
        ),
    )


def plot_ranked_bars(rows, output_path, title, metric_field="macro_f1"):
    import matplotlib.pyplot as plt

    if not rows:
        return

    ranked = sorted(rows, key=lambda row: float(row[metric_field]), reverse=True)
    labels = [display_combo(row["model"], row["method"]) for row in ranked]
    values = [float(row[metric_field]) for row in ranked]
    colors = [FAMILY_COLORS.get(row["family"], "#495057") for row in ranked]

    plt.figure(figsize=(10, max(4.5, len(labels) * 0.48)))
    bars = plt.barh(labels, values, color=colors, alpha=0.92)
    xlabel = metric_field.replace("_", " ").title()
    plt.xlabel(xlabel)
    plt.title(title)
    plt.xlim(0.0, 1.0)
    plt.grid(axis="x", linestyle="--", alpha=0.25)
    plt.gca().invert_yaxis()
    for bar, value in zip(bars, values):
        plt.text(min(0.98, value + 0.015), bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    ensure_parent(output_path)
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_metric_heatmap(rows, output_path, title):
    import matplotlib.pyplot as plt
    import numpy as np

    if not rows:
        return

    labels = [display_combo(row["model"], row["method"]) for row in rows]
    metrics = ["macro_f1", "balanced_accuracy", "accuracy"]
    metric_labels = ["Macro-F1", "Bal. Acc.", "Accuracy"]
    values = np.array([[float(row[field]) for field in metrics] for row in rows], dtype=float)

    fig_height = max(4.4, len(labels) * 0.42)
    plt.figure(figsize=(7.2, fig_height))
    image = plt.imshow(values, cmap="YlGnBu", aspect="auto", vmin=0.0, vmax=1.0)
    plt.xticks(range(len(metric_labels)), metric_labels)
    plt.yticks(range(len(labels)), labels)
    plt.title(title)
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            plt.text(x, y, f"{values[y, x]:.3f}", ha="center", va="center", color="black", fontsize=9)
    plt.colorbar(image, fraction=0.03, pad=0.02)
    plt.tight_layout()
    ensure_parent(output_path)
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_clip_lollipop(rows, output_path, title, metric_field="macro_f1"):
    import matplotlib.pyplot as plt

    if not rows:
        return

    ranked = sorted(rows, key=lambda row: float(row[metric_field]), reverse=True)
    labels = [display_combo(row["model"], row["method"]) for row in ranked]
    values = [float(row[metric_field]) for row in ranked]
    colors = [FAMILY_COLORS.get(row["family"], "#495057") for row in ranked]
    positions = list(range(len(labels)))

    plt.figure(figsize=(10, max(4.8, len(labels) * 0.46)))
    for position, value, color in zip(positions, values, colors):
        plt.hlines(position, 0.0, value, color=color, linewidth=2.2, alpha=0.9)
        plt.scatter(value, position, color=color, s=75, zorder=3)
        plt.text(min(0.98, value + 0.015), position, f"{value:.3f}", va="center", fontsize=9)
    plt.yticks(positions, labels)
    plt.xlabel(metric_field.replace("_", " ").title())
    plt.title(title)
    plt.xlim(0.0, 1.0)
    plt.grid(axis="x", linestyle="--", alpha=0.25)
    plt.gca().invert_yaxis()
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
    totals = negatives + neutrals + positives
    for index, total in enumerate(totals):
        plt.text(index, total + 0.6, f"n={int(total)}", ha="center", va="bottom", fontsize=9)
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


def plot_interrater_overview(summary_rows, output_path):
    import matplotlib.pyplot as plt

    if not summary_rows:
        return

    summary = {row["metric"]: row.get("value", "") for row in summary_rows}
    double_rated = int(float(summary.get("double_rated_clips", 0) or 0))
    observed = float(summary.get("observed_agreement", 0.0) or 0.0)
    kappa = float(summary.get("cohen_kappa", 0.0) or 0.0)
    agree = int(round(double_rated * observed))
    disagree = max(0, double_rated - agree)

    figure, axes = plt.subplots(1, 2, figsize=(9.0, 4.2))
    axes[0].bar(["Agree", "Disagree"], [agree, disagree], color=["#2f9e44", "#d94841"])
    axes[0].set_ylim(0, max(double_rated, 1))
    axes[0].set_ylabel("Clips")
    axes[0].set_title("Human agreement counts")
    for index, value in enumerate([agree, disagree]):
        axes[0].text(index, value + 1, str(value), ha="center", fontsize=10)

    axes[1].bar(["Observed agreement", "Cohen's kappa"], [observed, kappa], color=["#1f5aa6", "#6c757d"])
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Agreement strength")
    for index, value in enumerate([observed, kappa]):
        axes[1].text(index, value + 0.03, f"{value:.3f}", ha="center", fontsize=10)

    plt.tight_layout()
    ensure_parent(output_path)
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_confusion_panel(selected_rows, output_path, title):
    import matplotlib.pyplot as plt
    import numpy as np

    if not selected_rows:
        return

    labels = ["negative", "neutral", "positive"]
    figure, axes = plt.subplots(2, 2, figsize=(9.2, 8.4))
    axes = axes.flatten()

    for axis, panel in zip(axes, selected_rows):
        rows = panel["rows"]
        matrix = np.array([[int(row[label]) for label in labels] for row in rows], dtype=float)
        axis.imshow(matrix, cmap="Blues", vmin=0.0)
        axis.set_xticks(range(len(labels)), labels)
        axis.set_yticks(range(len(labels)), labels)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Gold")
        axis.set_title(panel["title"], fontsize=10)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                axis.text(j, i, int(matrix[i, j]), ha="center", va="center", color="black", fontsize=9)

    for axis in axes[len(selected_rows):]:
        axis.axis("off")

    figure.suptitle(title, fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    ensure_parent(output_path)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def maybe_generate_figures(main_rows, clip_rows, confusion_rows, clip_confusion_rows, interrater_rows, output_dir, label_distribution_rows=None):
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

    plot_ranked_bars(
        main_rows,
        Path(output_dir) / "figures" / "main_test_macro_f1.png",
        "Main test-set ranking by Macro-F1",
        metric_field="macro_f1",
    )
    plot_metric_heatmap(
        main_rows,
        Path(output_dir) / "figures" / "main_test_scorecard.png",
        "Test-set scorecard across main methods",
    )

    if clip_rows:
        plot_clip_lollipop(
            clip_rows,
            Path(output_dir) / "figures" / "clip_models_macro_f1.png",
            "Clip-level adaptation ranking by Macro-F1",
        )

    if label_distribution_rows:
        plot_label_distribution(
            label_distribution_rows,
            Path(output_dir) / "figures" / "label_distribution.png",
        )

    if interrater_rows:
        plot_interrater_overview(
            interrater_rows,
            Path(output_dir) / "figures" / "interrater_overview.png",
        )

    main_lookup = build_confusion_tables(confusion_rows)
    clip_lookup = build_clip_confusion_tables(clip_confusion_rows)
    selected_confusions = []

    best_by_family = {}
    for row in main_rows:
        family = row["family"]
        existing = best_by_family.get(family)
        if existing is None or float(row["macro_f1"]) > float(existing["macro_f1"]):
            best_by_family[family] = row
    for family in ("cnn", "vit", "hybrid"):
        row = best_by_family.get(family)
        if not row:
            continue
        key = (row["model"], row["method"], "test")
        if key not in main_lookup:
            continue
        selected_confusions.append(
            {
                "title": display_combo(row["model"], row["method"]),
                "rows": main_lookup[key],
            }
        )
    if clip_rows:
        best_clip = max(clip_rows, key=lambda row: float(row["macro_f1"]))
        key = (best_clip["model"], best_clip["method"])
        if key in clip_lookup:
            selected_confusions.append(
                {
                    "title": display_combo(best_clip["model"], best_clip["method"]),
                    "rows": clip_lookup[key],
                }
            )
    plot_confusion_panel(
        selected_confusions[:4],
        Path(output_dir) / "figures" / "selected_confusions.png",
        "Confusion matrices for the strongest representative models",
    )


def main():
    args = parse_args()
    pilot_dir = Path(args.pilot_dir)
    clip_model_dir = Path(args.clip_model_dir) if args.clip_model_dir else None
    output_dir = Path(args.output_dir)
    prepare_output_dir(output_dir)
    manifest_rows = read_csv_rows(args.manifest) if args.manifest else []
    raw_label_rows = read_csv_rows(args.labels) if args.labels else []
    gold_label_rows = labeled_rows(raw_label_rows)

    main_metrics = load_main_metrics(pilot_dir)
    main_table = build_main_comparison_table(select_test_main_rows(main_metrics))
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

    confusion_path = pilot_dir / "confusion_matrices.csv"
    confusion_rows = read_csv_rows(confusion_path) if confusion_path.exists() else []
    interrater_path = pilot_dir / "interrater_agreement.csv"
    interrater_rows = read_csv_rows(interrater_path) if interrater_path.exists() else []

    clip_table = []
    clip_confusion_rows = []
    if clip_model_dir:
        clip_metrics = select_clip_rows(load_clip_metrics(clip_model_dir))
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
        clip_confusion_path = clip_model_dir / "clip_model_confusion_matrices.csv"
        if clip_confusion_path.exists():
            clip_confusion_rows = read_csv_rows(clip_confusion_path)

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
        write_markdown_table(
            output_dir / "tables" / "label_distribution.md",
            label_distribution_rows,
            ["scope", "n_labeled", "negative", "neutral", "positive"],
        )
    if interrater_rows:
        interrater_table = build_interrater_table(interrater_rows)
        write_csv_rows(
            output_dir / "tables" / "interrater_summary.csv",
            interrater_table,
            ["metric", "value", "notes"],
        )
        write_markdown_table(
            output_dir / "tables" / "interrater_summary.md",
            interrater_table,
            ["metric", "value", "notes"],
        )

    notes_path = output_dir / "tables" / "asset_manifest.txt"
    ensure_parent(notes_path)
    notes_path.write_text(
        "\n".join(
            [
                "Generated paper assets:",
                "- tables/main_model_comparison.csv",
                "- tables/main_model_comparison.md",
                "- tables/clip_model_comparison.csv (if clip-model metrics exist)",
                "- tables/clip_model_comparison.md (if clip-model metrics exist)",
                "- tables/dataset_summary.csv (if manifest is provided)",
                "- tables/dataset_summary.md (if manifest is provided)",
                "- tables/label_distribution.csv (if labels are provided)",
                "- tables/label_distribution.md (if labels are provided)",
                "- tables/interrater_summary.csv (if interrater summary exists)",
                "- tables/interrater_summary.md (if interrater summary exists)",
                "- figures/main_test_macro_f1.png",
                "- figures/main_test_scorecard.png",
                "- figures/clip_models_macro_f1.png (if clip-model metrics exist)",
                "- figures/label_distribution.png (if labels are provided)",
                "- figures/interrater_overview.png (if interrater summary exists)",
                "- figures/selected_confusions.png",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    maybe_generate_figures(main_table, clip_table, confusion_rows, clip_confusion_rows, interrater_rows, output_dir, label_distribution_rows)
    print(f"Wrote paper assets under {output_dir}")


if __name__ == "__main__":
    main()
