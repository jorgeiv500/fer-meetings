import argparse
import json
from collections import Counter
from pathlib import Path

from fer_meetings.config import load_config, resolve_model_specs
from fer_meetings.labels import resolve_gold_label
from fer_meetings.utils import read_csv_rows


def parse_args():
    parser = argparse.ArgumentParser(description="Generate publication-facing experiment reports and registry files.")
    parser.add_argument("--config", required=True, help="Experiment config JSON.")
    parser.add_argument("--manifest", required=True, help="Manifest CSV.")
    parser.add_argument("--labels", required=True, help="Resolved clip labels CSV.")
    parser.add_argument("--pilot-dir", required=True, help="Directory with evaluation outputs.")
    parser.add_argument("--clip-model-dir", default="", help="Optional directory with clip-model outputs.")
    parser.add_argument("--output-dir", required=True, help="Directory for publication reports.")
    return parser.parse_args()


def dump_yaml_models(model_specs):
    lines = ["models:"]
    for spec in model_specs:
        lines.extend(
            [
                f"  - name: {spec['name']}",
                f"    hf_model_id: {spec['hf_model_id']}",
                f"    family: {spec['family']}",
                f"    face_detector: {spec['face_detector']}",
            ]
        )
    return "\n".join(lines) + "\n"


def labeled_rows(rows):
    labeled = []
    for row in rows:
        label, source = resolve_gold_label(row)
        if not label:
            continue
        merged = dict(row)
        merged["gold_label"] = label
        merged["label_source"] = source
        labeled.append(merged)
    return labeled


def load_metrics(path):
    if not Path(path).exists():
        return []
    if Path(path).suffix == ".json":
        return json.loads(Path(path).read_text(encoding="utf-8"))
    return read_csv_rows(path)


def metric_lookup(rows):
    return {row.get("metric", ""): row.get("value", "") for row in rows}


def build_experiment_card(config, model_specs, manifest_rows, label_rows, metrics_rows, clip_metrics_rows, interrater_rows):
    label_counts = Counter(row["gold_label"] for row in label_rows)
    lines = [
        "# Experiment Card",
        "",
        f"- Experiment name: `{config.get('experiment_name', 'unnamed_experiment')}`",
        f"- Manifest clips: `{len(manifest_rows)}`",
        f"- Resolved labeled clips: `{len(label_rows)}`",
        f"- Meetings: `{len({row.get('meeting_id', '') for row in manifest_rows if row.get('meeting_id', '')})}`",
        f"- Source backbones: `{', '.join(spec['name'] for spec in model_specs)}`",
        f"- Label distribution: `negative={label_counts.get('negative', 0)}`, `neutral={label_counts.get('neutral', 0)}`, `positive={label_counts.get('positive', 0)}`",
        "",
        "## Experimental Focus",
        "",
        "- Cross-domain transfer from FER2013-style backbones to AMI meeting video.",
        "- Comparison between CNN and Vision Transformer representations.",
        "- Temporal aggregation and clip-level adaptation under weak supervision.",
        "- Hybrid CNN+ViT fusion through probability ensembles and clip-level representation fusion.",
    ]

    if interrater_rows:
        lookup = metric_lookup(interrater_rows)
        if lookup.get("double_rated_clips"):
            lines.extend(
                [
                    "",
                    "## Human Agreement",
                    "",
                    f"- Double-rated clips: `{int(float(lookup.get('double_rated_clips', 0) or 0))}`",
                    f"- Observed agreement: `{float(lookup.get('observed_agreement', 0.0) or 0.0):.4f}`",
                    f"- Cohen's kappa: `{float(lookup.get('cohen_kappa', 0.0) or 0.0):.4f}`",
                ]
            )

    test_metrics = [row for row in metrics_rows if row.get("scope") == "test"]
    candidate_main = test_metrics or metrics_rows
    if candidate_main:
        best_main = max(candidate_main, key=lambda row: float(row.get("macro_f1", 0.0)))
        lines.extend(
            [
                "",
                "## Best Main Result",
                "",
                f"- Model: `{best_main.get('model_name', '')}`",
                f"- Scope: `{best_main.get('scope', '')}`",
                f"- Method: `{best_main.get('method', '')}`",
                f"- Macro-F1: `{float(best_main.get('macro_f1', 0.0)):.4f}`",
                f"- Balanced accuracy: `{float(best_main.get('balanced_accuracy', 0.0)):.4f}`",
            ]
        )
    if clip_metrics_rows:
        best_clip = max(clip_metrics_rows, key=lambda row: float(row.get("macro_f1", 0.0)))
        lines.extend(
            [
                "",
                "## Best Clip-Level Result",
                "",
                f"- Model: `{best_clip.get('model_name', '')}`",
                f"- Method: `{best_clip.get('method', '')}`",
                f"- Macro-F1: `{float(best_clip.get('macro_f1', 0.0)):.4f}`",
            ]
        )

    lines.extend(
        [
            "",
            "## Core Paper Assets",
            "",
            "- Main ranking figure: `paper_assets/figures/main_test_macro_f1.png`",
            "- Compact scorecard: `paper_assets/figures/main_test_scorecard.png`",
            "- Clip-level ranking: `paper_assets/figures/clip_models_macro_f1.png`",
            "- Human agreement overview: `paper_assets/figures/interrater_overview.png`",
            "- Selected confusion panel: `paper_assets/figures/selected_confusions.png`",
        ]
    )
    return "\n".join(lines) + "\n"


def build_data_sheet(manifest_rows, label_rows):
    split_counts = Counter(row["split"] for row in manifest_rows)
    label_counts = Counter(row["gold_label"] for row in label_rows)
    lines = [
        "# Data Sheet",
        "",
        "## Target Data",
        "",
        "- Dataset: `hhoangphuoc/ami-av` derived from the AMI Meeting Corpus.",
        "- Unit of analysis: close-up clip of a single participant.",
        f"- Manifest clips: `{len(manifest_rows)}`",
        f"- Split counts: `{dict(split_counts)}`",
        f"- Meetings represented: `{len({row.get('meeting_id', '') for row in manifest_rows if row.get('meeting_id', '')})}`",
        "",
        "## Outcome",
        "",
        "- Three-class observable valence: `negative`, `neutral`, `positive`.",
        f"- Current resolved labels: `negative={label_counts.get('negative', 0)}`, `neutral={label_counts.get('neutral', 0)}`, `positive={label_counts.get('positive', 0)}`",
        "- Labels should be framed as observable affect, not latent psychological state.",
    ]
    return "\n".join(lines) + "\n"


def build_limitations_and_ethics():
    lines = [
        "# Limitations and Ethics",
        "",
        "- The study infers observable facial valence, not internal emotion or intent.",
        "- AMI is a research corpus and should not be framed as a deployment-ready surveillance setting.",
        "- Domain shift remains severe because meeting video contains pose changes, occlusion, subtle affect and conversational context that are weakly captured by frame-only FER backbones.",
        "- The current pilot already includes a second human rater, but disagreements have not yet been fully adjudicated into a publication-grade consensus set.",
        "- Results should be interpreted as evidence about transfer robustness and representation choice, not as claims of universal affect recognition performance.",
    ]
    return "\n".join(lines) + "\n"


def build_reproducibility_checklist(config_path, manifest_path, labels_path, pilot_dir, clip_model_dir):
    checks = [
        ("Config file versioned", Path(config_path).exists()),
        ("Manifest file present", Path(manifest_path).exists()),
        ("Resolved labels present", Path(labels_path).exists()),
        ("Annotation guidelines present", Path("docs/annotation_guidelines.md").exists()),
        ("Scenario splits present", (Path(pilot_dir) / "scenario_splits.json").exists()),
        ("Interrater summary present", (Path(pilot_dir) / "interrater_agreement.csv").exists()),
        ("Main metrics present", (Path(pilot_dir) / "metrics.csv").exists()),
        ("Confusion matrices present", (Path(pilot_dir) / "confusion_matrices.csv").exists()),
        ("Clip-level metrics present", bool(clip_model_dir) and (Path(clip_model_dir) / "clip_model_metrics.csv").exists()),
        ("Paper assets generated", (Path(pilot_dir) / "paper_assets").exists()),
    ]
    lines = ["# Reproducibility Checklist", ""]
    for label, status in checks:
        marker = "[x]" if status else "[ ]"
        lines.append(f"- {marker} {label}")
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    model_specs = resolve_model_specs(config)
    manifest_rows = read_csv_rows(args.manifest)
    label_rows = labeled_rows(read_csv_rows(args.labels))
    metrics_rows = load_metrics(Path(args.pilot_dir) / "metrics.json")
    clip_metrics_rows = load_metrics(Path(args.clip_model_dir) / "clip_model_metrics.json") if args.clip_model_dir else []
    interrater_rows = load_metrics(Path(args.pilot_dir) / "interrater_agreement.csv")

    (output_dir / "model_registry.yaml").write_text(dump_yaml_models(model_specs), encoding="utf-8")
    (output_dir / "experiment_card.md").write_text(
        build_experiment_card(config, model_specs, manifest_rows, label_rows, metrics_rows, clip_metrics_rows, interrater_rows),
        encoding="utf-8",
    )
    (output_dir / "data_sheet.md").write_text(build_data_sheet(manifest_rows, label_rows), encoding="utf-8")
    (output_dir / "limitations_and_ethics.md").write_text(build_limitations_and_ethics(), encoding="utf-8")
    (output_dir / "reproducibility_checklist.md").write_text(
        build_reproducibility_checklist(
            args.config,
            args.manifest,
            args.labels,
            args.pilot_dir,
            args.clip_model_dir,
        ),
        encoding="utf-8",
    )
    guidelines_path = Path("docs/annotation_guidelines.md")
    if guidelines_path.exists():
        (output_dir / "annotation_guidelines.md").write_text(guidelines_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Wrote publication reports under {output_dir}")


if __name__ == "__main__":
    main()
