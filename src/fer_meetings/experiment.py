import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from fer_meetings.labels import resolve_gold_label
from fer_meetings.utils import read_csv_rows


def parse_args():
    parser = argparse.ArgumentParser(description="Run the AMI AV experiment end-to-end in prelabels or postlabels mode.")
    parser.add_argument("--config", required=True, help="Experiment config JSON.")
    parser.add_argument("--manifest", required=True, help="Clip manifest CSV.")
    parser.add_argument("--output-dir", required=True, help="Experiment output directory.")
    parser.add_argument(
        "--labels",
        default="",
        help="Optional labels CSV. Defaults to <output-dir>/annotation_pack/annotation_sheet.csv.",
    )
    parser.add_argument(
        "--phase",
        choices=["full", "prelabels", "postlabels"],
        default="full",
        help="prelabels runs pilot + annotation pack; postlabels runs evaluation/assets; full does both.",
    )
    parser.add_argument("--frames-per-clip", type=int, default=0, help="Optional override for fer-run-pilot.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or mps.")
    return parser.parse_args()


def run_step(args):
    printable = " ".join(shlex.quote(str(part)) for part in args)
    print(f"$ {printable}")
    subprocess.run(args, check=True)


def gold_label_count(path):
    if not Path(path).exists():
        return 0
    return sum(
        1
        for row in read_csv_rows(path)
        if resolve_gold_label(row)[0]
    )


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = output_dir / "predictions.csv"
    frame_details_path = output_dir / "frame_details.csv"
    clip_features_path = output_dir / "clip_features.csv"
    annotation_dir = output_dir / "annotation_pack"
    labels_path = Path(args.labels) if args.labels else annotation_dir / "annotation_sheet.csv"
    resolved_labels_path = output_dir / "clip_labels.csv"
    clip_model_dir = output_dir / "clip_models"
    paper_assets_dir = output_dir / "paper_assets"
    reports_dir = output_dir / "reports"
    scenario_splits_path = output_dir / "scenario_splits.json"

    if args.phase in {"full", "prelabels"}:
        pilot_command = [
            sys.executable,
            "-m",
            "fer_meetings.run_pilot",
            "--config",
            args.config,
            "--manifest",
            args.manifest,
            "--output",
            str(predictions_path),
            "--frame-details-output",
            str(frame_details_path),
            "--clip-features-output",
            str(clip_features_path),
            "--device",
            args.device,
        ]
        if args.frames_per_clip > 0:
            pilot_command.extend(["--frames-per-clip", str(args.frames_per_clip)])
        run_step(pilot_command)

        run_step(
            [
                sys.executable,
                "-m",
                "fer_meetings.annotation_pack",
                "--manifest",
                args.manifest,
                "--predictions",
                str(predictions_path),
                "--output-dir",
                str(annotation_dir),
                "--labels-output",
                str(labels_path),
            ]
        )

        if args.phase == "prelabels":
            return

    gold_count = gold_label_count(labels_path)
    if gold_count == 0:
        print(
            f"No human gold labels found in {labels_path}. "
            "Fill gold_label, adjudicated_label or matching rater_1_label/rater_2_label and rerun with --phase postlabels."
        )
        return

    run_step(
        [
            sys.executable,
            "-m",
            "fer_meetings.interrater",
            "--labels",
            str(labels_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    run_step(
        [
            sys.executable,
            "-m",
            "fer_meetings.scenario_splits",
            "--manifest",
            args.manifest,
            "--output",
            str(scenario_splits_path),
        ]
    )

    run_step(
        [
            sys.executable,
            "-m",
            "fer_meetings.evaluate",
            "--predictions",
            str(predictions_path),
            "--labels",
            str(resolved_labels_path),
            "--output-dir",
            str(output_dir),
            "--fit-calibrator",
        ]
    )

    run_step(
        [
            sys.executable,
            "-m",
            "fer_meetings.train_clip_models",
            "--clip-features",
            str(clip_features_path),
            "--labels",
            str(resolved_labels_path),
            "--output-dir",
            str(clip_model_dir),
            "--device",
            args.device,
        ]
    )

    run_step(
        [
            sys.executable,
            "-m",
            "fer_meetings.paper_assets",
            "--pilot-dir",
            str(output_dir),
            "--clip-model-dir",
            str(clip_model_dir),
            "--manifest",
            args.manifest,
            "--labels",
            str(resolved_labels_path),
            "--output-dir",
            str(paper_assets_dir),
        ]
    )

    run_step(
        [
            sys.executable,
            "-m",
            "fer_meetings.reporting",
            "--config",
            args.config,
            "--manifest",
            args.manifest,
            "--labels",
            str(resolved_labels_path),
            "--pilot-dir",
            str(output_dir),
            "--clip-model-dir",
            str(clip_model_dir),
            "--output-dir",
            str(reports_dir),
        ]
    )


if __name__ == "__main__":
    main()
