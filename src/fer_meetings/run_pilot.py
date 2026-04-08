import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from fer_meetings.config import load_config, resolve_model_specs
from fer_meetings.constants import LABEL_ORDER
from fer_meetings.model import HfEmotionClassifier
from fer_meetings.temporal import majority_vote, sample_frame_times
from fer_meetings.utils import read_csv_rows, write_csv_rows
from fer_meetings.video import open_video, read_frame_at


def parse_args():
    parser = argparse.ArgumentParser(description="Run zero-shot FER inference over an AMI clip manifest.")
    parser.add_argument("--config", default="", help="Optional JSON config file.")
    parser.add_argument("--manifest", required=True, help="Input manifest CSV.")
    parser.add_argument("--output", required=True, help="Output predictions CSV.")
    parser.add_argument(
        "--frame-details-output",
        default="",
        help="Optional CSV with per-frame predictions.",
    )
    parser.add_argument(
        "--clip-features-output",
        default="",
        help="Optional CSV with clip-level pooled embeddings and temporal features.",
    )
    parser.add_argument(
        "--model-id",
        action="append",
        default=None,
        help="Optional Hugging Face model id. Repeat the flag to compare multiple models.",
    )
    parser.add_argument(
        "--label-map",
        default="configs/label_map_3class.json",
        help="Path to the raw-label to 3-class mapping JSON.",
    )
    parser.add_argument(
        "--frames-per-clip",
        type=int,
        default=0,
        help="Number of frames sampled uniformly inside each clip.",
    )
    parser.add_argument(
        "--face-detector",
        choices=["haar", "full-frame"],
        default="",
        help="Face detector to use before the classifier.",
    )
    parser.add_argument(
        "--include-embeddings",
        action="store_true",
        help="Export pooled and frame-level embeddings when supported by the backbone.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda or mps.",
    )
    return parser.parse_args()


def aggregate_probabilities(frame_predictions):
    matrix = np.array(
        [[frame_prediction[label] for label in LABEL_ORDER] for frame_prediction in frame_predictions],
        dtype=float,
    )
    mean_probabilities = matrix.mean(axis=0)
    return {
        label: float(mean_probabilities[index])
        for index, label in enumerate(LABEL_ORDER)
    }


def aggregate_embeddings(frame_embeddings):
    if not frame_embeddings:
        return [], []
    matrix = np.array(frame_embeddings, dtype=float)
    return matrix.mean(axis=0).tolist(), matrix.std(axis=0).tolist()


def signed_valence(probabilities):
    return float(probabilities["positive"]) - float(probabilities["negative"])


def compact_json(value):
    return json.dumps(value, separators=(",", ":"))


def build_prediction_row(model_spec, clip_row, frame_records):
    center_frame = frame_records[len(frame_records) // 2]
    smoothed_probabilities = aggregate_probabilities(
        [frame_record["mapped_probabilities"] for frame_record in frame_records]
    )
    single_frame_probabilities = center_frame["mapped_probabilities"]

    prediction_row = dict(clip_row)
    prediction_row.update(
        {
            "model_name": model_spec["name"],
            "model_family": model_spec["family"],
            "hf_model_id": model_spec["hf_model_id"],
            "single_frame_label": center_frame["mapped_label"],
            "single_frame_raw_label": center_frame["raw_label"],
            "smoothed_label": max(LABEL_ORDER, key=lambda label: smoothed_probabilities[label]),
            "vote_label": majority_vote(
                [frame_record["mapped_label"] for frame_record in frame_records],
                LABEL_ORDER,
            ),
            "frames_used": str(len(frame_records)),
            "face_detected_ratio": f"{np.mean([frame_record['face_detected'] for frame_record in frame_records]):.3f}",
            "center_timestamp_s": f"{center_frame['timestamp_s']:.3f}",
        }
    )

    for label in LABEL_ORDER:
        prediction_row[f"single_frame_{label}_prob"] = f"{single_frame_probabilities[label]:.6f}"
        prediction_row[f"smoothed_{label}_prob"] = f"{smoothed_probabilities[label]:.6f}"

    return prediction_row


def build_frame_detail_rows(model_spec, frame_records, clip_row, include_embeddings):
    rows = []
    for frame_index, frame_record in enumerate(frame_records, start=1):
        row = {
            "clip_id": clip_row["clip_id"],
            "split": clip_row["split"],
            "model_name": model_spec["name"],
            "model_family": model_spec["family"],
            "hf_model_id": model_spec["hf_model_id"],
            "frame_index": frame_index,
            "timestamp_s": f"{frame_record['timestamp_s']:.3f}",
            "raw_label": frame_record["raw_label"],
            "mapped_label": frame_record["mapped_label"],
            "face_detected": str(bool(frame_record["face_detected"])).lower(),
        }
        for label in LABEL_ORDER:
            row[f"{label}_prob"] = f"{frame_record['mapped_probabilities'][label]:.6f}"
        if include_embeddings:
            row["embedding_json"] = compact_json(frame_record["embedding"])
        rows.append(row)
    return rows


def build_clip_feature_row(model_spec, clip_row, frame_records):
    frame_embeddings = [frame_record["embedding"] for frame_record in frame_records if frame_record["embedding"]]
    mean_embedding, std_embedding = aggregate_embeddings(frame_embeddings)
    signed_scores = [signed_valence(frame_record["mapped_probabilities"]) for frame_record in frame_records]
    smoothed_probabilities = aggregate_probabilities(
        [frame_record["mapped_probabilities"] for frame_record in frame_records]
    )

    return {
        "clip_id": clip_row["clip_id"],
        "split": clip_row["split"],
        "video_file": clip_row["video_file"],
        "video_path": clip_row["video_path"],
        "meeting_id": clip_row["meeting_id"],
        "camera": clip_row["camera"],
        "clip_start_s": clip_row["clip_start_s"],
        "clip_end_s": clip_row["clip_end_s"],
        "model_name": model_spec["name"],
        "model_family": model_spec["family"],
        "hf_model_id": model_spec["hf_model_id"],
        "frames_used": str(len(frame_records)),
        "face_detected_ratio": f"{np.mean([frame_record['face_detected'] for frame_record in frame_records]):.3f}",
        "signed_valence_mean": f"{np.mean(signed_scores):.6f}",
        "signed_valence_std": f"{np.std(signed_scores):.6f}",
        "signed_valence_delta": f"{(signed_scores[-1] - signed_scores[0]) if len(signed_scores) > 1 else 0.0:.6f}",
        "smoothed_label": max(LABEL_ORDER, key=lambda label: smoothed_probabilities[label]),
        "mean_embedding_json": compact_json(mean_embedding),
        "std_embedding_json": compact_json(std_embedding),
        "frame_embeddings_json": compact_json(frame_embeddings),
        "frame_probabilities_json": compact_json([frame_record["mapped_probabilities"] for frame_record in frame_records]),
        "frame_labels_json": compact_json([frame_record["mapped_label"] for frame_record in frame_records]),
        "frame_timestamps_json": compact_json([round(frame_record["timestamp_s"], 3) for frame_record in frame_records]),
    }


def main():
    args = parse_args()
    config = load_config(args.config)

    sampling_config = config.get("sampling", {})
    manifest_rows = read_csv_rows(args.manifest)
    model_specs = resolve_model_specs(
        config=config,
        requested_model_ids=args.model_id,
        requested_face_detector=args.face_detector,
    )
    frames_per_clip = args.frames_per_clip or int(sampling_config.get("frames_per_clip", 5))
    include_embeddings = bool(args.include_embeddings or args.clip_features_output)

    grouped_rows = defaultdict(list)
    for row in manifest_rows:
        grouped_rows[row["video_path"]].append(row)

    prediction_rows = []
    frame_detail_rows = []
    clip_feature_rows = []
    for model_spec in model_specs:
        classifier = HfEmotionClassifier(
            model_id=model_spec["hf_model_id"],
            label_map_path=args.label_map,
            face_detector=model_spec["face_detector"],
            device=args.device,
            model_name=model_spec["name"],
            model_family=model_spec["family"],
        )

        for video_path, clip_rows in grouped_rows.items():
            capture = open_video(video_path)
            fps = float(capture.get(5) or 0.0)
            if fps <= 0:
                capture.release()
                raise RuntimeError(f"Invalid FPS for video: {video_path}")

            for clip_row in clip_rows:
                frame_times = sample_frame_times(
                    start_s=float(clip_row["clip_start_s"]),
                    end_s=float(clip_row["clip_end_s"]),
                    frames_per_clip=frames_per_clip,
                )

                frame_records = []
                for timestamp_s in frame_times:
                    frame = read_frame_at(capture, fps, timestamp_s)
                    if frame is None:
                        continue
                    frame_prediction = classifier.predict_frame(frame, include_embedding=include_embeddings)
                    frame_prediction["timestamp_s"] = timestamp_s
                    frame_records.append(frame_prediction)

                if not frame_records:
                    continue

                prediction_rows.append(build_prediction_row(model_spec, clip_row, frame_records))
                if args.frame_details_output:
                    frame_detail_rows.extend(
                        build_frame_detail_rows(model_spec, frame_records, clip_row, include_embeddings)
                    )
                if args.clip_features_output:
                    clip_feature_rows.append(build_clip_feature_row(model_spec, clip_row, frame_records))

            capture.release()

    prediction_fieldnames = [
        "clip_id",
        "split",
        "video_file",
        "video_path",
        "meeting_id",
        "speaker_id",
        "camera",
        "clip_start_s",
        "clip_end_s",
        "video_duration_s",
        "source_dataset",
        "model_name",
        "model_family",
        "hf_model_id",
        "single_frame_label",
        "single_frame_raw_label",
        "smoothed_label",
        "vote_label",
        "frames_used",
        "face_detected_ratio",
        "center_timestamp_s",
    ] + [
        f"{method}_{label}_prob"
        for method in ("single_frame", "smoothed")
        for label in LABEL_ORDER
    ]
    write_csv_rows(args.output, prediction_rows, prediction_fieldnames)
    print(f"Wrote {len(prediction_rows)} clip predictions to {args.output}")

    if args.frame_details_output and frame_detail_rows:
        frame_fieldnames = [
            "clip_id",
            "split",
            "speaker_id",
            "source_dataset",
            "model_name",
            "model_family",
            "hf_model_id",
            "frame_index",
            "timestamp_s",
            "raw_label",
            "mapped_label",
            "face_detected",
        ] + [f"{label}_prob" for label in LABEL_ORDER]
        if include_embeddings:
            frame_fieldnames.append("embedding_json")
        write_csv_rows(args.frame_details_output, frame_detail_rows, frame_fieldnames)
        print(f"Wrote {len(frame_detail_rows)} frame-level rows to {args.frame_details_output}")

    if args.clip_features_output and clip_feature_rows:
        clip_feature_fieldnames = [
            "clip_id",
            "split",
            "video_file",
            "video_path",
            "meeting_id",
            "speaker_id",
            "camera",
            "clip_start_s",
            "clip_end_s",
            "source_dataset",
            "model_name",
            "model_family",
            "hf_model_id",
            "frames_used",
            "face_detected_ratio",
            "signed_valence_mean",
            "signed_valence_std",
            "signed_valence_delta",
            "smoothed_label",
            "mean_embedding_json",
            "std_embedding_json",
            "frame_embeddings_json",
            "frame_probabilities_json",
            "frame_labels_json",
            "frame_timestamps_json",
        ]
        write_csv_rows(args.clip_features_output, clip_feature_rows, clip_feature_fieldnames)
        print(f"Wrote {len(clip_feature_rows)} clip feature rows to {args.clip_features_output}")


if __name__ == "__main__":
    main()
