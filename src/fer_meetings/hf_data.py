import argparse
import csv
import json
import pickle
import shutil
from pathlib import Path

from fer_meetings.utils import ensure_parent


def parse_args():
    parser = argparse.ArgumentParser(description="Download and export datasets from Hugging Face.")
    parser.add_argument("--dataset-id", required=True, help="Dataset id on Hugging Face, e.g. Jeneral/fer-2013.")
    parser.add_argument("--subset", default="", help="Optional config/subset name.")
    parser.add_argument("--split", default="", help="Optional split to export. Leave empty to export all splits.")
    parser.add_argument("--output-dir", required=True, help="Directory where the exported files will be written.")
    parser.add_argument("--limit", type=int, default=0, help="Optional maximum number of rows per split.")
    parser.add_argument(
        "--image-column",
        default="",
        help="Image column to export. If omitted, a likely image column is inferred.",
    )
    parser.add_argument(
        "--label-column",
        default="",
        help="Optional label column used to create class subfolders for image export.",
    )
    parser.add_argument(
        "--bytes-column",
        default="",
        help="Optional bytes column to export when images are stored as raw bytes instead of Image features.",
    )
    parser.add_argument(
        "--video-column",
        default="",
        help="Optional video column to export when the dataset exposes a video feature.",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Only write metadata rows without exporting media files.",
    )
    parser.add_argument(
        "--snapshot-only",
        action="store_true",
        help="Download the raw dataset repository with huggingface_hub instead of using the datasets library.",
    )
    parser.add_argument(
        "--snapshot-dir",
        default="",
        help="Optional path to an already downloaded dataset snapshot. When provided, no network call is made.",
    )
    return parser.parse_args()


def load_dataset_builder(dataset_id, subset=""):
    from datasets import load_dataset_builder

    return load_dataset_builder(dataset_id, subset or None)


def load_hf_dataset(dataset_id, subset="", split=""):
    from datasets import get_dataset_split_names, load_dataset

    splits = [split] if split else list(get_dataset_split_names(dataset_id, subset or None))
    datasets_by_split = {}
    for split_name in splits:
        datasets_by_split[split_name] = load_dataset(dataset_id, subset or None, split=split_name)
    return datasets_by_split


def snapshot_dataset_repo(dataset_id, output_dir):
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=dataset_id,
        repo_type="dataset",
        local_dir=str(output_dir),
    )


def feature_type_name(feature):
    return feature.__class__.__name__.lower()


def infer_media_columns(features):
    image_candidates = []
    video_candidates = []
    bytes_candidates = []
    label_candidates = []

    for name, feature in features.items():
        type_name = feature_type_name(feature)
        if type_name == "image":
            image_candidates.append(name)
        elif type_name == "video":
            video_candidates.append(name)
        elif type_name == "classlabel":
            label_candidates.append(name)
        elif "binary" in type_name or "value" in type_name:
            lower_name = name.lower()
            if "byte" in lower_name or "image" in lower_name:
                bytes_candidates.append(name)
            if "label" in lower_name or "emotion" in lower_name:
                label_candidates.append(name)
    return {
        "image_column": image_candidates[0] if image_candidates else "",
        "video_column": video_candidates[0] if video_candidates else "",
        "bytes_column": bytes_candidates[0] if bytes_candidates else "",
        "label_column": label_candidates[0] if label_candidates else "",
    }


def decode_label(feature, raw_value):
    if raw_value is None:
        return ""
    if feature_type_name(feature) == "classlabel":
        return feature.int2str(int(raw_value))
    return str(raw_value)


def row_identifier(row, index):
    for key in ("id", "image_id", "clip_id", "utterance_id", "filename", "file_name", "path"):
        value = row.get(key)
        if value:
            return str(value)
    return f"{index:06d}"


def sanitize_label(value):
    text = str(value).strip().replace("/", "_").replace(" ", "_")
    return text or "unlabeled"


def export_image_row(row, index, split_name, output_dir, image_column="", bytes_column="", label_column="", features=None, metadata_only=False):
    image_value = row.get(image_column) if image_column else None
    byte_value = row.get(bytes_column) if bytes_column else None

    if image_value is None and byte_value is None:
        return None

    label_value = ""
    if label_column:
        raw_label = row.get(label_column)
        if features and label_column in features:
            label_value = decode_label(features[label_column], raw_label)
        elif raw_label is not None:
            label_value = str(raw_label)
    label_dir = sanitize_label(label_value) if label_value else "unlabeled"
    identifier = row_identifier(row, index)

    if metadata_only:
        relative_path = ""
    else:
        destination = Path(output_dir) / split_name / label_dir / f"{identifier}.png"
        ensure_parent(destination)
        if image_value is not None and hasattr(image_value, "save"):
            image_value.save(destination)
        else:
            raw_bytes = byte_value or image_value
            destination.write_bytes(bytes(raw_bytes))
        relative_path = str(destination.relative_to(output_dir))

    return {
        "split": split_name,
        "row_id": identifier,
        "label": label_value,
        "relative_path": relative_path,
    }


def export_video_row(row, index, split_name, output_dir, video_column="", metadata_only=False):
    video_value = row.get(video_column)
    if not video_value:
        return None

    identifier = row_identifier(row, index)
    relative_path = ""
    source_path = ""

    if isinstance(video_value, dict):
        source_path = str(video_value.get("path") or "")
        raw_bytes = video_value.get("bytes")
    else:
        source_path = str(video_value)
        raw_bytes = None

    if not metadata_only:
        suffix = Path(source_path).suffix or ".mp4"
        destination = Path(output_dir) / split_name / "videos" / f"{identifier}{suffix}"
        ensure_parent(destination)
        if raw_bytes:
            destination.write_bytes(raw_bytes)
        elif source_path:
            shutil.copy2(source_path, destination)
        else:
            return None
        relative_path = str(destination.relative_to(output_dir))

    return {
        "split": split_name,
        "row_id": identifier,
        "source_path": source_path,
        "relative_path": relative_path,
    }


def write_metadata_rows(output_dir, split_name, rows):
    metadata_path = Path(output_dir) / split_name / "metadata.csv"
    ensure_parent(metadata_path)
    if not rows:
        metadata_path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with open(metadata_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_pickled_snapshot(snapshot_dir, output_dir, bytes_column="", label_column="", metadata_only=False, limit=0, split=""):
    split_names = [split] if split else []
    if not split_names:
        split_names = [path.stem for path in sorted(Path(snapshot_dir).glob("*.pt"))]

    for split_name in split_names:
        source_path = Path(snapshot_dir) / f"{split_name}.pt"
        if not source_path.exists():
            continue
        with source_path.open("rb") as handle:
            records = pickle.load(handle)

        rows = []
        max_rows = limit if limit > 0 else len(records)
        for index, row in enumerate(records):
            if index >= max_rows:
                break
            exported = export_image_row(
                row,
                index=index,
                split_name=split_name,
                output_dir=output_dir,
                image_column="",
                bytes_column=bytes_column,
                label_column=label_column,
                features={},
                metadata_only=metadata_only,
            )
            if exported:
                rows.append(exported)
        write_metadata_rows(output_dir, split_name, rows)
        print(f"Exported {len(rows)} rows for split={split_name} into {output_dir}")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.snapshot_only:
        snapshot_path = Path(args.snapshot_dir) if args.snapshot_dir else Path(snapshot_dataset_repo(args.dataset_id, output_dir))
        manifest = {
            "dataset_id": args.dataset_id,
            "subset": args.subset,
            "source": "snapshot_download",
            "snapshot_path": str(snapshot_path),
            "bytes_column": args.bytes_column,
            "label_column": args.label_column,
        }
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if args.bytes_column:
            export_pickled_snapshot(
                snapshot_dir=snapshot_path,
                output_dir=output_dir,
                bytes_column=args.bytes_column,
                label_column=args.label_column,
                metadata_only=args.metadata_only,
                limit=args.limit,
                split=args.split,
            )
        return

    builder = load_dataset_builder(args.dataset_id, args.subset)
    feature_map = infer_media_columns(builder.info.features)
    image_column = args.image_column or feature_map["image_column"]
    bytes_column = args.bytes_column or feature_map["bytes_column"]
    video_column = args.video_column or feature_map["video_column"]
    label_column = args.label_column or feature_map["label_column"]

    datasets_by_split = load_hf_dataset(args.dataset_id, args.subset, args.split)
    manifest = {
        "dataset_id": args.dataset_id,
        "subset": args.subset,
        "splits": list(datasets_by_split.keys()),
        "image_column": image_column,
        "bytes_column": bytes_column,
        "video_column": video_column,
        "label_column": label_column,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for split_name, dataset in datasets_by_split.items():
        rows = []
        limit = args.limit if args.limit > 0 else len(dataset)
        features = dataset.features
        for index, row in enumerate(dataset):
            if index >= limit:
                break
            if video_column:
                exported = export_video_row(
                    row,
                    index=index,
                    split_name=split_name,
                    output_dir=output_dir,
                    video_column=video_column,
                    metadata_only=args.metadata_only,
                )
            else:
                exported = export_image_row(
                    row,
                    index=index,
                    split_name=split_name,
                    output_dir=output_dir,
                    image_column=image_column,
                    bytes_column=bytes_column,
                    label_column=label_column,
                    features=features,
                    metadata_only=args.metadata_only,
                )
            if exported:
                rows.append(exported)
        write_metadata_rows(output_dir, split_name, rows)
        print(f"Exported {len(rows)} rows for split={split_name} into {output_dir}")


if __name__ == "__main__":
    main()
