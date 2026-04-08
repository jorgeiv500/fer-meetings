import argparse
import csv
import tarfile
from pathlib import Path

import requests
from huggingface_hub import hf_hub_url

from fer_meetings.build_ami_av_manifest import assign_split
from fer_meetings.utils import ensure_parent


def parse_args():
    parser = argparse.ArgumentParser(description="Stream-extract a small AMI video subset from hhoangphuoc/ami-av.")
    parser.add_argument(
        "--metadata-csv",
        required=True,
        help="Path to ami-segments-info.csv used to filter durations and map basenames to metadata rows.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where video_segments/original_videos will be created.",
    )
    parser.add_argument("--min-duration", type=float, default=3.0, help="Minimum segment duration.")
    parser.add_argument("--max-duration", type=float, default=5.0, help="Maximum segment duration.")
    parser.add_argument("--per-split", type=int, default=5, help="Target number of dev and test clips to extract.")
    parser.add_argument("--timeout", type=int, default=120, help="HTTP timeout in seconds.")
    return parser.parse_args()


def read_metadata_by_basename(path):
    mapping = {}
    with open(path, "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            basename = Path(row.get("video", "")).name
            if basename:
                mapping[basename] = row
    return mapping


def keep_row(row, min_duration, max_duration):
    try:
        duration = float(row.get("duration", 0.0))
    except ValueError:
        return False
    has_video = str(row.get("has_video", "")).strip().lower() == "true"
    return has_video and min_duration <= duration <= max_duration


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    video_root = output_dir / "video_segments" / "original_videos"
    video_root.mkdir(parents=True, exist_ok=True)

    metadata_by_basename = read_metadata_by_basename(args.metadata_csv)
    selected_counts = {"dev": 0, "test": 0}
    selected_rows = []

    url = hf_hub_url("hhoangphuoc/ami-av", "video_segments.tar.gz", repo_type="dataset")
    with requests.get(url, stream=True, timeout=args.timeout) as response:
        response.raise_for_status()
        response.raw.decode_content = True
        archive = tarfile.open(fileobj=response.raw, mode="r|gz")

        for member in archive:
            if not member.isfile():
                continue
            basename = Path(member.name).name
            row = metadata_by_basename.get(basename)
            if row is None or not keep_row(row, args.min_duration, args.max_duration):
                continue

            split = assign_split(row.get("meeting_id", ""))
            if selected_counts[split] >= args.per_split:
                continue

            destination = video_root / basename
            ensure_parent(destination)
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            with open(destination, "wb") as handle:
                while True:
                    chunk = extracted.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)

            selected_counts[split] += 1
            selected_rows.append(
                {
                    "split": split,
                    "meeting_id": row.get("meeting_id", ""),
                    "speaker_id": row.get("speaker_id", ""),
                    "id": row.get("id", ""),
                    "duration": row.get("duration", ""),
                    "basename": basename,
                    "relative_path": str(destination.relative_to(output_dir)),
                }
            )
            print(f"Fetched {basename} -> {split}")

            if all(count >= args.per_split for count in selected_counts.values()):
                break

    manifest_path = output_dir / "subset_manifest.csv"
    if selected_rows:
        with open(manifest_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["split", "meeting_id", "speaker_id", "id", "duration", "basename", "relative_path"],
            )
            writer.writeheader()
            writer.writerows(selected_rows)
    print(f"Wrote {len(selected_rows)} selected segments to {manifest_path}")


if __name__ == "__main__":
    main()
