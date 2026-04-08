import argparse
import csv
import random
from pathlib import Path

from fer_meetings.utils import write_csv_rows


def parse_args():
    parser = argparse.ArgumentParser(description="Build a pilot manifest from the processed ami-av metadata CSV.")
    parser.add_argument("--metadata-csv", required=True, help="Path to ami-segments-info.csv.")
    parser.add_argument(
        "--video-root",
        required=True,
        help="Local directory that contains the extracted MP4 files, typically .../video_segments/original_videos.",
    )
    parser.add_argument("--output", required=True, help="Output manifest CSV.")
    parser.add_argument("--min-duration", type=float, default=3.0, help="Minimum segment duration in seconds.")
    parser.add_argument("--max-duration", type=float, default=5.0, help="Maximum segment duration in seconds.")
    parser.add_argument(
        "--max-clips-per-split",
        type=int,
        default=100,
        help="Maximum number of clips to keep in each split.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Keep rows even if the local video file does not exist yet.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed used to shuffle candidates before sampling.")
    return parser.parse_args()


def read_metadata_rows(path):
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def resolve_local_video_path(video_root, video_column_value):
    basename = Path(video_column_value or "").name
    if not basename:
        return ""
    return str((Path(video_root) / basename).resolve())


def assign_split(meeting_id):
    numeric_part = "".join(character for character in str(meeting_id) if character.isdigit())
    if numeric_part:
        return "dev" if int(numeric_part) % 2 == 0 else "test"
    return "dev" if sum(ord(character) for character in str(meeting_id)) % 2 == 0 else "test"


def keep_row(row, min_duration, max_duration):
    try:
        duration = float(row.get("duration", 0.0))
    except ValueError:
        return False

    has_video = str(row.get("has_video", "")).strip().lower() == "true"
    return has_video and min_duration <= duration <= max_duration


def build_rows(rows, video_root, min_duration, max_duration, max_clips_per_split, allow_missing, seed):
    candidates_by_split = {"dev": [], "test": []}

    for row in rows:
        if not keep_row(row, min_duration, max_duration):
            continue
        split = assign_split(row.get("meeting_id", ""))
        local_video_path = resolve_local_video_path(video_root, row.get("video", ""))
        if not allow_missing and (not local_video_path or not Path(local_video_path).is_file()):
            continue

        duration = float(row["duration"])
        manifest_row = {
            "clip_id": row["id"],
            "split": split,
            "video_file": Path(local_video_path).name if local_video_path else Path(row.get("video", "")).name,
            "video_path": local_video_path or row.get("video", ""),
            "meeting_id": row["meeting_id"],
            "speaker_id": row.get("speaker_id", ""),
            "camera": f"speaker_{row.get('speaker_id', '')}",
            "clip_start_s": "0.000",
            "clip_end_s": f"{duration:.3f}",
            "video_duration_s": f"{duration:.3f}",
            "source_dataset": "ami_av",
        }
        candidates_by_split[split].append(manifest_row)

    random_generator = random.Random(seed)
    final_rows = []
    for split in ("dev", "test"):
        candidates = candidates_by_split[split]
        random_generator.shuffle(candidates)
        if max_clips_per_split > 0:
            candidates = candidates[:max_clips_per_split]
        final_rows.extend(sorted(candidates, key=lambda item: item["clip_id"]))
    return final_rows


def main():
    args = parse_args()
    rows = read_metadata_rows(args.metadata_csv)
    manifest_rows = build_rows(
        rows=rows,
        video_root=args.video_root,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        max_clips_per_split=args.max_clips_per_split,
        allow_missing=args.allow_missing,
        seed=args.seed,
    )
    if not manifest_rows:
        raise RuntimeError("No manifest rows were produced. Check durations, local video paths, or use --allow-missing.")

    fieldnames = [
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
    ]
    write_csv_rows(args.output, manifest_rows, fieldnames)
    print(f"Wrote {len(manifest_rows)} ami-av clips to {args.output}")


if __name__ == "__main__":
    main()
