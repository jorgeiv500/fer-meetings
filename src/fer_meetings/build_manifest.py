import argparse
import json
from pathlib import Path

from fer_meetings.temporal import sample_clip_windows
from fer_meetings.utils import write_csv_rows
from fer_meetings.video import get_video_metadata, parse_video_name, resolve_video_path


def parse_args():
    parser = argparse.ArgumentParser(description="Build a clip manifest from local AMI videos.")
    parser.add_argument("--config", required=True, help="Path to JSON config file.")
    parser.add_argument("--raw-dir", required=True, help="Root directory where AMI videos live.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    return parser.parse_args()


def build_rows(config, raw_dir):
    sampling = config["sampling"]
    rows = []

    for split, requested_videos in config["videos"].items():
        for requested_name in requested_videos:
            video_path = resolve_video_path(raw_dir, requested_name)
            video_file = Path(video_path).name
            meeting_id, camera = parse_video_name(video_path)
            metadata = get_video_metadata(video_path)
            windows = sample_clip_windows(
                duration_s=metadata["duration_s"],
                clip_seconds=float(sampling["clip_seconds"]),
                stride_seconds=float(sampling["stride_seconds"]),
                start_offset_seconds=float(sampling.get("start_offset_seconds", 0.0)),
                max_clips=int(sampling.get("max_clips_per_video")) if sampling.get("max_clips_per_video") is not None else None,
            )

            video_id = Path(video_file).stem
            for index, (start_s, end_s) in enumerate(windows, start=1):
                rows.append(
                    {
                        "clip_id": f"{video_id}_{index:03d}",
                        "split": split,
                        "video_file": video_file,
                        "video_path": str(video_path),
                        "meeting_id": meeting_id,
                        "camera": camera,
                        "clip_start_s": f"{start_s:.3f}",
                        "clip_end_s": f"{end_s:.3f}",
                        "video_duration_s": f"{metadata['duration_s']:.3f}",
                    }
                )
    return rows


def main():
    args = parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    rows = build_rows(config, args.raw_dir)
    if not rows:
        raise RuntimeError("No clips were generated from the provided configuration.")

    fieldnames = [
        "clip_id",
        "split",
        "video_file",
        "video_path",
        "meeting_id",
        "camera",
        "clip_start_s",
        "clip_end_s",
        "video_duration_s",
    ]
    write_csv_rows(args.output, rows, fieldnames)
    print(f"Wrote {len(rows)} clips to {args.output}")


if __name__ == "__main__":
    main()
