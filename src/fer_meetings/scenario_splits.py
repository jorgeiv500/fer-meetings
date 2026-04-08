import argparse
import json
from collections import defaultdict
from pathlib import Path

from fer_meetings.utils import read_csv_rows


def parse_args():
    parser = argparse.ArgumentParser(description="Export explicit split assignments by meeting/scenario from a manifest CSV.")
    parser.add_argument("--manifest", required=True, help="Input manifest CSV.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    return parser.parse_args()


def build_splits(manifest_rows):
    splits = defaultdict(lambda: {"clip_ids": [], "meeting_ids": set(), "video_files": []})
    clip_to_split = {}
    for row in manifest_rows:
        split = row.get("split", "")
        meeting_id = row.get("meeting_id", "")
        clip_id = row.get("clip_id", "")
        splits[split]["clip_ids"].append(clip_id)
        if meeting_id:
            splits[split]["meeting_ids"].add(meeting_id)
        if row.get("video_file", ""):
            splits[split]["video_files"].append(row["video_file"])
        clip_to_split[clip_id] = split

    payload = {"splits": {}, "clip_to_split": clip_to_split}
    for split, values in sorted(splits.items()):
        payload["splits"][split] = {
            "clip_count": len(values["clip_ids"]),
            "meeting_count": len(values["meeting_ids"]),
            "meeting_ids": sorted(values["meeting_ids"]),
            "clip_ids": sorted(values["clip_ids"]),
            "video_files": sorted(values["video_files"]),
        }
    return payload


def main():
    args = parse_args()
    manifest_rows = read_csv_rows(args.manifest)
    payload = build_splits(manifest_rows)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote scenario splits to {output_path}")


if __name__ == "__main__":
    main()
