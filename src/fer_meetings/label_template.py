import argparse

from fer_meetings.utils import read_csv_rows, write_csv_rows


def parse_args():
    parser = argparse.ArgumentParser(description="Create a manual annotation template from a clip manifest.")
    parser.add_argument("--manifest", required=True, help="Input manifest CSV.")
    parser.add_argument("--output", required=True, help="Output labels CSV.")
    return parser.parse_args()


def main():
    args = parse_args()
    manifest_rows = read_csv_rows(args.manifest)
    label_rows = []
    for row in manifest_rows:
        label_rows.append(
            {
                "clip_id": row["clip_id"],
                "split": row["split"],
                "video_file": row["video_file"],
                "meeting_id": row["meeting_id"],
                "camera": row["camera"],
                "clip_start_s": row["clip_start_s"],
                "clip_end_s": row["clip_end_s"],
                "rater_1_label": "",
                "rater_2_label": "",
                "adjudicated_label": "",
                "gold_label": "",
                "annotator": "",
                "adjudicator": "",
                "exclude_from_gold": "",
                "agreement_status": "",
                "notes": "",
            }
        )

    fieldnames = [
        "clip_id",
        "split",
        "video_file",
        "meeting_id",
        "camera",
        "clip_start_s",
        "clip_end_s",
        "rater_1_label",
        "rater_2_label",
        "adjudicated_label",
        "gold_label",
        "annotator",
        "adjudicator",
        "exclude_from_gold",
        "agreement_status",
        "notes",
    ]
    write_csv_rows(args.output, label_rows, fieldnames)
    print(f"Wrote annotation template with {len(label_rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
