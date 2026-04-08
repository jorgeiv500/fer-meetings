import argparse
from pathlib import Path

from sklearn.metrics import cohen_kappa_score

from fer_meetings.labels import canonical_gold_label, resolve_gold_label
from fer_meetings.utils import read_csv_rows, write_csv_rows


def parse_args():
    parser = argparse.ArgumentParser(description="Resolve final clip labels and compute interrater agreement summaries.")
    parser.add_argument("--labels", required=True, help="Annotation sheet CSV.")
    parser.add_argument("--output-dir", required=True, help="Directory where clip_labels.csv and agreement files are written.")
    return parser.parse_args()


def parse_bool(value):
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def agreement_status(rater_1, rater_2):
    if rater_1 and rater_2:
        return "agree" if rater_1 == rater_2 else "disagree"
    if rater_1 or rater_2:
        return "partial"
    return "missing"


def build_outputs(label_rows):
    resolved_rows = []
    pair_rows = []
    agreement_true = []
    agreement_pred = []

    counts = {
        "total_rows": 0,
        "resolved_labels": 0,
        "double_rated_clips": 0,
        "adjudicated_clips": 0,
        "excluded_clips": 0,
    }

    for row in label_rows:
        counts["total_rows"] += 1
        rater_1 = canonical_gold_label(row.get("rater_1_label", ""))
        rater_2 = canonical_gold_label(row.get("rater_2_label", ""))
        excluded = parse_bool(row.get("exclude_from_gold", ""))
        final_label, label_source = resolve_gold_label(row)
        if excluded:
            final_label = ""
            label_source = "excluded"
            counts["excluded_clips"] += 1
        if label_source == "adjudicated_label":
            counts["adjudicated_clips"] += 1
        if final_label:
            counts["resolved_labels"] += 1

        status = agreement_status(rater_1, rater_2)
        if rater_1 and rater_2:
            counts["double_rated_clips"] += 1
            agreement_true.append(rater_1)
            agreement_pred.append(rater_2)
            pair_rows.append(
                {
                    "clip_id": row.get("clip_id", ""),
                    "split": row.get("split", ""),
                    "rater_1_label": rater_1,
                    "rater_2_label": rater_2,
                    "agreement_status": status,
                }
            )

        resolved_rows.append(
            {
                "clip_id": row.get("clip_id", ""),
                "split": row.get("split", ""),
                "video_file": row.get("video_file", ""),
                "meeting_id": row.get("meeting_id", ""),
                "camera": row.get("camera", ""),
                "clip_start_s": row.get("clip_start_s", ""),
                "clip_end_s": row.get("clip_end_s", ""),
                "gold_label": final_label,
                "label_source": label_source,
                "exclude_from_gold": str(excluded).lower(),
                "agreement_status": status,
                "rater_1_label": rater_1,
                "rater_2_label": rater_2,
                "adjudicated_label": canonical_gold_label(row.get("adjudicated_label", "")),
                "annotator": row.get("annotator", ""),
                "adjudicator": row.get("adjudicator", ""),
                "notes": row.get("notes", ""),
            }
        )

    observed_agreement = ""
    kappa = ""
    if agreement_true:
        observed_agreement = sum(int(a == b) for a, b in zip(agreement_true, agreement_pred)) / len(agreement_true)
        try:
            kappa = cohen_kappa_score(agreement_true, agreement_pred, labels=["negative", "neutral", "positive"])
        except Exception:
            kappa = ""

    summary_rows = [
        {"metric": "total_rows", "value": counts["total_rows"], "notes": "Rows found in the annotation sheet."},
        {"metric": "resolved_labels", "value": counts["resolved_labels"], "notes": "Rows with a final gold label after resolution."},
        {"metric": "double_rated_clips", "value": counts["double_rated_clips"], "notes": "Rows with both rater_1_label and rater_2_label."},
        {"metric": "adjudicated_clips", "value": counts["adjudicated_clips"], "notes": "Rows where adjudicated_label defined the final label."},
        {"metric": "excluded_clips", "value": counts["excluded_clips"], "notes": "Rows flagged with exclude_from_gold=true."},
        {"metric": "observed_agreement", "value": observed_agreement, "notes": "Fraction of exact matches between rater 1 and rater 2."},
        {"metric": "cohen_kappa", "value": kappa, "notes": "Cohen's kappa across the three valence classes when double-rated clips exist."},
    ]
    return resolved_rows, summary_rows, pair_rows


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_rows = read_csv_rows(args.labels)
    resolved_rows, summary_rows, pair_rows = build_outputs(label_rows)

    write_csv_rows(
        output_dir / "clip_labels.csv",
        resolved_rows,
        [
            "clip_id",
            "split",
            "video_file",
            "meeting_id",
            "camera",
            "clip_start_s",
            "clip_end_s",
            "gold_label",
            "label_source",
            "exclude_from_gold",
            "agreement_status",
            "rater_1_label",
            "rater_2_label",
            "adjudicated_label",
            "annotator",
            "adjudicator",
            "notes",
        ],
    )
    write_csv_rows(
        output_dir / "interrater_agreement.csv",
        summary_rows,
        ["metric", "value", "notes"],
    )
    if pair_rows:
        write_csv_rows(
            output_dir / "interrater_pairs.csv",
            pair_rows,
            ["clip_id", "split", "rater_1_label", "rater_2_label", "agreement_status"],
        )

    print(f"Wrote resolved labels to {output_dir / 'clip_labels.csv'}")
    print(f"Wrote agreement summary to {output_dir / 'interrater_agreement.csv'}")


if __name__ == "__main__":
    main()
