import argparse
import html
import os
from collections import defaultdict
from pathlib import Path

from fer_meetings.temporal import sample_frame_times
from fer_meetings.utils import ensure_parent, read_csv_rows, write_csv_rows
from fer_meetings.video import open_video, read_frame_at


def parse_args():
    parser = argparse.ArgumentParser(description="Build an annotation pack with thumbnails and local HTML preview.")
    parser.add_argument("--manifest", required=True, help="Input manifest CSV.")
    parser.add_argument(
        "--predictions",
        default="",
        help="Optional predictions CSV used only for weak suggestions in a separate column.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for HTML, thumbnails and CSV review sheet.")
    parser.add_argument(
        "--labels-output",
        default="",
        help="Optional path for the annotation CSV. Defaults to <output-dir>/annotation_sheet.csv.",
    )
    parser.add_argument(
        "--frames-per-clip",
        type=int,
        default=3,
        help="Number of frames used in each thumbnail strip.",
    )
    parser.add_argument(
        "--thumb-height",
        type=int,
        default=160,
        help="Per-frame height in pixels for the thumbnail strip.",
    )
    return parser.parse_args()


def parse_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_prediction_index(prediction_rows):
    grouped = defaultdict(list)
    for row in prediction_rows:
        confidence = max(
            parse_float(row.get("smoothed_negative_prob")),
            parse_float(row.get("smoothed_neutral_prob")),
            parse_float(row.get("smoothed_positive_prob")),
        )
        grouped[row["clip_id"]].append(
            {
                "model_name": row.get("model_name", ""),
                "smoothed_label": row.get("smoothed_label", ""),
                "vote_label": row.get("vote_label", ""),
                "confidence": confidence,
                "face_detected_ratio": row.get("face_detected_ratio", ""),
            }
        )

    for clip_id in grouped:
        grouped[clip_id] = sorted(
            grouped[clip_id],
            key=lambda item: (-item["confidence"], item["model_name"]),
        )
    return grouped


def summarize_predictions(predictions):
    if not predictions:
        return {
            "suggested_label": "",
            "suggested_confidence": "",
            "suggestion_summary": "",
            "face_detected_ratio": "",
        }

    best = predictions[0]
    summary = "; ".join(
        f"{item['model_name']}={item['smoothed_label']} ({item['confidence']:.3f})"
        for item in predictions
    )
    return {
        "suggested_label": best["smoothed_label"],
        "suggested_confidence": f"{best['confidence']:.3f}",
        "suggestion_summary": summary,
        "face_detected_ratio": best["face_detected_ratio"],
    }


def annotate_frame(frame_bgr, timestamp_s, label):
    import cv2

    overlay = frame_bgr.copy()
    text = f"{label} {timestamp_s:.2f}s"
    cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(
        overlay,
        text,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return overlay


def make_thumbnail_strip(video_path, clip_start_s, clip_end_s, output_path, frames_per_clip, thumb_height):
    import cv2

    capture = open_video(video_path)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        capture.release()
        raise RuntimeError(f"Invalid FPS for video: {video_path}")

    thumbs = []
    timestamps = sample_frame_times(float(clip_start_s), float(clip_end_s), frames_per_clip)
    for index, timestamp_s in enumerate(timestamps, start=1):
        frame = read_frame_at(capture, fps, timestamp_s)
        if frame is None:
            continue
        scale = thumb_height / max(frame.shape[0], 1)
        width = max(1, int(round(frame.shape[1] * scale)))
        frame = cv2.resize(frame, (width, thumb_height), interpolation=cv2.INTER_AREA)
        frame = annotate_frame(frame, timestamp_s, f"f{index}")
        thumbs.append(frame)

    capture.release()
    if not thumbs:
        raise RuntimeError(f"Could not extract thumbnail frames for: {video_path}")

    strip = cv2.hconcat(thumbs)
    ensure_parent(output_path)
    ok = cv2.imwrite(str(output_path), strip)
    if not ok:
        raise RuntimeError(f"Could not write thumbnail strip: {output_path}")


def relative_posix_path(path, start):
    return Path(os.path.relpath(path, start)).as_posix()


def build_annotation_rows(
    manifest_rows,
    prediction_index,
    existing_labels,
    output_dir,
    thumbnails_dir,
    frames_per_clip,
    thumb_height,
):
    rows = []
    for row in manifest_rows:
        suggestions = summarize_predictions(prediction_index.get(row["clip_id"], []))
        existing = existing_labels.get(row["clip_id"], {})
        thumbnail_path = thumbnails_dir / f"{row['clip_id']}.jpg"
        make_thumbnail_strip(
            video_path=row["video_path"],
            clip_start_s=row["clip_start_s"],
            clip_end_s=row["clip_end_s"],
            output_path=thumbnail_path,
            frames_per_clip=frames_per_clip,
            thumb_height=thumb_height,
        )

        rows.append(
            {
                "clip_id": row["clip_id"],
                "split": row["split"],
                "video_file": row["video_file"],
                "meeting_id": row["meeting_id"],
                "camera": row["camera"],
                "clip_start_s": row["clip_start_s"],
                "clip_end_s": row["clip_end_s"],
                "video_path": row["video_path"],
                "thumbnail_path": relative_posix_path(thumbnail_path, output_dir),
                "rater_1_label": existing.get("rater_1_label", ""),
                "rater_2_label": existing.get("rater_2_label", ""),
                "adjudicated_label": existing.get("adjudicated_label", ""),
                "gold_label": existing.get("gold_label", ""),
                "annotator": existing.get("annotator", ""),
                "adjudicator": existing.get("adjudicator", ""),
                "exclude_from_gold": existing.get("exclude_from_gold", ""),
                "agreement_status": existing.get("agreement_status", ""),
                "notes": existing.get("notes", ""),
                "suggested_label": suggestions["suggested_label"],
                "suggested_confidence": suggestions["suggested_confidence"],
                "face_detected_ratio": suggestions["face_detected_ratio"],
                "suggestion_summary": suggestions["suggestion_summary"],
            }
        )
    return rows


def render_html(rows, output_dir, output_path):
    row_html = []
    for row in rows:
        video_rel = relative_posix_path(row["video_path"], output_dir)
        thumb_rel = row["thumbnail_path"]
        row_html.append(
            "<tr>"
            f"<td>{html.escape(row['clip_id'])}</td>"
            f"<td>{html.escape(row['split'])}</td>"
            f"<td>{html.escape(row['meeting_id'])}</td>"
            f"<td>{html.escape(row['camera'])}</td>"
            f"<td>{html.escape(row['clip_start_s'])} - {html.escape(row['clip_end_s'])}</td>"
            f"<td><img src=\"{html.escape(thumb_rel)}\" alt=\"thumb\" loading=\"lazy\"></td>"
            f"<td><video controls preload=\"metadata\" src=\"{html.escape(video_rel)}\"></video></td>"
            f"<td>{html.escape(row['suggested_label'])}</td>"
            f"<td>{html.escape(row['suggested_confidence'])}</td>"
            f"<td>{html.escape(row['face_detected_ratio'])}</td>"
            f"<td>{html.escape(row['suggestion_summary'])}</td>"
            "</tr>"
        )

    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>FER Meetings Annotation Pack</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; color: #111; }}
    h1 {{ margin: 0 0 8px; }}
    p {{ margin: 0 0 16px; max-width: 900px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
    th {{ background: #f5f5f5; position: sticky; top: 0; }}
    img {{ max-width: 480px; height: auto; display: block; }}
    video {{ width: 320px; max-height: 220px; display: block; }}
    code {{ background: #f4f4f4; padding: 1px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Annotation Pack</h1>
  <p>
    Use this page only to review clips and record labels in <code>annotation_sheet.csv</code>.
    The columns <code>suggested_label</code> and <code>suggestion_summary</code> are weak model hints and must not replace human judgment.
    For publication-quality annotation, prefer <code>rater_1_label</code>, <code>rater_2_label</code> and <code>adjudicated_label</code>.
  </p>
  <table>
    <thead>
      <tr>
        <th>clip_id</th>
        <th>split</th>
        <th>meeting</th>
        <th>camera</th>
        <th>window</th>
        <th>frames</th>
        <th>video</th>
        <th>suggested_label</th>
        <th>confidence</th>
        <th>face_detected_ratio</th>
        <th>suggestion_summary</th>
      </tr>
    </thead>
    <tbody>
      {''.join(row_html)}
    </tbody>
  </table>
</body>
</html>
"""
    ensure_parent(output_path)
    Path(output_path).write_text(document, encoding="utf-8")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    thumbnails_dir = output_dir / "thumbnails"
    labels_output = Path(args.labels_output) if args.labels_output else output_dir / "annotation_sheet.csv"

    manifest_rows = read_csv_rows(args.manifest)
    prediction_rows = read_csv_rows(args.predictions) if args.predictions else []
    prediction_index = build_prediction_index(prediction_rows)
    existing_labels = {}
    if labels_output.exists():
        existing_labels = {row["clip_id"]: row for row in read_csv_rows(labels_output)}
    annotation_rows = build_annotation_rows(
        manifest_rows,
        prediction_index,
        existing_labels,
        output_dir,
        thumbnails_dir,
        frames_per_clip=args.frames_per_clip,
        thumb_height=args.thumb_height,
    )

    fieldnames = [
        "clip_id",
        "split",
        "video_file",
        "meeting_id",
        "camera",
        "clip_start_s",
        "clip_end_s",
        "video_path",
        "thumbnail_path",
        "rater_1_label",
        "rater_2_label",
        "adjudicated_label",
        "gold_label",
        "annotator",
        "adjudicator",
        "exclude_from_gold",
        "agreement_status",
        "notes",
        "suggested_label",
        "suggested_confidence",
        "face_detected_ratio",
        "suggestion_summary",
    ]
    write_csv_rows(labels_output, annotation_rows, fieldnames)

    html_output = output_dir / "index.html"
    render_html(annotation_rows, output_dir, html_output)

    print(f"Wrote annotation sheet to {labels_output}")
    print(f"Wrote HTML review page to {html_output}")
    print(f"Wrote {len(annotation_rows)} thumbnail strips to {thumbnails_dir}")


if __name__ == "__main__":
    main()
