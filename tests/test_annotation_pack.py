import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from fer_meetings.annotation_pack import build_prediction_index, render_html, summarize_predictions


class AnnotationPackTests(unittest.TestCase):
    def test_build_prediction_index_sorts_by_confidence(self):
        rows = [
            {
                "clip_id": "clip_1",
                "model_name": "vit_a",
                "smoothed_label": "neutral",
                "smoothed_negative_prob": "0.200",
                "smoothed_neutral_prob": "0.500",
                "smoothed_positive_prob": "0.300",
                "face_detected_ratio": "0.500",
            },
            {
                "clip_id": "clip_1",
                "model_name": "cnn_a",
                "smoothed_label": "negative",
                "smoothed_negative_prob": "0.800",
                "smoothed_neutral_prob": "0.100",
                "smoothed_positive_prob": "0.100",
                "face_detected_ratio": "1.000",
            },
        ]
        index = build_prediction_index(rows)
        self.assertEqual(index["clip_1"][0]["model_name"], "cnn_a")
        self.assertEqual(index["clip_1"][0]["smoothed_label"], "negative")

    def test_summarize_predictions_uses_best_prediction(self):
        predictions = [
            {"model_name": "cnn_a", "smoothed_label": "negative", "confidence": 0.8, "face_detected_ratio": "1.000"},
            {"model_name": "vit_a", "smoothed_label": "neutral", "confidence": 0.5, "face_detected_ratio": "0.500"},
        ]
        summary = summarize_predictions(predictions)
        self.assertEqual(summary["suggested_label"], "negative")
        self.assertEqual(summary["suggested_confidence"], "0.800")
        self.assertIn("cnn_a=negative (0.800)", summary["suggestion_summary"])

    def test_render_html_includes_interactive_label_controls(self):
        rows = [
            {
                "clip_id": "clip_1",
                "split": "dev",
                "video_file": "clip_1-video.mp4",
                "meeting_id": "meeting_a",
                "camera": "speaker_A",
                "clip_start_s": "0.000",
                "clip_end_s": "3.500",
                "video_path": "/tmp/clip_1-video.mp4",
                "thumbnail_path": "thumbnails/clip_1.jpg",
                "rater_1_label": "",
                "rater_2_label": "",
                "adjudicated_label": "",
                "gold_label": "neutral",
                "annotator": "tester",
                "adjudicator": "",
                "exclude_from_gold": "false",
                "agreement_status": "",
                "notes": "example",
                "suggested_label": "negative",
                "suggested_confidence": "0.800",
                "face_detected_ratio": "1.000",
                "suggestion_summary": "cnn_a=negative (0.800)",
            }
        ]
        with TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "index.html"
            render_html(rows, Path(temp_dir), output_path)
            html_text = output_path.read_text(encoding="utf-8")

        self.assertIn("Guardar CSV", html_text)
        self.assertIn("data-action=\"set-label\"", html_text)
        self.assertIn("window.showSaveFilePicker", html_text)
        self.assertIn("\"clip_id\": \"clip_1\"", html_text)


if __name__ == "__main__":
    unittest.main()
