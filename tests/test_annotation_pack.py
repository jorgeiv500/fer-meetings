import unittest

from fer_meetings.annotation_pack import build_prediction_index, summarize_predictions


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


if __name__ == "__main__":
    unittest.main()
