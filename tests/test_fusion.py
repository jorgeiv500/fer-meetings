import importlib.util
import unittest


HAS_NUMPY = importlib.util.find_spec("numpy") is not None

if HAS_NUMPY:
    from fer_meetings.fusion import (
        concatenate_clip_feature_rows,
        entropy_weighted_probability_fusion,
    )


@unittest.skipUnless(HAS_NUMPY, "numpy is required for fusion unit tests")
class FusionTests(unittest.TestCase):
    def test_entropy_weighted_fusion_prefers_low_entropy_predictions(self):
        fused, weights = entropy_weighted_probability_fusion(
            [
                [0.90, 0.05, 0.05],
                [0.34, 0.33, 0.33],
            ]
        )
        self.assertGreater(weights[0], weights[1])
        self.assertAlmostEqual(float(sum(weights)), 1.0, places=6)
        self.assertEqual(int(fused.argmax()), 0)

    def test_concatenate_clip_feature_rows_stacks_backbone_embeddings(self):
        rows = [
            {
                "clip_id": "clip_1",
                "split": "dev",
                "meeting_id": "meeting_a",
                "camera": "speaker_A",
                "model_name": "cnn_model",
                "model_family": "cnn",
                "hf_model_id": "cnn/id",
                "mean_embedding_json": "[1,2]",
                "std_embedding_json": "[0.1,0.2]",
                "frame_embeddings_json": "[[1,2],[3,4]]",
                "frame_probabilities_json": "[[0.8,0.1,0.1],[0.7,0.2,0.1]]",
                "face_detected_ratio": "1.0",
                "signed_valence_mean": "0.5",
                "signed_valence_std": "0.1",
                "signed_valence_delta": "0.2",
            },
            {
                "clip_id": "clip_1",
                "split": "dev",
                "meeting_id": "meeting_a",
                "camera": "speaker_A",
                "model_name": "vit_model",
                "model_family": "vit",
                "hf_model_id": "vit/id",
                "mean_embedding_json": "[5,6,7]",
                "std_embedding_json": "[0.3,0.4,0.5]",
                "frame_embeddings_json": "[[5,6,7],[8,9,10]]",
                "frame_probabilities_json": "[[0.1,0.2,0.7],[0.2,0.3,0.5]]",
                "face_detected_ratio": "0.5",
                "signed_valence_mean": "0.1",
                "signed_valence_std": "0.2",
                "signed_valence_delta": "0.3",
            },
        ]
        fused = concatenate_clip_feature_rows(rows)
        self.assertEqual(fused["model_name"], "cnn_vit_fusion")
        self.assertEqual(fused["model_family"], "hybrid")
        self.assertIn("cnn_model + vit_model", fused["component_models"])
        self.assertIn("[1.0,2.0,5.0,6.0,7.0]", fused["mean_embedding_json"])


if __name__ == "__main__":
    unittest.main()
