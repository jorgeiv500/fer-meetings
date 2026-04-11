import importlib.util
import unittest


HAS_NUMPY = importlib.util.find_spec("numpy") is not None

if HAS_NUMPY:
    import numpy as np
    import torch

    from fer_meetings.clip_models import coral_align_source_to_target, metric_bundle, pad_sequences, rbf_mmd


@unittest.skipUnless(HAS_NUMPY, "numpy is required for clip-model unit tests")
class ClipModelTests(unittest.TestCase):
    def test_pad_sequences(self):
        inputs, mask = pad_sequences(
            [
                np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                np.array([[5.0, 6.0]], dtype=np.float32),
            ]
        )
        self.assertEqual(tuple(inputs.shape), (2, 2, 2))
        self.assertEqual(tuple(mask.shape), (2, 2))
        self.assertTrue(bool(mask[0, 1]))
        self.assertFalse(bool(mask[1, 1]))

    def test_metric_bundle(self):
        metrics = metric_bundle(np.array([0, 1, 2]), np.array([0, 1, 1]))
        self.assertEqual(metrics["n_clips"], 3)
        self.assertGreaterEqual(metrics["macro_f1"], 0.0)
        self.assertLessEqual(metrics["macro_f1"], 1.0)

    def test_coral_align_source_to_target_preserves_shape(self):
        source = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]], dtype=np.float32)
        target = np.array([[3.0, 2.0], [2.5, 2.5], [2.0, 3.0]], dtype=np.float32)
        aligned = coral_align_source_to_target(source, target, regularization=1e-2)
        self.assertEqual(aligned.shape, source.shape)
        self.assertTrue(np.isfinite(aligned).all())

    def test_rbf_mmd_is_non_negative(self):
        source = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)
        target = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)
        value = rbf_mmd(source, target).item()
        self.assertGreaterEqual(value, -1e-6)


if __name__ == "__main__":
    unittest.main()
