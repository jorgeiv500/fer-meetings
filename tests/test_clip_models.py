import importlib.util
import unittest


HAS_NUMPY = importlib.util.find_spec("numpy") is not None

if HAS_NUMPY:
    import numpy as np

    from fer_meetings.clip_models import metric_bundle, pad_sequences


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


if __name__ == "__main__":
    unittest.main()
