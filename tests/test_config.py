import unittest

from fer_meetings.config import infer_model_family, resolve_model_specs, slugify_model_name


class ConfigTests(unittest.TestCase):
    def test_slugify_model_name(self):
        self.assertEqual(slugify_model_name("trpakov/vit-face-expression"), "trpakov_vit_face_expression")

    def test_infer_model_family(self):
        self.assertEqual(infer_model_family("trpakov/vit-face-expression"), "vit")
        self.assertEqual(infer_model_family("org/efficientnet-fer"), "cnn")
        self.assertEqual(infer_model_family("org/custom-model"), "unknown")

    def test_resolve_model_specs_from_models_block(self):
        config = {
            "models": [
                {"name": "cnn_a", "hf_model_id": "org/resnet-fer", "family": "cnn"},
                {"hf_model_id": "org/vit-fer"},
            ]
        }
        specs = resolve_model_specs(config)
        self.assertEqual(len(specs), 2)
        self.assertEqual(specs[0]["name"], "cnn_a")
        self.assertEqual(specs[1]["family"], "vit")

    def test_resolve_model_specs_from_cli(self):
        specs = resolve_model_specs({}, requested_model_ids=["org/resnet-fer", "org/vit-fer"])
        self.assertEqual([spec["family"] for spec in specs], ["cnn", "vit"])


if __name__ == "__main__":
    unittest.main()
