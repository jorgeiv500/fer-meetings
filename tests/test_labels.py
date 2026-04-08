import unittest

from fer_meetings.labels import (
    canonical_gold_label,
    collapse_probabilities,
    load_label_map,
    normalize_label,
    resolve_gold_label,
)


class LabelMappingTests(unittest.TestCase):
    def test_normalize_label(self):
        self.assertEqual(normalize_label(" Sadness "), "sadness")
        self.assertEqual(normalize_label("Anger!"), "anger")

    def test_canonical_gold_label(self):
        self.assertEqual(canonical_gold_label("NEG"), "negative")
        self.assertEqual(canonical_gold_label("neutral"), "neutral")
        self.assertEqual(canonical_gold_label(""), "")

    def test_resolve_gold_label(self):
        label, source = resolve_gold_label({"gold_label": "positive"})
        self.assertEqual((label, source), ("positive", "gold_label"))

        label, source = resolve_gold_label({"adjudicated_label": "neutral"})
        self.assertEqual((label, source), ("neutral", "adjudicated_label"))

        label, source = resolve_gold_label({"rater_1_label": "neg", "rater_2_label": "negative"})
        self.assertEqual((label, source), ("negative", "rater_agreement"))

    def test_collapse_probabilities(self):
        probabilities = {"Happy": 0.60, "Surprise": 0.20, "Anger": 0.20}
        collapsed = collapse_probabilities(probabilities, load_label_map())
        self.assertAlmostEqual(collapsed["positive"], 0.60)
        self.assertAlmostEqual(collapsed["neutral"], 0.20)
        self.assertAlmostEqual(collapsed["negative"], 0.20)


if __name__ == "__main__":
    unittest.main()
