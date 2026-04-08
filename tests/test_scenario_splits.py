import unittest

from fer_meetings.scenario_splits import build_splits


class ScenarioSplitTests(unittest.TestCase):
    def test_build_splits(self):
        payload = build_splits(
            [
                {"clip_id": "c1", "split": "dev", "meeting_id": "M1", "video_file": "a.mp4"},
                {"clip_id": "c2", "split": "dev", "meeting_id": "M1", "video_file": "b.mp4"},
                {"clip_id": "c3", "split": "test", "meeting_id": "M2", "video_file": "c.mp4"},
            ]
        )
        self.assertEqual(payload["splits"]["dev"]["clip_count"], 2)
        self.assertEqual(payload["splits"]["dev"]["meeting_count"], 1)
        self.assertEqual(payload["clip_to_split"]["c3"], "test")


if __name__ == "__main__":
    unittest.main()
