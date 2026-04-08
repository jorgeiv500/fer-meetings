import unittest

from fer_meetings.temporal import majority_vote, sample_clip_windows, sample_frame_times


class TemporalTests(unittest.TestCase):
    def test_sample_clip_windows(self):
        windows = sample_clip_windows(
            duration_s=40,
            clip_seconds=5,
            stride_seconds=10,
            start_offset_seconds=0,
            max_clips=3,
        )
        self.assertEqual(windows, [(0.0, 5.0), (10.0, 15.0), (20.0, 25.0)])

    def test_sample_frame_times(self):
        timestamps = sample_frame_times(0.0, 3.0, 3)
        self.assertEqual(timestamps, [0.75, 1.5, 2.25])

    def test_majority_vote(self):
        label = majority_vote(["neutral", "neutral", "negative"], ["negative", "neutral", "positive"])
        self.assertEqual(label, "neutral")


if __name__ == "__main__":
    unittest.main()
