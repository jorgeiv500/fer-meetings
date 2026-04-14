import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from fer_meetings.github_bundle import bundle_paths, copy_bundle


class GitHubBundleTests(unittest.TestCase):
    def test_bundle_paths_include_core_repo_files(self):
        paths = bundle_paths()
        self.assertIn(".github", paths)
        self.assertIn("README.md", paths)
        self.assertIn("docs", paths)
        self.assertIn("results/ami_av_publication", paths)
        self.assertTrue(all(not Path(path).is_absolute() for path in paths))

    def test_copy_bundle_copies_expected_tree(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "source"
            output = Path(temp_dir) / "bundle"
            (root / "docs").mkdir(parents=True)
            (root / "results" / "ami_av_publication").mkdir(parents=True)
            (root / "README.md").write_text("root readme", encoding="utf-8")
            (root / "docs" / "reproducibility.md").write_text("docs", encoding="utf-8")
            (root / "results" / "ami_av_publication" / "summary.md").write_text("summary", encoding="utf-8")

            copy_bundle(
                root_dir=root,
                output_dir=output,
                relative_paths=("README.md", "docs", "results/ami_av_publication"),
                force=False,
            )

            self.assertEqual((output / "README.md").read_text(encoding="utf-8"), "root readme")
            self.assertEqual((output / "docs" / "reproducibility.md").read_text(encoding="utf-8"), "docs")
            self.assertEqual(
                (output / "results" / "ami_av_publication" / "summary.md").read_text(encoding="utf-8"),
                "summary",
            )


if __name__ == "__main__":
    unittest.main()
