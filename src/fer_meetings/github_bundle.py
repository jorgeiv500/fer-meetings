import argparse
import shutil
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE_PATHS = (
    ".github",
    ".gitignore",
    "Makefile",
    "README.md",
    "pyproject.toml",
    "configs",
    "docs",
    "src",
    "tests",
    "data/raw/.gitkeep",
    "data/interim/.gitkeep",
    "data/interim/ami_av_manifest_publication.csv",
    "data/annotations/.gitkeep",
    "results/.gitkeep",
    "results/ami_av_publication",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Build a GitHub-ready export bundle for the fer-meetings repository.")
    parser.add_argument("--output-dir", default="build/github_repo", help="Destination directory for the exported bundle.")
    parser.add_argument("--force", action="store_true", help="Overwrite the destination directory if it already exists.")
    return parser.parse_args()


def bundle_paths():
    return DEFAULT_BUNDLE_PATHS


def copy_bundle(root_dir, output_dir, relative_paths=None, force=False):
    root_dir = Path(root_dir)
    output_dir = Path(output_dir)
    relative_paths = tuple(relative_paths or DEFAULT_BUNDLE_PATHS)

    if output_dir.exists():
        has_contents = any(output_dir.iterdir())
        if has_contents and not force:
            raise RuntimeError(f"Output directory already exists and is not empty: {output_dir}")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    for relative_path in relative_paths:
        relative = Path(relative_path)
        if relative.is_absolute():
            raise ValueError(f"Bundle path must be relative, got: {relative_path}")

        source = root_dir / relative
        if not source.exists():
            raise FileNotFoundError(f"Bundle source does not exist: {source}")

        destination = output_dir / relative
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

    return output_dir


def main():
    args = parse_args()
    output_dir = copy_bundle(
        root_dir=REPOSITORY_ROOT,
        output_dir=args.output_dir,
        relative_paths=bundle_paths(),
        force=args.force,
    )
    print(f"Wrote GitHub bundle to {output_dir}")


if __name__ == "__main__":
    main()
