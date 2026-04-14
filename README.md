# fer-meetings

Reproducible pilot study on cross-domain transfer of facial affect recognition from standard FER backbones to AMI meeting video.

The canonical experiment compares a `CNN` backbone and a `Vision Transformer` under the same pipeline:

- `single-frame` inference
- lightweight temporal pooling (`smoothed`, `vote`)
- dev-set calibration
- clip-level adaptation on frozen embeddings
- `CNN+ViT` fusion at probability and representation level

The point of the repository is not to claim a universal winner between `CNN` and `ViT`. The point is to measure how much performance breaks under domain shift and how much of that signal can be recovered with cheap temporal aggregation and light supervised adaptation.

## Canonical experiment

- Config: [configs/ami_av_cnn_vit_publication.json](configs/ami_av_cnn_vit_publication.json)
- Manifest: [data/interim/ami_av_manifest_publication.csv](data/interim/ami_av_manifest_publication.csv)
- Canonical run: [results/ami_av_publication](results/ami_av_publication)
- Paper-facing assets: [results/ami_av_publication/paper_assets](results/ami_av_publication/paper_assets)
- Writing status: [docs/paper_progress.md](docs/paper_progress.md)
- Reproducibility guide: [docs/reproducibility.md](docs/reproducibility.md)

## What this repository evaluates

- Cross-domain transfer from FER2013-style training sources to AMI close-up meeting clips.
- Representation differences between `CNN` and `ViT` backbones under the same three-class valence mapping.
- Whether temporal aggregation helps beyond a single center frame.
- Whether clip-level adaptation on frozen embeddings recovers more signal than zero-shot frame predictions alone.

## Repository layout

- `src/fer_meetings/`: experiment code, reporting utilities, annotation pack, and bundle export.
- `tests/`: unit tests for configuration, temporal logic, annotation tooling, fusion, and clip models.
- `configs/`: canonical experiment configuration and label mapping.
- `docs/`: annotation instructions, paper outline, writing status, and reproducibility notes.
- `data/interim/`: versioned manifest for the canonical AMI subset.
- `results/ami_av_publication/`: compact versioned outputs from the canonical run.

## What is versioned

The repository keeps the files needed to inspect, reproduce, and write up the canonical run:

- source code, tests, configuration, and GitHub workflow files
- the canonical AMI manifest
- the HTML annotation pack and review sheet
- compact evaluation metrics and resolved clip labels
- paper figures, tables, and short experiment reports

The repository does not keep bulky artifacts that are easy to regenerate:

- downloaded AMI videos
- downloaded FER snapshots
- full frame-level dumps such as `frame_details.csv`
- full clip feature dumps such as `clip_features.csv`
- duplicate zip archives and exhaustive plotting exports

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
make test
```

## End-to-end reproduction

### 1. Fetch the source FER data

```bash
make fetch-fer
```

This downloads the `Jeneral/fer-2013` dataset snapshot into `data/raw/hf/fer2013`.

### 2. Fetch a small AMI AV subset and build the manifest

```bash
make fetch-ami-av-subset
make manifest-ami-av
```

The resulting manifest is written to `data/interim/ami_av_manifest_publication.csv`.

### 3. Run the pre-label stage

```bash
make experiment-prelabels
```

This stage produces:

- `predictions.csv`
- `frame_details.csv`
- `clip_features.csv`
- `annotation_pack/index.html`
- `annotation_pack/annotation_sheet.csv`

### 4. Review and annotate clips

Open [results/ami_av_publication/annotation_pack/index.html](results/ami_av_publication/annotation_pack/index.html) in a browser.

The annotation interface lets you:

- set `gold_label`
- inspect `rater_1_label`, `rater_2_label`, `adjudicated_label`, and `agreement_status`
- filter by split, label status, and free text
- export an updated `annotation_sheet.csv`

If you export a revised sheet manually, place it back at:

- [results/ami_av_publication/annotation_pack/annotation_sheet.csv](results/ami_av_publication/annotation_pack/annotation_sheet.csv)

### 5. Run the post-label stage

```bash
make experiment-postlabels
```

This stage runs:

- interrater agreement
- scenario split generation
- main evaluation and calibration
- clip-level models
- paper tables and figures
- reproducibility reports

## Main outputs

Core tables:

- [main_model_comparison.csv](results/ami_av_publication/paper_assets/tables/main_model_comparison.csv)
- [clip_model_comparison.csv](results/ami_av_publication/paper_assets/tables/clip_model_comparison.csv)
- [dataset_summary.csv](results/ami_av_publication/paper_assets/tables/dataset_summary.csv)
- [label_distribution.csv](results/ami_av_publication/paper_assets/tables/label_distribution.csv)
- [interrater_summary.csv](results/ami_av_publication/paper_assets/tables/interrater_summary.csv)

Core figures:

- [main_test_macro_f1.png](results/ami_av_publication/paper_assets/figures/main_test_macro_f1.png)
- [main_test_scorecard.png](results/ami_av_publication/paper_assets/figures/main_test_scorecard.png)
- [clip_models_macro_f1.png](results/ami_av_publication/paper_assets/figures/clip_models_macro_f1.png)
- [label_distribution.png](results/ami_av_publication/paper_assets/figures/label_distribution.png)
- [interrater_overview.png](results/ami_av_publication/paper_assets/figures/interrater_overview.png)
- [selected_confusions.png](results/ami_av_publication/paper_assets/figures/selected_confusions.png)

Core reports:

- [experiment_card.md](results/ami_av_publication/reports/experiment_card.md)
- [data_sheet.md](results/ami_av_publication/reports/data_sheet.md)
- [limitations_and_ethics.md](results/ami_av_publication/reports/limitations_and_ethics.md)
- [reproducibility_checklist.md](results/ami_av_publication/reports/reproducibility_checklist.md)

## Current headline results

- Best zero-shot main result on `test`: `ViT | single_frame`
  `Macro-F1 = 0.4552`, `Balanced accuracy = 0.4472`
- Best clip-level result on `test`: `Fusion | mean_embedding_logreg`
  `Macro-F1 = 0.6244`, `Balanced accuracy = 0.6944`
- Human agreement on the double-rated set:
  `observed agreement = 0.7700`, `Cohen's kappa = 0.6175`

## GitHub-ready export

To create a clean standalone folder that can be uploaded as a repository:

```bash
make github-bundle
```

That command creates `build/github_repo/` with the English documentation, source tree, tests, canonical config, versioned manifest, and compact canonical results.

## Short command list

```bash
make fetch-fer
make fetch-ami-av-subset
make manifest-ami-av
make experiment-prelabels
make experiment-postlabels
make github-bundle
make test
```
