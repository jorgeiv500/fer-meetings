# Reproducibility Guide

## Scope

This document describes how to reproduce the canonical `CNN` versus `ViT` AMI transfer experiment from a clean checkout.

The repository already includes:

- the canonical config
- the canonical AMI manifest
- compact metrics and paper assets from the canonical run
- unit tests for the core experiment components

The repository does not include downloaded AMI videos or downloaded FER snapshots. Those must be fetched locally.

## Environment

- Python `>=3.9`
- network access for Hugging Face downloads
- enough disk space for the local AMI subset and model caches
- optional GPU or Apple `mps`; CPU also works for small runs

Set up the environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Sanity-check the local checkout:

```bash
make test
```

## Canonical files

- Config: [../configs/ami_av_cnn_vit_publication.json](../configs/ami_av_cnn_vit_publication.json)
- Manifest: [../data/interim/ami_av_manifest_publication.csv](../data/interim/ami_av_manifest_publication.csv)
- Canonical outputs: [../results/ami_av_publication](../results/ami_av_publication)

## Data preparation

### Fetch FER2013 from Hugging Face

```bash
make fetch-fer
```

This stores a dataset snapshot under `data/raw/hf/fer2013`.

### Fetch a small AMI AV subset

```bash
make fetch-ami-av-subset
```

This streams a compact subset from `hhoangphuoc/ami-av` into:

- `data/raw/hf/ami_av/video_segments/original_videos`

### Build the AMI manifest

```bash
make manifest-ami-av
```

This writes the canonical manifest to:

- `data/interim/ami_av_manifest_publication.csv`

## Experiment stages

### Stage 1. Pre-label run

```bash
make experiment-prelabels
```

Expected outputs under `results/ami_av_publication`:

- `predictions.csv`
- `frame_details.csv`
- `clip_features.csv`
- `annotation_pack/index.html`
- `annotation_pack/annotation_sheet.csv`

### Stage 2. Human annotation

Open the generated annotation pack:

- [../results/ami_av_publication/annotation_pack/index.html](../results/ami_av_publication/annotation_pack/index.html)

Recommended workflow:

1. fill `rater_1_label`
2. fill `rater_2_label`
3. set `adjudicated_label` for disagreements when needed
4. keep `gold_label` only for final resolved labels

The repository-wide annotation guidance is in:

- [annotation_guidelines.md](annotation_guidelines.md)

### Stage 3. Post-label run

```bash
make experiment-postlabels
```

This generates:

- interrater agreement summaries
- scenario splits
- main evaluation metrics
- clip-level model metrics
- paper assets
- experiment reports

## What to compare against the canonical run

Main references:

- [../results/ami_av_publication/summary.md](../results/ami_av_publication/summary.md)
- [../results/ami_av_publication/reports/experiment_card.md](../results/ami_av_publication/reports/experiment_card.md)
- [../results/ami_av_publication/paper_assets/tables/main_model_comparison.csv](../results/ami_av_publication/paper_assets/tables/main_model_comparison.csv)
- [../results/ami_av_publication/paper_assets/tables/clip_model_comparison.csv](../results/ami_av_publication/paper_assets/tables/clip_model_comparison.csv)

Expected headline values from the current canonical run:

- `ViT | single_frame | test`: `Macro-F1 = 0.4552`
- `Fusion | mean_embedding_logreg | test`: `Macro-F1 = 0.6244`
- human agreement: `observed agreement = 0.7700`, `Cohen's kappa = 0.6175`

Small numerical differences can appear if dependencies, model revisions, or the local AMI subset differ.

## Bundle export

To create a clean standalone folder suitable for GitHub upload:

```bash
make github-bundle
```

That writes a curated export to:

- `build/github_repo`

The export includes English documentation, source code, tests, configuration, the versioned manifest, the compact canonical outputs, and the `.github` workflow files.
