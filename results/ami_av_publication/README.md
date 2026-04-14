# Versioned Results

This directory keeps only the artifacts that make the canonical run inspectable, reproducible, and easy to write up without committing downloaded datasets or heavyweight regenerable dumps.

## Included

- `annotation_pack/`: HTML review package for clip annotation and human validation
- `clip_labels.csv`: resolved clip labels used for evaluation
- `metrics.csv`, `metrics.json`, `confusion_matrices.csv`, `per_class_metrics.csv`: compact main-evaluation outputs
- `clip_models/`: compact metrics and summaries for clip-level adaptation
- `paper_assets/`: curated manuscript figures and tables
- `reports/`: experiment card, data sheet, ethics note, and reproducibility checklist
- `summary.md`: quick overview of the canonical run

## Not included

- downloaded videos and dataset snapshots from Hugging Face
- `frame_details.csv` and `clip_features.csv`
- full frame-level or clip-level prediction dumps
- exhaustive `PR/ROC` exports and duplicate zip archives

## How to regenerate omitted artifacts

From the repository root:

```bash
make fetch-fer
make fetch-ami-av-subset
make manifest-ami-av
make experiment-prelabels
make experiment-postlabels
```

The canonical configuration for this run is [../../configs/ami_av_cnn_vit_publication.json](../../configs/ami_av_cnn_vit_publication.json).
