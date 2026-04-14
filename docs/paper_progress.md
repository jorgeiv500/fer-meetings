# Paper Progress

## Current state

- Canonical run: [results/ami_av_publication](../results/ami_av_publication)
- Annotated dataset: `100` AMI close-up clips
- Splits: `50 dev / 50 test`
- Backbone families: `CNN`, `ViT`
- Evaluated extensions: `temporal pooling`, `calibration`, `clip-level adaptation`, `hybrid fusion`

## Consolidated findings

- Best main result on `test`: `ViT | Single frame`
  - `Macro-F1 = 0.4552`
  - `Balanced accuracy = 0.4472`
- Best clip-level result: `Fusion | Clip LogReg`
  - `Macro-F1 = 0.6244`
  - `Balanced accuracy = 0.6944`
- Human agreement
  - `100` double-rated clips
  - `observed agreement = 0.7700`
  - `Cohen's kappa = 0.6175`

## Methodological readout

- `ViT` is the strongest zero-shot backbone on `test`, but the largest improvement comes from lightweight supervised clip adaptation.
- The standalone `CNN` degrades heavily in zero-shot transfer, although it improves materially after calibration.
- `CNN+ViT` fusion is more useful as a clip representation than as a simple probability ensemble.
- The main failure mode remains the separation of `neutral` from the two extremes, especially in zero-shot settings.

## Final visual package

### Core figures

1. [main_test_macro_f1.png](../results/ami_av_publication/paper_assets/figures/main_test_macro_f1.png)
2. [main_test_scorecard.png](../results/ami_av_publication/paper_assets/figures/main_test_scorecard.png)
3. [clip_models_macro_f1.png](../results/ami_av_publication/paper_assets/figures/clip_models_macro_f1.png)
4. [label_distribution.png](../results/ami_av_publication/paper_assets/figures/label_distribution.png)
5. [interrater_overview.png](../results/ami_av_publication/paper_assets/figures/interrater_overview.png)
6. [selected_confusions.png](../results/ami_av_publication/paper_assets/figures/selected_confusions.png)

### Core tables

1. [main_model_comparison.csv](../results/ami_av_publication/paper_assets/tables/main_model_comparison.csv)
2. [clip_model_comparison.csv](../results/ami_av_publication/paper_assets/tables/clip_model_comparison.csv)
3. [dataset_summary.csv](../results/ami_av_publication/paper_assets/tables/dataset_summary.csv)
4. [label_distribution.csv](../results/ami_av_publication/paper_assets/tables/label_distribution.csv)
5. [interrater_summary.csv](../results/ami_av_publication/paper_assets/tables/interrater_summary.csv)

## What no longer belongs in the paper-facing package

- overloaded `PR/ROC` curves with too many series
- complete `confusion_*.png` galleries for every method
- auxiliary tables that repeat per-class metrics without changing the story
- training-history artifacts that do not support a central paper claim

The raw outputs remain in `results/ami_av_publication` for reproducible inspection, but they are no longer duplicated as manuscript-facing assets.

## Writing backlog

1. Add a schematic figure for the full pipeline.
2. Select three or four qualitative success, failure, and pooling-improvement cases.
3. Decide whether remaining rater disagreements should be fully adjudicated before the final manuscript freeze.
4. Draft the error-analysis section around `selected_confusions.png`.
5. Turn [docs/paper_outline.md](paper_outline.md) into section-ready prose and captions.

## Suggested editorial order

### Main body

1. `dataset_summary.csv/md`
2. `label_distribution.png`
3. `interrater_summary.csv/md`
4. `interrater_overview.png`
5. `main_test_macro_f1.png`
6. `main_test_scorecard.png`
7. `main_model_comparison.csv/md`
8. `clip_models_macro_f1.png`
9. `clip_model_comparison.csv/md`
10. `selected_confusions.png`

### Narrative role by block

- **Data and annotation**: the corpus is small but usable, and human agreement supports the task.
- **Main results**: domain shift hurts zero-shot transfer substantially, and one metric is not enough.
- **Clip adaptation**: most of the recoverable gain appears once frozen clip embeddings are used with lightweight supervision.
- **Error analysis**: `neutral` versus the extremes remains the fragile decision boundary.

## Result directory status

- The only canonical run under `results` is [results/ami_av_publication](../results/ami_av_publication)
- `results/latest_publication` and `results/latest_paper_assets` are symlinks that point to the canonical run
- No additional historical run directories were found that still needed cleanup
