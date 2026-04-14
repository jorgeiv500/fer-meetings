# Paper Outline

## Working title

Cross-Domain Transfer of Facial Affect Recognition to Meeting Video: Comparing CNN and Vision Transformer Representations on the AMI Corpus

## Research question

How severe is out-of-domain degradation when FER models pretrained on standard facial datasets are transferred to close-up AMI meeting clips, how does that degradation differ between `CNN` and `Vision Transformer` backbones, and how much of the lost signal can be recovered with lightweight temporal aggregation and clip-level supervised adaptation?

## Main hypothesis

1. `Single-frame` performance drops sharply on AMI for any backbone.
2. Temporal aggregation improves stability and `macro-F1` relative to a single-frame prediction.
3. Frozen clip representations are more useful than raw frame-level probabilities for low-cost adaptation.
4. Any `CNN` versus `ViT` advantage should be demonstrated empirically rather than assumed from architecture alone.

## Recommended framing

Do not frame the paper as "we applied an FER model to AMI."

Frame it as:

1. a **domain generalization** study
2. a comparison between visual representation families: `CNN` versus `ViT`
3. a study of **low-cost temporal aggregation** for observable clip-level valence
4. an evaluation of **lightweight, reproducible adaptation** on a small target dataset

## What the current project already supports

The repository already implements a reproducible pilot:

- an AMI clip manifest
- frame-level inference with Hugging Face models
- simple temporal aggregation
- evaluation against `gold_label`
- lightweight calibration on `dev`

The `HfEmotionClassifier` class already uses `AutoImageProcessor` and `AutoModelForImageClassification`, so the same pipeline can load both `CNN` and `ViT` backbones without redesign.

## Technical reframing

### Track A. Direct transfer baselines

- `A0`: center-frame `single-frame`
- `A1`: probability `mean pooling`
- `A2`: `majority vote`
- `A3`: `temperature scaling` or multinomial calibration on `dev`

These are the mandatory baselines because they preserve the current repository logic and make the rest of the paper interpretable.

### Track B. Backbone-family comparison

Compare at least two source feature extractors:

- `CNN-FER`: for example `ResNet`, `EfficientNet`, or `ConvNeXt` fine-tuned for facial expression
- `ViT-FER`: for example `ViT` or `DeiT` fine-tuned for facial expression

The comparison should go beyond final accuracy and inspect:

- out-of-domain robustness
- confidence and calibration
- temporal stability
- sensitivity to pose, blur, occlusion, and subtle expressivity

### Track C. Lightweight clip-level adaptation

Starting from frozen frame embeddings:

- `C0`: `Logistic Regression` over the mean clip embedding
- `C1`: `HistGradientBoosting` or `XGBoost` over aggregated clip features
- `C2`: `Linear probe` over pooled embeddings

This is stronger than staying at frame labels only because it turns the study into a representation-transfer paper without requiring heavy backbone fine-tuning.

### Track D. Strongest extension still within scope

The most interesting extension in scope is not a forced GNN. It is one of these:

1. **Temporal attention pooling / a small temporal transformer**
   - input: embeddings from several frames of the clip
   - output: one clip-level valence prediction
   - value: models subtle facial dynamics better than a plain mean

2. **Explicit domain-shift analysis in representation space**
   - measure `FER2013` versus `AMI` separability in `CNN` and `ViT` embeddings
   - use centroid distances, `MMD`, `CORAL`, or a simple domain classifier as analysis tools
   - value: repositions the paper as a representation study under domain shift

If only one extension is kept, the best mix of interest and feasibility is:

- `CNN` versus `ViT`
- `mean pooling` versus `attention pooling`
- `linear probe` over clip embeddings

## Experimental families

### Family 1. Direct zero-shot transfer

- `E0-CNN`: single frame
- `E0-ViT`: single frame

### Family 2. Temporal aggregation without training

- `E1-CNN`: mean pooling / vote
- `E1-ViT`: mean pooling / vote

### Family 3. Lightweight supervised adaptation

- `E2-CNN`: calibration and/or `linear probe` on clip embeddings
- `E2-ViT`: calibration and/or `linear probe` on clip embeddings

### Family 4. Learned temporal aggregation

- `E3-CNN`: temporal attention pooling over embeddings
- `E3-ViT`: temporal attention pooling over embeddings

## Recommended final scope

To keep the pilot defensible, the paper should close around this core:

1. two source backbones: `CNN` and `ViT`
2. three inference levels
   - single frame
   - simple temporal pooling
   - learned pooling or `linear probe` over clip embeddings
3. three-class human valence labels
4. evaluation grouped by meeting or scenario
5. error analysis around visual quality and subtle expressivity

Do not make these the main focus:

- clip GNNs
- seven discrete emotions
- full backbone fine-tuning on AMI
- audio-video multimodality in the first paper

That would widen the scope too much and weaken the paper's central claim.

## Recommended visual package

### Figures already ready

1. `main_test_macro_f1.png`
   clear horizontal ranking of the main methods on `test`
2. `main_test_scorecard.png`
   compact heatmap of `Macro-F1`, `Balanced Accuracy`, and `Accuracy`
3. `clip_models_macro_f1.png`
   ranking of clip-level supervised adaptation methods
4. `label_distribution.png`
   label distribution by split
5. `interrater_overview.png`
   human agreement overview
6. `selected_confusions.png`
   small confusion-matrix panel for representative methods

### Tables already ready

1. `dataset_summary.csv/md`
2. `interrater_summary.csv/md`
3. `main_model_comparison.csv/md`
4. `clip_model_comparison.csv/md`
5. `label_distribution.csv/md`

### Figures still worth producing manually

1. Full experimental pipeline diagram
2. Visual examples of the `FER2013` versus `AMI` domain shift
3. Qualitative cases for success, failure, and temporal aggregation improvement

### What was intentionally removed from the editorial package

- `ROC/PR` figures with too many series in one plot
- full `confusion_*.png` batteries for every method
- charts with method names too long for the axis
- auxiliary tables that repeated results without improving the story

Editorial rule: each figure should answer one question and remain legible without reading the experiment code.

## Publishable outcomes

- show quantitatively that meeting video breaks standard FER assumptions
- show whether `ViT` is more robust than `CNN`, or not
- show that simple temporal pooling already recovers useful signal
- show whether lightweight adaptation on embeddings beats zero-shot transfer
- report calibration and stability, not just accuracy

## Publishable negative outcomes

If `ViT` does not beat `CNN`, the paper is still useful if it shows that:

- the target dataset is small
- the affect is subtle
- the gain depends more on temporal pooling than on architecture
- the target domain favors local low-resolution facial cues where a `CNN` remains competitive

If temporal attention pooling does not beat mean pooling, that is also useful:

- it suggests that most recoverable signal is already captured by a stable low-cost aggregation rule
- it strengthens the lightweight and reproducible framing

## Concrete code changes that matter

1. Allow multiple `model_id` entries in configuration and record the family (`cnn` / `vit`).
2. Export frame embeddings in addition to probabilities.
3. Save aggregated clip features for lightweight supervised training.
4. Keep a clean separation between
   - backbone inference
   - temporal pooling
   - clip-level supervised adaptation
   - evaluation and calibration
5. Maintain a curated `paper_assets` package with
   - few figures
   - short labels
   - genuine visual variety
   - no redundant outputs

## Current project status

- the `annotation_pack` already supports `rater_1_label`, `rater_2_label`, `adjudicated_label`, and `agreement_status`
- the current set contains `100` double-rated clips
- the best zero-shot method on `test` is still `ViT | Single frame`
- the best overall project result is `Fusion | Clip LogReg`
- the next priority is paper narrative and qualitative selection, not more benchmarking

## Recommended final claim

The paper should position itself as a reproducible study of cross-domain facial affect transfer, centered on the comparison between `CNN` and `ViT` representations and on the practical value of lightweight temporal aggregation for observable valence inference in meeting video.

## Claims to avoid

Do not claim:

- that the system detects deep internal emotional states
- that it is ready for real-world deployment
- that `ViT`, or any architecture, is universally superior
- that a marginal performance gain implies contextual understanding of the meeting

## Recommended manuscript order

### Introduction

Open with three points:

1. FER models trained on standard datasets degrade sharply when moved to meeting video
2. it is unclear whether a `ViT` advantage over `CNN` survives this domain shift
3. lightweight temporal aggregation and small supervised adaptation can recover part of the signal without heavy fine-tuning

Do not place figures here. The introduction should sell the problem and the question.

### Data and annotation

Use first:

1. `dataset_summary.csv/md`
2. `label_distribution.png`
3. `interrater_summary.csv/md`
4. `interrater_overview.png`

Section message:

- the dataset is small but controlled and double-rated
- the split distribution is readable and acceptable for a pilot
- human agreement is strong enough that annotation noise is not the only bottleneck

Base captions:

- **Figure. `label_distribution.png`**: Valence label distribution in `dev` and `test`, showing moderate class balance and comparable split sizes.
- **Figure. `interrater_overview.png`**: Summary of agreement between the two human raters, including agreement-disagreement counts and global consistency metrics.
- **Table. `dataset_summary.csv/md`**: Summary of the AMI close-up subset used in the study, with clip counts, split sizes, and annotation coverage.
- **Table. `interrater_summary.csv/md`**: Interrater agreement metrics over the double-labeled clips.

### Experimental protocol

Describe the three method families:

1. direct transfer with `single-frame`
2. lightweight temporal aggregation without strong training
3. clip-level supervised adaptation with frozen embeddings

If the pipeline diagram is later produced, this is where it belongs.

### Main results

Recommended order:

1. `main_test_macro_f1.png`
2. `main_test_scorecard.png`
3. `main_model_comparison.csv/md`

Section message:

- first compare method families
- then show that the ranking should be checked against several metrics
- finish with the table so the reader can recover exact values

Base captions:

- **Figure. `main_test_macro_f1.png`**: Ranking of the main methods on `test` by `Macro-F1`, highlighting domain-shift degradation and the differences between `CNN` and `ViT`.
- **Figure. `main_test_scorecard.png`**: Compact comparison of `Macro-F1`, `Balanced Accuracy`, and `Accuracy` for the main methods on `test`.
- **Table. `main_model_comparison.csv/md`**: Full quantitative results for the main methods on the `test` split.

### Clip-level adaptation results

Recommended order:

1. `clip_models_macro_f1.png`
2. `clip_model_comparison.csv/md`

Section message:

- this is where the largest gain in the project appears
- emphasize that the jump does not come from the backbone alone, but from clip representation and lightweight adaptation

Base captions:

- **Figure. `clip_models_macro_f1.png`**: Performance of clip-level supervised models, showing that lightweight adaptation on embeddings clearly outperforms direct transfer.
- **Table. `clip_model_comparison.csv/md`**: Detailed comparison of clip-level adaptation and fusion models.

### Error analysis

Use:

1. `selected_confusions.png`

Section message:

- the dominant error is the boundary between `neutral` and the extremes
- the figure should support a short discussion, not become a confusion-matrix gallery

Base caption:

- **Figure. `selected_confusions.png`**: Confusion matrices for representative models, used to identify recurring error patterns under domain shift.

## Final asset sequence

If the manuscript needs to stay compact, the cleanest sequence is:

1. `label_distribution.png`
2. `interrater_overview.png`
3. `main_test_macro_f1.png`
4. `main_test_scorecard.png`
5. `clip_models_macro_f1.png`
6. `selected_confusions.png`

And the tables:

1. `dataset_summary.csv/md`
2. `interrater_summary.csv/md`
3. `main_model_comparison.csv/md`
4. `clip_model_comparison.csv/md`

That keeps the story simple: data and agreement, main results, gain from clip adaptation, and error analysis.
