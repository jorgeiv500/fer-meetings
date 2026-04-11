# Experiment Card

- Experiment name: `ami_av_cnn_vit_publication`
- Manifest clips: `100`
- Resolved labeled clips: `100`
- Meetings: `75`
- Source backbones: `convnext_tiny_emotion, vit_face_expression`
- Label distribution: `negative=27`, `neutral=51`, `positive=22`

## Experimental Focus

- Cross-domain transfer from FER2013-style backbones to AMI meeting video.
- Comparison between CNN and Vision Transformer representations.
- Temporal aggregation and clip-level adaptation under weak supervision.
- Hybrid CNN+ViT fusion through probability ensembles and clip-level representation fusion.

## Human Agreement

- Double-rated clips: `100`
- Observed agreement: `0.7700`
- Cohen's kappa: `0.6175`

## Best Main Result

- Model: `vit_face_expression`
- Scope: `test`
- Method: `single_frame`
- Macro-F1: `0.4552`
- Balanced accuracy: `0.4472`

## Best Clip-Level Result

- Model: `cnn_vit_fusion`
- Method: `mean_embedding_logreg`
- Macro-F1: `0.6244`

## Core Paper Assets

- Main ranking figure: `paper_assets/figures/main_test_macro_f1.png`
- Compact scorecard: `paper_assets/figures/main_test_scorecard.png`
- Clip-level ranking: `paper_assets/figures/clip_models_macro_f1.png`
- Human agreement overview: `paper_assets/figures/interrater_overview.png`
- Selected confusion panel: `paper_assets/figures/selected_confusions.png`
