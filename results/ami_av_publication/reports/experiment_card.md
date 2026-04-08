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

## Best Main Result

- Model: `vit_face_expression`
- Scope: `test`
- Method: `single_frame`
- Macro-F1: `0.4552`
- Balanced accuracy: `0.4472`

## Best Clip-Level Result

- Model: `vit_face_expression`
- Method: `attention_pooling`
- Macro-F1: `0.6033`
