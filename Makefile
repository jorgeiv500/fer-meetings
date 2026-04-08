CONFIG ?= configs/pilot_fast.json
RAW_DIR ?= data/raw
HF_DATA_DIR ?= data/raw/hf
AMI_AV_META ?= data/raw/hf/ami_av_meta/ami-segments-info.csv
AMI_AV_VIDEO_ROOT ?= data/raw/hf/ami_av/video_segments/original_videos
AMI_AV_ROOT ?= data/raw/hf/ami_av
MANIFEST ?= data/interim/ami_manifest.csv
LABELS ?= data/annotations/ami_gold_labels.csv
PREDICTIONS ?= results/pilot/predictions.csv
FRAME_DETAILS ?= results/pilot/frame_details.csv
CLIP_FEATURES ?= results/pilot/clip_features.csv
OUTPUT_DIR ?= results/pilot
PAPER_ASSETS_DIR ?= results/paper_assets

.PHONY: fetch-fer fetch-ami-av-subset manifest manifest-ami-av labels annotation-pack publication-labels publication-reports pilot pilot-ami-av-fast experiment-ami-av-prelabels experiment-ami-av-postlabels evaluate clip-models paper-assets test

fetch-fer:
	fer-fetch-hf-data --dataset-id Jeneral/fer-2013 --output-dir $(HF_DATA_DIR)/fer2013 --snapshot-only --bytes-column img_bytes --label-column labels

fetch-ami-av-subset:
	fer-fetch-ami-av-subset --metadata-csv $(AMI_AV_META) --output-dir $(AMI_AV_ROOT)

manifest:
	fer-build-manifest --config $(CONFIG) --raw-dir $(RAW_DIR) --output $(MANIFEST)

manifest-ami-av:
	fer-build-ami-av-manifest --metadata-csv $(AMI_AV_META) --video-root $(AMI_AV_VIDEO_ROOT) --output $(MANIFEST)

labels:
	fer-make-label-template --manifest $(MANIFEST) --output $(LABELS)

annotation-pack:
	fer-build-annotation-pack --manifest $(MANIFEST) --predictions $(PREDICTIONS) --output-dir $(OUTPUT_DIR)/annotation_pack

publication-labels:
	fer-compute-interrater --labels $(LABELS) --output-dir $(OUTPUT_DIR)
	fer-build-scenario-splits --manifest $(MANIFEST) --output $(OUTPUT_DIR)/scenario_splits.json

pilot:
	fer-run-pilot --config $(CONFIG) --manifest $(MANIFEST) --output $(PREDICTIONS) --frame-details-output $(FRAME_DETAILS) --clip-features-output $(CLIP_FEATURES)

pilot-ami-av-fast:
	fer-run-pilot --config configs/pilot_fast_convnext.json --manifest data/interim/ami_av_manifest.csv --output results/ami_av_pilot_convnext/predictions.csv --frame-details-output results/ami_av_pilot_convnext/frame_details.csv --clip-features-output results/ami_av_pilot_convnext/clip_features.csv --frames-per-clip 3

experiment-ami-av-prelabels:
	fer-run-experiment --config configs/ami_av_cnn_vit.json --manifest data/interim/ami_av_manifest.csv --output-dir results/ami_av_experiment --phase prelabels --frames-per-clip 3

experiment-ami-av-postlabels:
	fer-run-experiment --config configs/ami_av_cnn_vit.json --manifest data/interim/ami_av_manifest.csv --output-dir results/ami_av_experiment --phase postlabels

evaluate:
	fer-evaluate --predictions $(PREDICTIONS) --labels $(LABELS) --output-dir $(OUTPUT_DIR) --fit-calibrator

clip-models:
	fer-train-clip-models --clip-features $(CLIP_FEATURES) --labels $(LABELS) --output-dir $(OUTPUT_DIR)/clip_models

paper-assets:
	fer-make-paper-assets --pilot-dir $(OUTPUT_DIR) --clip-model-dir $(OUTPUT_DIR)/clip_models --manifest $(MANIFEST) --labels $(LABELS) --output-dir $(PAPER_ASSETS_DIR)

publication-reports:
	fer-generate-reporting --config $(CONFIG) --manifest $(MANIFEST) --labels $(OUTPUT_DIR)/clip_labels.csv --pilot-dir $(OUTPUT_DIR) --clip-model-dir $(OUTPUT_DIR)/clip_models --output-dir $(OUTPUT_DIR)/reports

test:
	PYTHONPATH=src python3 -m unittest discover -s tests
