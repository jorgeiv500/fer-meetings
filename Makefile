CONFIG ?= configs/ami_av_cnn_vit_publication.json
HF_DATA_DIR ?= data/raw/hf
AMI_AV_META ?= data/raw/hf/ami_av_meta/ami-segments-info.csv
AMI_AV_VIDEO_ROOT ?= data/raw/hf/ami_av/video_segments/original_videos
AMI_AV_ROOT ?= data/raw/hf/ami_av
MANIFEST ?= data/interim/ami_av_manifest_publication.csv
OUTPUT_DIR ?= results/ami_av_publication

.PHONY: fetch-fer fetch-ami-av-subset manifest-ami-av experiment-prelabels experiment-postlabels test

fetch-fer:
	fer-fetch-hf-data --dataset-id Jeneral/fer-2013 --output-dir $(HF_DATA_DIR)/fer2013 --snapshot-only --bytes-column img_bytes --label-column labels

fetch-ami-av-subset:
	fer-fetch-ami-av-subset --metadata-csv $(AMI_AV_META) --output-dir $(AMI_AV_ROOT)

manifest-ami-av:
	fer-build-ami-av-manifest --metadata-csv $(AMI_AV_META) --video-root $(AMI_AV_VIDEO_ROOT) --output $(MANIFEST)

experiment-prelabels:
	fer-run-experiment --config $(CONFIG) --manifest $(MANIFEST) --output-dir $(OUTPUT_DIR) --phase prelabels --frames-per-clip 3

experiment-postlabels:
	fer-run-experiment --config $(CONFIG) --manifest $(MANIFEST) --output-dir $(OUTPUT_DIR) --phase postlabels

test:
	PYTHONPATH=src python3 -m unittest discover -s tests
