import json

import numpy as np

from fer_meetings.constants import LABEL_ORDER


def parse_json_vector(raw_value):
    values = json.loads(raw_value or "[]")
    return np.array(values, dtype=np.float32)


def parse_json_matrix(raw_value):
    values = json.loads(raw_value or "[]")
    if not values:
        return np.zeros((0, 0), dtype=np.float32)
    return np.array(values, dtype=np.float32)


def parse_probability_matrix(raw_value):
    values = json.loads(raw_value or "[]")
    if not values:
        return np.zeros((0, len(LABEL_ORDER)), dtype=np.float32)
    if isinstance(values[0], dict):
        return np.array(
            [[float(item.get(label, 0.0)) for label in LABEL_ORDER] for item in values],
            dtype=np.float32,
        )
    return np.array(values, dtype=np.float32)


def probability_vector(row, prefix):
    return np.array([float(row[f"{prefix}_{label}_prob"]) for label in LABEL_ORDER], dtype=np.float64)


def label_from_probabilities(probabilities):
    return LABEL_ORDER[int(np.argmax(probabilities))]


def normalized_entropy(probabilities, epsilon=1e-12):
    clipped = np.clip(np.asarray(probabilities, dtype=np.float64), epsilon, 1.0)
    entropy = -np.sum(clipped * np.log(clipped))
    return float(entropy / np.log(clipped.shape[-1]))


def mean_probability_fusion(probability_vectors):
    stacked = np.stack(probability_vectors, axis=0)
    fused = stacked.mean(axis=0)
    fused = fused / fused.sum()
    weights = np.full(stacked.shape[0], 1.0 / stacked.shape[0], dtype=np.float64)
    return fused, weights


def entropy_weighted_probability_fusion(probability_vectors, minimum_weight=0.05):
    stacked = np.stack(probability_vectors, axis=0)
    confidences = np.array(
        [max(1.0 - normalized_entropy(vector), minimum_weight) for vector in stacked],
        dtype=np.float64,
    )
    weights = confidences / confidences.sum()
    fused = np.sum(stacked * weights[:, None], axis=0)
    fused = fused / fused.sum()
    return fused, weights


def sorted_backbone_rows(rows):
    return sorted(
        rows,
        key=lambda row: (row.get("model_family", ""), row.get("model_name", ""), row.get("hf_model_id", "")),
    )


def concatenate_clip_feature_rows(rows):
    ordered_rows = sorted_backbone_rows(rows)
    if len(ordered_rows) < 2:
        raise ValueError("Hybrid clip fusion requires at least two backbone rows for the same clip.")

    mean_vectors = [parse_json_vector(row["mean_embedding_json"]) for row in ordered_rows]
    std_vectors = [parse_json_vector(row.get("std_embedding_json", "")) for row in ordered_rows]
    frame_matrices = [parse_json_matrix(row.get("frame_embeddings_json", "")) for row in ordered_rows]
    probability_matrices = [parse_probability_matrix(row.get("frame_probabilities_json", "")) for row in ordered_rows]

    min_frames = min((matrix.shape[0] for matrix in frame_matrices if matrix.ndim == 2), default=0)
    if min_frames > 0:
        fused_frames = np.concatenate([matrix[:min_frames] for matrix in frame_matrices], axis=1)
    else:
        fused_frames = np.zeros((0, sum(vector.shape[0] for vector in mean_vectors)), dtype=np.float32)

    if probability_matrices and min_frames > 0 and all(matrix.shape[1] == len(LABEL_ORDER) for matrix in probability_matrices):
        fused_frame_probabilities = np.mean([matrix[:min_frames] for matrix in probability_matrices], axis=0)
    else:
        fused_frame_probabilities = np.zeros((0, len(LABEL_ORDER)), dtype=np.float32)

    base_row = dict(ordered_rows[0])
    base_row["model_name"] = "cnn_vit_fusion"
    base_row["model_family"] = "hybrid"
    base_row["hf_model_id"] = " + ".join(row.get("hf_model_id", "") for row in ordered_rows if row.get("hf_model_id", ""))
    base_row["component_models"] = " + ".join(row.get("model_name", "") for row in ordered_rows if row.get("model_name", ""))
    base_row["mean_embedding_json"] = json.dumps(np.concatenate(mean_vectors, axis=0).tolist(), separators=(",", ":"))
    base_row["std_embedding_json"] = json.dumps(np.concatenate(std_vectors, axis=0).tolist(), separators=(",", ":"))
    base_row["frame_embeddings_json"] = json.dumps(fused_frames.tolist(), separators=(",", ":"))
    base_row["frame_probabilities_json"] = json.dumps(fused_frame_probabilities.tolist(), separators=(",", ":"))
    base_row["frames_used"] = str(min_frames)
    base_row["face_detected_ratio"] = f"{np.mean([float(row.get('face_detected_ratio', 0.0) or 0.0) for row in ordered_rows]):.6f}"
    for scalar_key in ["signed_valence_mean", "signed_valence_std", "signed_valence_delta"]:
        numeric_values = [float(row.get(scalar_key, 0.0) or 0.0) for row in ordered_rows]
        base_row[scalar_key] = f"{float(np.mean(numeric_values)):.6f}"
    return base_row
