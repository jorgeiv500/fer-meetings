import json
import re
from pathlib import Path

from fer_meetings.constants import DEFAULT_LABEL_MAP, GOLD_LABEL_ALIASES, LABEL_ORDER


def normalize_label(label):
    if label is None:
        return ""
    return re.sub(r"[^a-z]+", "", str(label).strip().lower())


def canonical_gold_label(label):
    normalized = normalize_label(label)
    return GOLD_LABEL_ALIASES.get(normalized, "")


def resolve_gold_label(row):
    direct = canonical_gold_label(row.get("gold_label", ""))
    if direct:
        return direct, "gold_label"

    adjudicated = canonical_gold_label(row.get("adjudicated_label", ""))
    if adjudicated:
        return adjudicated, "adjudicated_label"

    rater_1 = canonical_gold_label(row.get("rater_1_label", ""))
    rater_2 = canonical_gold_label(row.get("rater_2_label", ""))
    if rater_1 and rater_2 and rater_1 == rater_2:
        return rater_1, "rater_agreement"

    return "", ""


def load_label_map(path=None):
    if path is None:
        mapping = DEFAULT_LABEL_MAP
    else:
        mapping = json.loads(Path(path).read_text(encoding="utf-8"))

    normalized = {normalize_label(key): value.strip().lower() for key, value in mapping.items()}
    invalid_targets = sorted(set(normalized.values()) - set(LABEL_ORDER))
    if invalid_targets:
        raise ValueError(f"Invalid mapped labels: {invalid_targets}")
    return normalized


def collapse_probabilities(raw_probabilities, label_map, label_order=None):
    label_order = label_order or LABEL_ORDER
    collapsed = {label: 0.0 for label in label_order}

    for raw_label, probability in raw_probabilities.items():
        mapped_label = label_map.get(normalize_label(raw_label))
        if mapped_label is None:
            continue
        collapsed[mapped_label] += float(probability)

    total = sum(collapsed.values())
    if total > 0:
        collapsed = {label: value / total for label, value in collapsed.items()}
    return collapsed


def top_label(probabilities, label_order=None):
    label_order = label_order or LABEL_ORDER
    return max(label_order, key=lambda label: probabilities.get(label, 0.0))
