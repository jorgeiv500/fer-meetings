import json
import re
from pathlib import Path


def load_config(path=""):
    if not path:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def slugify_model_name(value):
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "model"


def infer_model_family(model_id, explicit_family=""):
    if explicit_family:
        return str(explicit_family).strip().lower()

    normalized = str(model_id).strip().lower()
    if any(token in normalized for token in ("vit", "deit", "beit", "swin")):
        return "vit"
    if any(token in normalized for token in ("convnext", "resnet", "efficientnet", "mobilenet", "cnn")):
        return "cnn"
    return "unknown"


def resolve_model_specs(config, requested_model_ids=None, requested_face_detector="", default_model_id=""):
    requested_model_ids = requested_model_ids or []
    default_model_id = default_model_id or "HardlyHumans/Facial-expression-detection"

    if requested_model_ids:
        return [
            {
                "name": slugify_model_name(model_id),
                "hf_model_id": model_id,
                "family": infer_model_family(model_id),
                "face_detector": requested_face_detector or "haar",
            }
            for model_id in requested_model_ids
        ]

    raw_specs = config.get("models")
    if raw_specs:
        specs = []
        for raw_spec in raw_specs:
            model_id = raw_spec["hf_model_id"]
            specs.append(
                {
                    "name": raw_spec.get("name") or slugify_model_name(model_id),
                    "hf_model_id": model_id,
                    "family": infer_model_family(model_id, raw_spec.get("family", "")),
                    "face_detector": requested_face_detector or raw_spec.get("face_detector", "haar"),
                }
            )
        return specs

    raw_model = config.get("model", {})
    model_id = raw_model.get("hf_model_id", default_model_id)
    return [
        {
            "name": raw_model.get("name") or slugify_model_name(model_id),
            "hf_model_id": model_id,
            "family": infer_model_family(model_id, raw_model.get("family", "")),
            "face_detector": requested_face_detector or raw_model.get("face_detector", "haar"),
        }
    ]
