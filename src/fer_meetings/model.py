from pathlib import Path

from fer_meetings.labels import collapse_probabilities, load_label_map, top_label


class HfEmotionClassifier:
    def __init__(
        self,
        model_id,
        label_map_path=None,
        face_detector="haar",
        device="auto",
        model_name="",
        model_family="",
    ):
        import cv2
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        self.model_id = model_id
        self.model_name = model_name or model_id
        self.model_family = (model_family or "").strip().lower() or self._infer_family(model_id)
        self.label_map = load_label_map(label_map_path)
        self.face_detector = face_detector
        self.device = self._select_device(device, torch)
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageClassification.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()
        self.id2label = {int(index): label for index, label in self.model.config.id2label.items()}

        if face_detector == "haar":
            cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            self.cascade = cv2.CascadeClassifier(str(cascade_path))
        elif face_detector == "full-frame":
            self.cascade = None
        else:
            raise ValueError("face_detector must be 'haar' or 'full-frame'")

    @staticmethod
    def _infer_family(model_id):
        normalized = str(model_id).strip().lower()
        if any(token in normalized for token in ("vit", "deit", "beit", "swin")):
            return "vit"
        if any(token in normalized for token in ("convnext", "resnet", "efficientnet", "mobilenet", "cnn")):
            return "cnn"
        return "unknown"

    @staticmethod
    def _select_device(requested_device, torch_module):
        if requested_device != "auto":
            return requested_device
        if torch_module.cuda.is_available():
            return "cuda"
        if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _extract_face(self, frame_bgr):
        import cv2

        if self.cascade is None:
            return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), False

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(48, 48),
        )
        if len(faces) == 0:
            return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), False

        x, y, width, height = max(faces, key=lambda item: item[2] * item[3])
        margin = int(min(width, height) * 0.15)
        x0 = max(0, x - margin)
        y0 = max(0, y - margin)
        x1 = min(frame_bgr.shape[1], x + width + margin)
        y1 = min(frame_bgr.shape[0], y + height + margin)
        crop = frame_bgr[y0:y1, x0:x1]
        return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), True

    def _extract_embedding(self, outputs):
        import torch

        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states:
            tensor = hidden_states[-1][0].detach()
            if tensor.ndim == 1:
                embedding = tensor
            elif tensor.ndim == 2:
                # Transformer encoders often expose a CLS token in the first position.
                embedding = tensor[0] if self.model_family == "vit" else tensor.mean(dim=0)
            else:
                reduce_dims = tuple(range(1, tensor.ndim))
                embedding = tensor.mean(dim=reduce_dims)
        else:
            embedding = outputs.logits[0].detach()

        if embedding.dtype != torch.float32:
            embedding = embedding.float()
        return embedding.cpu().tolist()

    def predict_frame(self, frame_bgr, include_embedding=False):
        import torch
        from PIL import Image

        face_rgb, face_detected = self._extract_face(frame_bgr)
        image = Image.fromarray(face_rgb)
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=include_embedding)

        probabilities = torch.softmax(outputs.logits[0], dim=-1).detach().cpu().tolist()
        raw_probabilities = {
            self.id2label[index]: float(probability)
            for index, probability in enumerate(probabilities)
        }
        mapped_probabilities = collapse_probabilities(raw_probabilities, self.label_map)

        return {
            "raw_label": top_label(raw_probabilities, list(raw_probabilities.keys())),
            "mapped_label": top_label(mapped_probabilities),
            "raw_probabilities": raw_probabilities,
            "mapped_probabilities": mapped_probabilities,
            "face_detected": face_detected,
            "embedding": self._extract_embedding(outputs) if include_embedding else [],
        }
