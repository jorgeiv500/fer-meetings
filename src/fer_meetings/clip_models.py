import json
from copy import deepcopy

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from fer_meetings.constants import LABEL_ORDER


def parse_json_vector(raw_value):
    values = json.loads(raw_value or "[]")
    return np.array(values, dtype=np.float32)


def parse_json_matrix(raw_value):
    values = json.loads(raw_value or "[]")
    if not values:
        return np.zeros((0, 0), dtype=np.float32)
    return np.array(values, dtype=np.float32)


def labels_to_indices(rows):
    return np.array([LABEL_ORDER.index(row["gold_label"]) for row in rows], dtype=np.int64)


def mean_embedding_matrix(rows):
    return np.stack([parse_json_vector(row["mean_embedding_json"]) for row in rows], axis=0)


def sequence_matrices(rows):
    return [parse_json_matrix(row["frame_embeddings_json"]) for row in rows]


def metric_bundle(y_true, y_pred):
    gold = [LABEL_ORDER[int(index)] for index in y_true]
    predicted = [LABEL_ORDER[int(index)] for index in y_pred]
    return {
        "n_clips": len(y_true),
        "accuracy": accuracy_score(gold, predicted),
        "balanced_accuracy": balanced_accuracy_score(gold, predicted),
        "macro_f1": f1_score(gold, predicted, labels=LABEL_ORDER, average="macro", zero_division=0),
    }


def fit_logistic_probe(train_rows):
    model = LogisticRegression(max_iter=2000, multi_class="multinomial")
    model.fit(mean_embedding_matrix(train_rows), labels_to_indices(train_rows))
    return model


def fit_hist_gradient_probe(train_rows):
    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=4,
        max_iter=200,
        random_state=42,
    )
    model.fit(mean_embedding_matrix(train_rows), labels_to_indices(train_rows))
    return model


def covariance_with_regularization(features, regularization=1e-3):
    covariance = np.cov(features, rowvar=False)
    if covariance.ndim == 0:
        covariance = np.array([[float(covariance)]], dtype=np.float64)
    covariance = np.array(covariance, dtype=np.float64)
    covariance += np.eye(covariance.shape[0], dtype=np.float64) * float(regularization)
    return covariance


def matrix_power_symmetric(matrix, power, epsilon=1e-12):
    values, vectors = np.linalg.eigh(matrix)
    values = np.clip(values, epsilon, None)
    powered = np.diag(np.power(values, power))
    return vectors @ powered @ vectors.T


def coral_align_source_to_target(source_features, target_features, regularization=1e-3):
    source = np.array(source_features, dtype=np.float64)
    target = np.array(target_features, dtype=np.float64)
    if source.ndim != 2 or target.ndim != 2:
        raise ValueError("CORAL alignment expects 2D feature matrices.")

    source_mean = source.mean(axis=0, keepdims=True)
    target_mean = target.mean(axis=0, keepdims=True)
    source_centered = source - source_mean
    target_centered = target - target_mean

    source_cov = covariance_with_regularization(source_centered, regularization=regularization)
    target_cov = covariance_with_regularization(target_centered, regularization=regularization)
    alignment = matrix_power_symmetric(source_cov, -0.5) @ matrix_power_symmetric(target_cov, 0.5)
    return ((source_centered @ alignment) + target_mean).astype(np.float32)


def fit_coral_logistic_probe(train_rows, target_rows, regularization=1e-3):
    aligned_features = coral_align_source_to_target(
        mean_embedding_matrix(train_rows),
        mean_embedding_matrix(target_rows),
        regularization=regularization,
    )
    model = LogisticRegression(max_iter=2000, multi_class="multinomial")
    model.fit(aligned_features, labels_to_indices(train_rows))
    return model


def pairwise_squared_distances(left, right):
    left_norm = (left * left).sum(dim=1, keepdim=True)
    right_norm = (right * right).sum(dim=1, keepdim=True).transpose(0, 1)
    distances = left_norm + right_norm - 2.0 * left @ right.transpose(0, 1)
    return torch.clamp(distances, min=0.0)


def rbf_mmd(source_embeddings, target_embeddings, epsilon=1e-6):
    with torch.no_grad():
        combined = torch.cat([source_embeddings.detach(), target_embeddings.detach()], dim=0)
        distances = pairwise_squared_distances(combined, combined)
        positive = distances[distances > 0]
        bandwidth = positive.median() if positive.numel() else torch.tensor(1.0, device=combined.device)
        gamma = 1.0 / torch.clamp(2.0 * bandwidth, min=epsilon)

    k_xx = torch.exp(-gamma * pairwise_squared_distances(source_embeddings, source_embeddings))
    k_yy = torch.exp(-gamma * pairwise_squared_distances(target_embeddings, target_embeddings))
    k_xy = torch.exp(-gamma * pairwise_squared_distances(source_embeddings, target_embeddings))
    return k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()


class MMDAdapterClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_classes=3, dropout=0.2):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )
        self.classifier = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, inputs):
        embeddings = self.encoder(inputs)
        logits = self.classifier(embeddings)
        return logits, embeddings


def fit_mmd_adapter_probe(train_rows, target_rows, seed=42, device="cpu", mmd_weight=0.2):
    source_rows, validation_rows = make_validation_split(train_rows, seed=seed)
    source_features = torch.tensor(mean_embedding_matrix(source_rows), dtype=torch.float32, device=device)
    source_targets = torch.tensor(labels_to_indices(source_rows), dtype=torch.long, device=device)
    target_features = torch.tensor(mean_embedding_matrix(target_rows), dtype=torch.float32, device=device)

    input_dim = int(source_features.shape[1])
    hidden_dim = min(128, max(48, input_dim // 8))
    model = MMDAdapterClassifier(input_dim=input_dim, hidden_dim=hidden_dim).to(device)

    class_counts = np.bincount(source_targets.detach().cpu().numpy(), minlength=len(LABEL_ORDER))
    total = float(class_counts.sum())
    class_weights = [total / (len(LABEL_ORDER) * max(float(count), 1.0)) for count in class_counts]
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    if validation_rows:
        validation_features = torch.tensor(mean_embedding_matrix(validation_rows), dtype=torch.float32, device=device)
        validation_targets = labels_to_indices(validation_rows)
    else:
        validation_features = None
        validation_targets = None

    best_state = deepcopy(model.state_dict())
    best_score = -1.0
    remaining_patience = 25

    for _ in range(200):
        model.train()
        optimizer.zero_grad()
        logits, source_embeddings = model(source_features)
        _, target_embeddings = model(target_features)
        classification_loss = criterion(logits, source_targets)
        loss = classification_loss + (float(mmd_weight) * rbf_mmd(source_embeddings, target_embeddings))
        loss.backward()
        optimizer.step()

        if validation_rows:
            model.eval()
            with torch.no_grad():
                validation_logits, _ = model(validation_features)
            predicted = validation_logits.argmax(dim=-1).cpu().numpy()
            score = f1_score(validation_targets, predicted, average="macro", zero_division=0)
        else:
            score = float(-loss.detach().cpu().item())

        if score > best_score:
            best_score = score
            best_state = deepcopy(model.state_dict())
            remaining_patience = 25
        else:
            remaining_patience -= 1
            if remaining_patience <= 0:
                break

    model.load_state_dict(best_state)
    return model


def predict_probe(model, rows, return_probabilities=False):
    features = mean_embedding_matrix(rows)
    predictions = model.predict(features)
    if not return_probabilities:
        return predictions
    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)
    return predictions, probabilities


def pad_sequences(sequences):
    max_length = max(sequence.shape[0] for sequence in sequences)
    embedding_dim = sequences[0].shape[1]
    batch = np.zeros((len(sequences), max_length, embedding_dim), dtype=np.float32)
    mask = np.zeros((len(sequences), max_length), dtype=bool)
    for index, sequence in enumerate(sequences):
        length = sequence.shape[0]
        batch[index, :length] = sequence
        mask[index, :length] = True
    return torch.tensor(batch), torch.tensor(mask)


class TemporalAttentionPoolingClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_classes=3, dropout=0.2):
        super().__init__()
        self.projection = torch.nn.Linear(input_dim, hidden_dim)
        self.attention = torch.nn.Linear(hidden_dim, 1)
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, inputs, mask):
        hidden = torch.tanh(self.projection(inputs))
        scores = self.attention(hidden).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(hidden * weights.unsqueeze(-1), dim=1)
        logits = self.classifier(self.dropout(pooled))
        return logits, weights


def make_validation_split(rows, seed=42):
    if len(rows) < 12:
        return rows, []
    labels = labels_to_indices(rows)
    class_counts = np.bincount(labels, minlength=len(LABEL_ORDER))
    if np.count_nonzero(class_counts) < 2 or np.any(class_counts[class_counts > 0] < 2):
        return rows, []

    train_indices, val_indices = train_test_split(
        np.arange(len(rows)),
        test_size=0.25,
        random_state=seed,
        stratify=labels,
    )
    train_rows = [rows[int(index)] for index in train_indices]
    val_rows = [rows[int(index)] for index in val_indices]
    return train_rows, val_rows


def fit_attention_pooler(train_rows, seed=42, device="cpu"):
    sequences = sequence_matrices(train_rows)
    if not sequences or sequences[0].size == 0:
        raise ValueError("Temporal attention pooling requires exported frame embeddings.")

    inner_train_rows, validation_rows = make_validation_split(train_rows, seed=seed)
    train_sequences = sequence_matrices(inner_train_rows)
    train_targets = torch.tensor(labels_to_indices(inner_train_rows), dtype=torch.long, device=device)
    train_inputs, train_mask = pad_sequences(train_sequences)
    train_inputs = train_inputs.to(device)
    train_mask = train_mask.to(device)

    input_dim = train_inputs.shape[-1]
    hidden_dim = min(128, max(32, input_dim // 2))
    model = TemporalAttentionPoolingClassifier(input_dim=input_dim, hidden_dim=hidden_dim).to(device)

    class_counts = np.bincount(train_targets.cpu().numpy(), minlength=len(LABEL_ORDER))
    class_weights = []
    total = float(class_counts.sum())
    for count in class_counts:
        class_weights.append(total / (len(LABEL_ORDER) * max(float(count), 1.0)))
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_state = deepcopy(model.state_dict())
    best_score = -1.0
    patience = 25
    remaining_patience = patience
    history = []

    if validation_rows:
        val_inputs, val_mask = pad_sequences(sequence_matrices(validation_rows))
        val_inputs = val_inputs.to(device)
        val_mask = val_mask.to(device)
        val_targets = labels_to_indices(validation_rows)
    else:
        val_inputs = val_mask = None
        val_targets = None

    for epoch_index in range(200):
        model.train()
        optimizer.zero_grad()
        logits, _ = model(train_inputs, train_mask)
        loss = criterion(logits, train_targets)
        loss.backward()
        optimizer.step()

        train_predictions = logits.argmax(dim=-1).detach().cpu().numpy()
        train_accuracy = accuracy_score(train_targets.detach().cpu().numpy(), train_predictions)

        if validation_rows:
            model.eval()
            with torch.no_grad():
                val_logits, _ = model(val_inputs, val_mask)
            predicted = val_logits.argmax(dim=-1).cpu().numpy()
            score = f1_score(val_targets, predicted, average="macro", zero_division=0)
            val_accuracy = accuracy_score(val_targets, predicted)
        else:
            score = float(-loss.detach().cpu().item())
            val_accuracy = ""

        history.append(
            {
                "epoch": epoch_index + 1,
                "train_loss": float(loss.detach().cpu().item()),
                "train_accuracy": float(train_accuracy),
                "val_macro_f1": float(score) if validation_rows else "",
                "val_accuracy": float(val_accuracy) if validation_rows else "",
            }
        )

        if score > best_score:
            best_score = score
            best_state = deepcopy(model.state_dict())
            remaining_patience = patience
        else:
            remaining_patience -= 1
            if remaining_patience <= 0:
                break

    model.load_state_dict(best_state)
    return model, history


def predict_attention_pooler(model, rows, device="cpu"):
    inputs, mask = pad_sequences(sequence_matrices(rows))
    model.eval()
    with torch.no_grad():
        logits, weights = model(inputs.to(device), mask.to(device))
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
        predicted = probabilities.argmax(axis=-1)
    return predicted, probabilities, weights.cpu().numpy()


def predict_mmd_adapter(model, rows, device="cpu"):
    features = torch.tensor(mean_embedding_matrix(rows), dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        logits, _ = model(features)
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
        predicted = probabilities.argmax(axis=-1)
    return predicted, probabilities
