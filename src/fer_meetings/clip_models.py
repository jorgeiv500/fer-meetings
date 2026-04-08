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
