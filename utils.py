import random
from typing import Dict

import numpy as np
import pandas as pd
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_column_text(data_str: str, max_chars: int = 4096) -> str:
    if pd.isna(data_str):
        return ""
    text = str(data_str).strip()
    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars]
    return text


def serialize_columns_inline(columns_text: Dict[int, str]) -> str:
    if not columns_text:
        return ""

    lines = []
    for col_idx in sorted(columns_text.keys()):
        lines.append(f"- col_{col_idx}: {columns_text[col_idx]}")
    return "\n".join(lines)


def select_local_prefix_columns(columns_text: Dict[int, str], target_col_idx: int, context_width: int):
    if context_width < 0 or not columns_text or target_col_idx not in columns_text:
        return dict(columns_text)

    if context_width == 0:
        return {}

    ordered_indices = sorted(columns_text.keys())
    target_pos = ordered_indices.index(target_col_idx)
    chosen = []
    offset = 1

    while len(chosen) < context_width:
        added = False

        left_pos = target_pos - offset
        if left_pos >= 0:
            chosen.append(ordered_indices[left_pos])
            added = True
            if len(chosen) >= context_width:
                break

        right_pos = target_pos + offset
        if right_pos < len(ordered_indices):
            chosen.append(ordered_indices[right_pos])
            added = True
            if len(chosen) >= context_width:
                break

        if not added:
            break
        offset += 1

    chosen_set = set(chosen)
    return {col_idx: columns_text[col_idx] for col_idx in ordered_indices if col_idx in chosen_set}


def is_multi_label_value(value) -> bool:
    return isinstance(value, (list, tuple, np.ndarray))


def has_valid_label(value) -> bool:
    if is_multi_label_value(value):
        return np.asarray(value).sum() > 0
    return value != -1


def multilabel_exact_match_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.0
    return float(np.all(y_true == y_pred, axis=1).mean())


def multilabel_f1_reveal_style(y_true: np.ndarray, y_pred: np.ndarray):
    from sklearn.metrics import multilabel_confusion_matrix

    conf_mat = multilabel_confusion_matrix(y_true, y_pred)
    agg_conf_mat = conf_mat.sum(axis=0)

    pred_pos = agg_conf_mat[1, :].sum()
    true_pos = agg_conf_mat[:, 1].sum()
    precision = agg_conf_mat[1, 1] / pred_pos if pred_pos > 0 else 0.0
    recall = agg_conf_mat[1, 1] / true_pos if true_pos > 0 else 0.0
    micro_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    class_pred_pos = conf_mat[:, 1, :].sum(axis=1)
    class_true_pos = conf_mat[:, :, 1].sum(axis=1)
    class_precision = np.divide(
        conf_mat[:, 1, 1],
        class_pred_pos,
        out=np.zeros(conf_mat.shape[0], dtype=np.float64),
        where=class_pred_pos != 0,
    )
    class_recall = np.divide(
        conf_mat[:, 1, 1],
        class_true_pos,
        out=np.zeros(conf_mat.shape[0], dtype=np.float64),
        where=class_true_pos != 0,
    )
    class_f1 = np.divide(
        2 * class_precision * class_recall,
        class_precision + class_recall,
        out=np.zeros(conf_mat.shape[0], dtype=np.float64),
        where=(class_precision + class_recall) != 0,
    )
    macro_f1 = float(np.nan_to_num(class_f1).mean())
    return micro_f1, macro_f1


def filter_positive_multilabel_rows(logits, labels, hidden=None):
    if labels is None or labels.ndim <= 1:
        return logits, labels, hidden

    keep_mask = labels.sum(dim=1) > 0
    if not torch.any(keep_mask):
        return None, None, None

    logits = logits[keep_mask]
    labels = labels[keep_mask]
    if hidden is not None:
        hidden = hidden[keep_mask]
    return logits, labels, hidden
