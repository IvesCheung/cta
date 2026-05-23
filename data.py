import ast
import os
import random

import pandas as pd
from torch.utils.data import Dataset

from config import TASK_CONFIG
from utils import has_valid_label, normalize_column_text


def load_reveal_csv(csv_path: str, task_config: dict) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    cfg = task_config

    rename_map = {}
    if cfg["col_name_col"] != "col_idx":
        rename_map[cfg["col_name_col"]] = "col_idx"
    if cfg["label_col"] != "class_id":
        rename_map[cfg["label_col"]] = "class_id"
    if cfg["data_col"] != "data":
        rename_map[cfg["data_col"]] = "data"
    if rename_map:
        df = df.rename(columns=rename_map)

    if not cfg.get("multi_label"):
        df["class_id"] = df["class_id"].astype(int)
    else:
        df["class_id"] = df["class_id"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    df["col_idx"] = df["col_idx"].astype(int)
    df["data"] = df["data"].astype(str)

    if not cfg.get("multi_label"):
        df = df[~((df["data"] == "nan") & (df["class_id"] == -1))
                ].reset_index(drop=True)
    else:
        df = df[df["class_id"].apply(has_valid_label)].reset_index(drop=True)

    return df


def load_data_reveal(data_path: str, task: str):
    cfg = TASK_CONFIG[task]

    dir_name_map = {
        "sotab": "SOTAB-CTA",
        "turl": "WikiTables-CTA",
    }
    dir_name = dir_name_map.get(task, task)
    data_dir = os.path.join(data_path, dir_name)

    if not os.path.isdir(data_dir):
        if os.path.isdir(data_path):
            data_dir = data_path
        else:
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

    num_classes = cfg["num_classes"]

    if cfg["split"] == "5fold":
        splits = {}
        for fold_idx in range(5):
            fp = os.path.join(
                data_dir, cfg["basename_pattern"].format(fold_idx))
            if not os.path.exists(fp):
                raise FileNotFoundError(f"Fold file not found: {fp}")
            splits[fold_idx] = load_reveal_csv(fp, cfg)
        return splits, num_classes, cfg, "5fold"

    if cfg["split"] == "train_test":
        splits = {}
        for split_name in ["train", "valid", "test"]:
            fp = os.path.join(data_dir, f"{split_name}.csv")
            if not os.path.exists(fp):
                raise FileNotFoundError(f"Split file not found: {fp}")
            splits[split_name] = load_reveal_csv(fp, cfg)
        return splits, num_classes, cfg, "train_test"

    raise ValueError(f"Unknown split type: {cfg['split']}")


class CTADatasetInline:
    def __init__(self, dfs, task_config: dict, sample_rows: int = 5, is_train: bool = False, max_column_chars: int = 4096):
        df = pd.concat(dfs, ignore_index=True)
        if not task_config.get("multi_label"):
            df = df[~((df["data"] == "nan") & (df["class_id"] == -1))
                    ].reset_index(drop=True)

        self.task_config = task_config
        self.sample_rows = sample_rows
        self.is_train = is_train
        self.max_column_chars = max_column_chars
        self.groups = list(df.groupby("table_id"))
        if is_train:
            random.shuffle(self.groups)

        self._table_data = {}
        self._table_labels = {}
        for table_id, group_df in self.groups:
            group_df = group_df.sort_values("col_idx")
            col_data = {}
            col_labels_map = {}
            for _, row in group_df.iterrows():
                col_idx = int(row["col_idx"])
                col_data[col_idx] = normalize_column_text(
                    row["data"], max_chars=self.max_column_chars)
                if task_config.get("multi_label"):
                    col_labels_map[col_idx] = row["class_id"]
                else:
                    col_labels_map[col_idx] = int(row["class_id"])
            self._table_data[table_id] = col_data
            self._table_labels[table_id] = col_labels_map

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        table_id, _ = self.groups[idx]
        columns_text = self._table_data[table_id]
        col_labels_map = self._table_labels[table_id]
        col_labels = [(ci, cid)
                      for ci, cid in col_labels_map.items() if has_valid_label(cid)]
        if not col_labels:
            return None

        table_info = {
            "name": str(table_id),
            "columns_text": columns_text,
        }
        return table_id, col_labels, table_info


class CTAColumnDatasetInline(Dataset):
    def __init__(self, dfs, task_config: dict, max_column_chars: int = 4096):
        df = pd.concat(dfs, ignore_index=True)
        if not task_config.get("multi_label"):
            df = df[df["class_id"] != -1].reset_index(drop=True)

        self.samples = []
        for _, row in df.iterrows():
            label = row["class_id"]
            if not has_valid_label(label):
                continue

            col_idx = int(row["col_idx"])
            self.samples.append(
                {
                    "table_id": str(row["table_id"]),
                    "col_name": f"col_{col_idx}",
                    "col_text": normalize_column_text(row["data"], max_chars=max_column_chars),
                    "label": label if task_config.get("multi_label") else int(label),
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
