import argparse
from typing import Dict, List


TASK_CONFIG = {
    "gt-semtab22-dbpedia-all": {
        "split": "5fold",
        "num_classes": 101,
        "col_name_col": "col_idx",
        "label_col": "class_id",
        "data_col": "data",
        "basename_pattern": "db_semi1_cv_{}.csv",
        "delimiter": ";",
    },
    "gt-semtab22-schema-property-all": {
        "split": "5fold",
        "num_classes": 53,
        "col_name_col": "col_idx",
        "label_col": "class_id",
        "data_col": "data",
        "basename_pattern": "sp_semi1_cv_{}.csv",
        "delimiter": ";",
    },
    "sotab": {
        "split": "train_test",
        "num_classes": 91,
        "col_name_col": "column_index",
        "label_col": "label",
        "data_col": "data",
        "delimiter": " ",
    },
    "turl": {
        "split": "train_test",
        "num_classes": 255,
        "col_name_col": "column_index",
        "label_col": "label",
        "data_col": "data",
        "multi_label": True,
        "delimiter": ";",
    },
}


def parse_repr_layers(num_hidden_layers: int, layers_str: str) -> List[int]:
    raw = [s.strip() for s in str(layers_str).split(",") if s.strip()]
    if not raw:
        return [-1]

    abs_layers = []
    for token in raw:
        idx = int(token)
        abs_idx = idx if idx >= 0 else (num_hidden_layers + 1 + idx)
        abs_idx = max(0, min(num_hidden_layers, abs_idx))
        abs_layers.append(abs_idx)
    return sorted(set(abs_layers))


def resolve_model_hparams(model_path: str) -> Dict[str, int]:
    model_path = model_path.replace("\\", "/").lower()

    if "qwen3-0.6b" in model_path:
        return {"none_batch_size": 16, "lora_r": 32, "lora_alpha": 64}
    if "qwen3-4b" in model_path or "llama3.2-3b" in model_path:
        return {"none_batch_size": 8, "lora_r": 16, "lora_alpha": 32}
    if "qwen3-8b" in model_path:
        return {"none_batch_size": 4, "lora_r": 8, "lora_alpha": 16}

    return {"none_batch_size": 16, "lora_r": 32, "lora_alpha": 64}


def apply_model_dependent_defaults(args):
    inferred = resolve_model_hparams(args.model_path)
    if args.none_batch_size is None:
        args.none_batch_size = inferred["none_batch_size"]
    if args.lora_r is None:
        args.lora_r = inferred["lora_r"]
    if args.lora_alpha is None:
        args.lora_alpha = inferred["lora_alpha"]
    return args


def parse_args():
    parser = argparse.ArgumentParser(description="SemanticCTA (LoRA) - REVEAL data format")

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--task", type=str, required=True, choices=list(TASK_CONFIG.keys()))
    parser.add_argument("--data_path", type=str, required=True, help="Dataset root directory")
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--sample_rows", type=int, default=5)
    parser.add_argument("--max_prefix_length", type=int, default=256)
    parser.add_argument(
        "--max_suffix_length",
        type=int,
        default=512,
        help="Maximum suffix or prompt token length",
    )
    parser.add_argument(
        "--max_column_chars",
        type=int,
        default=128,
        help="Character clipping before tokenization",
    )
    parser.add_argument(
        "--prefix_mode",
        type=str,
        default="none",
        choices=["none", "full"],
        help="none: full prompt per column; full: shared table prefix plus suffix",
    )
    parser.add_argument(
        "--prefix_context_width",
        type=int,
        default=0,
        help="Neighbor column count visible to each target column when prefix_mode=full",
    )
    parser.add_argument(
        "--repr_layers",
        type=str,
        default="-4,-8",
        help="Comma-separated representation layers, supports negative indexing",
    )
    parser.add_argument(
        "--repr_pool",
        type=str,
        default="last",
        choices=["mean", "last"],
        help="Pooling strategy for hidden states",
    )
    parser.add_argument(
        "--repr_l2_norm",
        action="store_true",
        help="Apply L2 normalization before classification",
    )
    parser.add_argument(
        "--head_type",
        type=str,
        default="ln_mlp",
        choices=["linear", "cosine", "mlp", "ln_mlp", "res_mlp"],
        help="Classification head type",
    )
    parser.add_argument(
        "--none_batch_size",
        type=int,
        default=None,
        help="Column mini-batch size when prefix_mode=none",
    )
    parser.add_argument(
        "--max_train_steps_per_epoch",
        type=int,
        default=0,
        help="Limit train steps per epoch, 0 means no limit",
    )

    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--lora_dropout", type=float, default=0.3)
    parser.add_argument("--num_unfrozen_layers", type=int, default=0)

    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--grad_accum_steps", type=int, default=16)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--label_smoothing", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--warmup_ratio", type=float, default=0.15)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--contrastive_weight", type=float, default=0.1)
    parser.add_argument("--contrastive_temperature", type=float, default=0.07)
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--gpu_id", type=int, default=0)

    return parser.parse_args()
