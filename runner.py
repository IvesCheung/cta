import json
import os
import random

import numpy as np
import pandas as pd
import torch

from config import apply_model_dependent_defaults, parse_args, parse_repr_layers
from data import load_data_reveal
from engine import fit_single_split
from utils import set_seed


def run_5fold_cv(args, splits, num_classes, task_config):
    os.makedirs(args.result_dir, exist_ok=True)
    fold_results = []

    for fold_idx in range(5):
        print(f"\n{'=' * 60}")
        print(f"Fold {fold_idx}/4")
        print(f"{'=' * 60}")

        test_df = splits[fold_idx]
        other_dfs = [splits[i] for i in range(5) if i != fold_idx]
        other_all = pd.concat(other_dfs, ignore_index=True)

        unique_tables = other_all["table_id"].unique().tolist()
        random.shuffle(unique_tables)
        split_point = int(len(unique_tables) * 0.8)
        train_tables = set(unique_tables[:split_point])
        valid_tables = set(unique_tables[split_point:])

        train_df = other_all[other_all["table_id"].isin(train_tables)]
        valid_df = other_all[other_all["table_id"].isin(valid_tables)]

        result, state = fit_single_split(
            args,
            train_df,
            valid_df,
            test_df,
            task_config,
            num_classes,
            parse_repr_layers_for_model(args),
            fold_idx=fold_idx,
        )
        model, classifier, optimizer, scheduler = state
        fold_results.append({"fold": fold_idx, **result})
        print(
            f"  Fold {fold_idx} best val - Macro F1: {result['best_val_macro_f1']:.4f}, "
            f"Micro F1: {result['best_val_micro_f1']:.4f}"
        )

        del model, classifier, optimizer, scheduler
        torch.cuda.empty_cache()

    test_results = {
        "macro": {f"fold_{item['fold']}": item["test_at_best_macro"] for item in fold_results},
        "micro": {f"fold_{item['fold']}": item["test_at_best_micro"] for item in fold_results},
    }

    test_avg = {}
    for metric_name in ["macro", "micro"]:
        avg_acc = np.mean([metrics["accuracy"]
                          for metrics in test_results[metric_name].values()])
        avg_macro = np.mean([metrics["macro_f1"]
                            for metrics in test_results[metric_name].values()])
        avg_micro = np.mean([metrics["micro_f1"]
                            for metrics in test_results[metric_name].values()])
        test_avg[metric_name] = {
            "accuracy": avg_acc,
            "macro_f1": avg_macro,
            "micro_f1": avg_micro,
        }

    return {
        "per_fold_cv": fold_results,
        "test": test_results,
        "test_avg": test_avg,
    }


def run_train_test(args, splits, num_classes, task_config):
    os.makedirs(args.result_dir, exist_ok=True)
    result, state = fit_single_split(
        args,
        splits["train"],
        splits["valid"],
        splits["test"],
        task_config,
        num_classes,
        parse_repr_layers_for_model(args),
        fold_idx=None,
    )
    model, classifier, optimizer, scheduler = state
    del model, classifier, optimizer, scheduler
    torch.cuda.empty_cache()

    return {
        "best_val_macro_f1": result["best_val_macro_f1"],
        "best_val_micro_f1": result["best_val_micro_f1"],
        "test": {
            "macro": result["test_at_best_macro"],
            "micro": result["test_at_best_micro"],
        },
    }


def parse_repr_layers_for_model(args):
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        args.model_path, trust_remote_code=True)
    return parse_repr_layers(config.num_hidden_layers, args.repr_layers)


def print_run_header(args, num_classes, splits, split_type):
    print("=" * 60)
    print("SemanticCTA (LoRA) - REVEAL Data Format")
    print("=" * 60)
    print(f"  Model:        {args.model_path}")
    print(f"  Task:         {args.task}")
    print(f"  Data path:    {args.data_path}")
    print(f"  Result dir:   {args.result_dir}")
    print(f"  LoRA r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"  Epochs:       {args.num_epochs}")
    print(f"  LR:           {args.learning_rate}")
    print(f"  Prefix mode:  {args.prefix_mode}")
    print(f"  Max prefix:   {args.max_prefix_length}")
    print(f"  Max suffix:   {args.max_suffix_length}")
    print(f"  Max col char: {args.max_column_chars}")
    print(f"  None batch:   {args.none_batch_size}")
    print(f"  Max tr step:  {args.max_train_steps_per_epoch}")
    print(f"  Repr layers:  {args.repr_layers}")
    print(f"  Repr pool:    {args.repr_pool}")
    print(f"  Repr L2 norm: {args.repr_l2_norm}")
    print(f"  Head type:    {args.head_type}")
    print(f"  Contrastive:  weight={args.contrastive_weight}")
    print("=" * 60)
    print(f"\n  Classes: {num_classes}")

    if split_type == "5fold":
        print(f"  Split: 5-fold CV ({len(splits)} folds)")
        for fold_idx in range(5):
            print(f"    Fold {fold_idx}: {len(splits[fold_idx])} rows")
    else:
        print("  Split: train/valid/test")
        for split_name in ["train", "valid", "test"]:
            print(f"    {split_name}: {len(splits[split_name])} rows")


def main():
    args = apply_model_dependent_defaults(parse_args())
    set_seed(args.seed)

    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    splits, num_classes, task_config, split_type = load_data_reveal(
        args.data_path, args.task)
    print_run_header(args, num_classes, splits, split_type)

    os.makedirs(args.result_dir, exist_ok=True)
    with open(os.path.join(args.result_dir, "config.json"), "w", encoding="utf-8") as file_obj:
        json.dump(vars(args), file_obj, indent=2, default=str)

    if split_type == "5fold":
        final_results = run_5fold_cv(args, splits, num_classes, task_config)
    else:
        final_results = run_train_test(args, splits, num_classes, task_config)

    results_path = os.path.join(args.result_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as file_obj:
        json.dump(final_results, file_obj, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")
