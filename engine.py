import math
import random
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import CTAColumnDatasetInline, CTADatasetInline
from losses import EarlyStopping, SupConLoss, build_eval_criterion, build_train_criterion
from modeling import create_adamw, get_decoder_backbone, setup_model
from tokenization import tokenize_column_prompts_batch, tokenize_prefix, tokenize_suffixes_batch
from utils import (
    filter_positive_multilabel_rows,
    has_valid_label,
    is_multi_label_value,
    multilabel_exact_match_accuracy,
    multilabel_f1_reveal_style,
    select_local_prefix_columns,
)


def expand_kv_cache(kv_cache, num_cols: int):
    from transformers import DynamicCache

    batch_cache = DynamicCache()
    for layer_idx in range(len(kv_cache)):
        k_tensor, v_tensor = kv_cache[layer_idx]
        batch_cache.update(
            k_tensor.expand(num_cols, -1, -1, -1).contiguous(),
            v_tensor.expand(num_cols, -1, -1, -1).contiguous(),
            layer_idx,
        )
    return batch_cache


def _compose_hidden(all_hidden, attn_mask, repr_layers, repr_pool: str, repr_l2_norm: bool, device):
    per_layer = []
    for layer_idx in repr_layers:
        hidden = all_hidden[layer_idx]
        if repr_pool == "last":
            last_pos = attn_mask.sum(dim=1) - 1
            pooled = hidden[torch.arange(
                hidden.shape[0], device=device), last_pos]
        else:
            weights = attn_mask.unsqueeze(-1).float()
            pooled = (hidden * weights).sum(dim=1) / \
                weights.sum(dim=1).clamp_min(1.0)
        per_layer.append(pooled)

    hidden = per_layer[0] if len(per_layer) == 1 else torch.stack(
        per_layer, dim=0).mean(dim=0)
    if repr_l2_norm:
        hidden = F.normalize(hidden, p=2, dim=-1)
    return hidden


def process_table(model, tokenizer, classifier, device, table_info, col_labels, sample_rows, max_prefix_length, max_suffix_length, prefix_mode: str = "none", repr_layers=None, repr_pool: str = "mean", repr_l2_norm: bool = False, prefix_context_width: int = 0):
    columns_text = table_info.get("columns_text", {})

    valid_cols = [
        (col_idx, class_id)
        for col_idx, class_id in col_labels
        if col_idx in columns_text and (not is_multi_label_value(class_id) or has_valid_label(class_id))
    ]
    if not valid_cols:
        return None, None, None

    col_abs_indices, labels = zip(*valid_cols)
    col_names = [f"col_{col_idx}" for col_idx in col_abs_indices]
    col_values = [columns_text.get(col_idx, "") for col_idx in col_abs_indices]

    if is_multi_label_value(labels[0]):
        labels = torch.tensor(np.asarray(
            labels), dtype=torch.float32, device=device)
    else:
        labels = torch.tensor(labels, dtype=torch.long, device=device)

    backbone = get_decoder_backbone(model)
    repr_layers = [-1] if repr_layers is None else repr_layers

    if prefix_mode == "none":
        prompt_ids, prompt_mask = tokenize_column_prompts_batch(
            tokenizer, col_names, col_values, max_prefix_length)
        prompt_ids = prompt_ids.to(device)
        prompt_mask = prompt_mask.to(device)

        out = backbone(input_ids=prompt_ids, attention_mask=prompt_mask,
                       output_hidden_states=True, return_dict=True)
        hidden = _compose_hidden(
            out.hidden_states, prompt_mask, repr_layers, repr_pool, repr_l2_norm, device)
        logits = classifier(hidden.float())
        del out
        return filter_positive_multilabel_rows(logits, labels, hidden)

    if prefix_context_width >= 0:
        all_logits = []
        all_hidden = []

        for col_idx, col_name, col_value in zip(col_abs_indices, col_names, col_values):
            local_columns_text = select_local_prefix_columns(
                columns_text, col_idx, prefix_context_width)
            local_table_info = {
                "name": table_info["name"], "columns_text": local_columns_text}

            prefix_ids = tokenize_prefix(
                tokenizer, local_table_info, sample_rows, max_prefix_length).to(device)
            with torch.no_grad():
                prefix_out = backbone(
                    input_ids=prefix_ids, use_cache=True, return_dict=True)
                kv_cache = prefix_out.past_key_values
                del prefix_out

            suffix_ids, suffix_mask = tokenize_suffixes_batch(
                tokenizer, [col_name], [col_value], max_length=max_suffix_length)
            suffix_ids = suffix_ids.to(device)
            suffix_mask = suffix_mask.to(device)
            batch_kv_cache = expand_kv_cache(kv_cache, 1)
            del kv_cache

            out = backbone(
                input_ids=suffix_ids,
                attention_mask=suffix_mask,
                past_key_values=batch_kv_cache,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden = _compose_hidden(
                out.hidden_states, suffix_mask, repr_layers, repr_pool, repr_l2_norm, device)
            logit = classifier(hidden.float())

            all_hidden.append(hidden.squeeze(0))
            all_logits.append(logit)
            del out, batch_kv_cache

        logits = torch.cat(all_logits, dim=0)
        hidden = torch.stack(all_hidden, dim=0)
        return filter_positive_multilabel_rows(logits, labels, hidden)

    prefix_ids = tokenize_prefix(
        tokenizer, table_info, sample_rows, max_prefix_length).to(device)
    with torch.no_grad():
        prefix_out = backbone(input_ids=prefix_ids,
                              use_cache=True, return_dict=True)
        kv_cache = prefix_out.past_key_values
        del prefix_out

    suffix_ids, suffix_mask = tokenize_suffixes_batch(
        tokenizer, col_names, col_values, max_length=max_suffix_length)
    suffix_ids = suffix_ids.to(device)
    suffix_mask = suffix_mask.to(device)
    batch_kv_cache = expand_kv_cache(kv_cache, len(col_names))
    del kv_cache

    out = backbone(
        input_ids=suffix_ids,
        attention_mask=suffix_mask,
        past_key_values=batch_kv_cache,
        output_hidden_states=True,
        return_dict=True,
    )
    hidden = _compose_hidden(
        out.hidden_states, suffix_mask, repr_layers, repr_pool, repr_l2_norm, device)
    logits = classifier(hidden.float())

    del out, batch_kv_cache
    return filter_positive_multilabel_rows(logits, labels, hidden)


def process_none_batch(model, tokenizer, classifier, device, batch_samples, max_length: int, repr_layers=None, repr_pool: str = "mean", repr_l2_norm: bool = False):
    if not batch_samples:
        return None, None, None

    batch_samples = [
        sample for sample in batch_samples if not is_multi_label_value(sample["label"]) or has_valid_label(sample["label"])
    ]
    if not batch_samples:
        return None, None, None

    col_names = [sample["col_name"] for sample in batch_samples]
    col_values = [sample["col_text"] for sample in batch_samples]
    raw_labels = [sample["label"] for sample in batch_samples]

    if is_multi_label_value(raw_labels[0]):
        labels = torch.tensor(np.asarray(raw_labels),
                              dtype=torch.float32, device=device)
    else:
        labels = torch.tensor(raw_labels, dtype=torch.long, device=device)

    prompt_ids, prompt_mask = tokenize_column_prompts_batch(
        tokenizer, col_names, col_values, max_length)
    prompt_ids = prompt_ids.to(device)
    prompt_mask = prompt_mask.to(device)

    backbone = get_decoder_backbone(model)
    repr_layers = [-1] if repr_layers is None else repr_layers
    out = backbone(input_ids=prompt_ids, attention_mask=prompt_mask,
                   output_hidden_states=True, return_dict=True)
    hidden = _compose_hidden(
        out.hidden_states, prompt_mask, repr_layers, repr_pool, repr_l2_norm, device)
    logits = classifier(hidden.float())
    del out
    return filter_positive_multilabel_rows(logits, labels, hidden)


def train_one_epoch(model, tokenizer, classifier, optimizer, scheduler, criterion, con_criterion, contrastive_weight, train_dataset, num_epochs, epoch, grad_accum_steps, device, sample_rows, max_prefix_length, max_suffix_length, max_grad_norm, prefix_mode, repr_layers, repr_pool, repr_l2_norm, none_batch_size, max_train_steps_per_epoch, prefix_context_width):
    model.train()
    classifier.train()
    optimizer.zero_grad(set_to_none=True)

    train_loss = 0.0
    train_cls_loss = 0.0
    train_con_loss = 0.0
    train_steps = 0

    if prefix_mode == "none":
        train_loader = DataLoader(
            train_dataset, batch_size=none_batch_size, shuffle=True, collate_fn=lambda batch: batch)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        iterator = pbar
    else:
        random.shuffle(train_dataset.groups)
        pbar = tqdm(range(len(train_dataset)),
                    desc=f"Epoch {epoch + 1}/{num_epochs}")
        iterator = pbar

    for batch in iterator:
        if prefix_mode == "none":
            logits, labels, hidden = process_none_batch(
                model,
                tokenizer,
                classifier,
                device,
                batch,
                max_prefix_length,
                repr_layers=repr_layers,
                repr_pool=repr_pool,
                repr_l2_norm=repr_l2_norm,
            )
        else:
            item = train_dataset[batch]
            if item is None:
                continue
            _, col_labels, table_info = item
            logits, labels, hidden = process_table(
                model,
                tokenizer,
                classifier,
                device,
                table_info,
                col_labels,
                sample_rows,
                max_prefix_length,
                max_suffix_length,
                prefix_mode=prefix_mode,
                repr_layers=repr_layers,
                repr_pool=repr_pool,
                repr_l2_norm=repr_l2_norm,
                prefix_context_width=prefix_context_width,
            )

        if logits is None:
            continue

        cls_loss = criterion(logits, labels)
        con_loss = con_criterion(
            hidden.float(), labels) if labels.ndim == 1 else hidden.new_zeros(())
        total_loss = (cls_loss + contrastive_weight *
                      con_loss) / grad_accum_steps

        total_loss.backward()
        train_loss += total_loss.item() * grad_accum_steps
        train_cls_loss += cls_loss.item()
        train_con_loss += con_loss.item()
        train_steps += 1

        if train_steps % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(classifier.parameters()), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        pbar.set_postfix(
            loss=f"{train_loss / max(train_steps, 1):.4f}",
            cls=f"{train_cls_loss / max(train_steps, 1):.4f}",
            con=f"{train_con_loss / max(train_steps, 1):.4f}",
        )

        if max_train_steps_per_epoch > 0 and train_steps >= max_train_steps_per_epoch:
            break

    if train_steps > 0 and train_steps % grad_accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(classifier.parameters()), max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    return train_loss / max(train_steps, 1)


@torch.no_grad()
def evaluate(model, tokenizer, classifier, dataset, criterion, device, sample_rows, max_prefix_length, max_suffix_length, prefix_mode, repr_layers, repr_pool, repr_l2_norm, none_batch_size, prefix_context_width):
    model.eval()
    classifier.eval()

    all_preds = []
    all_labels = []
    total_loss = 0.0
    total_steps = 0

    if prefix_mode == "none":
        data_loader = DataLoader(
            dataset, batch_size=none_batch_size, shuffle=False, collate_fn=lambda batch: batch)
        iterator = tqdm(data_loader, desc="Evaluating", leave=False)
    else:
        iterator = tqdm(range(len(dataset)), desc="Evaluating", leave=False)

    for batch in iterator:
        if prefix_mode == "none":
            logits, labels, _ = process_none_batch(
                model,
                tokenizer,
                classifier,
                device,
                batch,
                max_prefix_length,
                repr_layers=repr_layers,
                repr_pool=repr_pool,
                repr_l2_norm=repr_l2_norm,
            )
        else:
            item = dataset[batch]
            if item is None:
                continue
            _, col_labels, table_info = item
            logits, labels, _ = process_table(
                model,
                tokenizer,
                classifier,
                device,
                table_info,
                col_labels,
                sample_rows,
                max_prefix_length,
                max_suffix_length,
                prefix_mode=prefix_mode,
                repr_layers=repr_layers,
                repr_pool=repr_pool,
                repr_l2_norm=repr_l2_norm,
                prefix_context_width=prefix_context_width,
            )

        if logits is None:
            continue

        loss = criterion(logits, labels)
        total_loss += loss.item()
        total_steps += 1

        preds = logits >= 0 if labels.ndim > 1 else logits.argmax(dim=-1)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    if not all_preds:
        return {"accuracy": 0, "macro_f1": 0, "micro_f1": 0, "loss": 0}

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    if all_labels.ndim > 1:
        all_preds_np = all_preds.int().numpy()
        all_labels_np = all_labels.int().numpy()
        acc = multilabel_exact_match_accuracy(all_labels_np, all_preds_np)
        micro_f1, macro_f1 = multilabel_f1_reveal_style(
            all_labels_np, all_preds_np)
    else:
        from sklearn.metrics import accuracy_score, f1_score

        all_preds_np = all_preds.numpy()
        all_labels_np = all_labels.numpy()
        acc = accuracy_score(all_labels_np, all_preds_np)
        macro_f1 = f1_score(all_labels_np, all_preds_np,
                            average="macro", zero_division=0)
        micro_f1 = f1_score(all_labels_np, all_preds_np,
                            average="micro", zero_division=0)

    return {
        "accuracy": acc,
        "exact_match_accuracy": acc,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "loss": total_loss / max(total_steps, 1),
    }


def evaluate_best_on_test(model, tokenizer, classifier, test_dataset, criterion_eval, args, repr_layers, metric_name: str, score: float, fold_idx=None):
    metrics = evaluate(
        model,
        tokenizer,
        classifier,
        test_dataset,
        criterion_eval,
        model.device,
        args.sample_rows,
        args.max_prefix_length,
        args.max_suffix_length,
        args.prefix_mode,
        repr_layers,
        args.repr_pool,
        args.repr_l2_norm,
        args.none_batch_size,
        args.prefix_context_width,
    )

    prefix = f"  -> [{metric_name}] Fold {fold_idx} test" if fold_idx is not None else f"  -> [{metric_name}] Test"
    print(
        f"{prefix} @ best={score:.4f} | "
        f"Acc: {metrics['accuracy']:.4f} | "
        f"Macro F1: {metrics['macro_f1']:.4f} | "
        f"Micro F1: {metrics['micro_f1']:.4f}"
    )
    return metrics


def build_dataset(task_config, dfs, prefix_mode: str, sample_rows: int, is_train: bool, max_column_chars: int):
    dataset_cls = CTAColumnDatasetInline if prefix_mode == "none" else CTADatasetInline
    if dataset_cls is CTADatasetInline:
        return dataset_cls(dfs, task_config, sample_rows, is_train, max_column_chars=max_column_chars)
    return dataset_cls(dfs, task_config, max_column_chars=max_column_chars)


def create_training_state(args, train_df, task_config, num_classes: int):
    device = f"cuda:{args.gpu_id}"
    model, tokenizer, classifier, _ = setup_model(
        args.model_path,
        num_classes,
        args.lora_r,
        args.lora_alpha,
        args.lora_dropout,
        args.num_unfrozen_layers,
        args.head_type,
        device,
    )

    criterion = build_train_criterion(
        train_df, task_config, num_classes, args, model.device)
    criterion_eval = build_eval_criterion(task_config, args)
    con_criterion = SupConLoss(temperature=args.contrastive_temperature)
    return model, tokenizer, classifier, criterion, criterion_eval, con_criterion


def create_optimizer_scheduler(args, model, classifier, train_dataset):
    trainable_params = list(model.parameters()) + list(classifier.parameters())
    optimizer = create_adamw(
        trainable_params, args.learning_rate, args.weight_decay)
    raw_steps_per_epoch = math.ceil(
        len(train_dataset) / (args.none_batch_size if args.prefix_mode == "none" else 1))
    steps_per_epoch = min(
        raw_steps_per_epoch, args.max_train_steps_per_epoch) if args.max_train_steps_per_epoch > 0 else raw_steps_per_epoch
    total_steps = (steps_per_epoch // args.grad_accum_steps +
                   1) * args.num_epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        total_steps=total_steps,
        pct_start=args.warmup_ratio,
        anneal_strategy="cos",
    )
    return optimizer, scheduler


def fit_single_split(args, train_df, valid_df, test_df, task_config, num_classes: int, repr_layers, fold_idx=None):
    train_dataset = build_dataset(task_config, [
                                  train_df], args.prefix_mode, args.sample_rows, True, args.max_column_chars)
    val_dataset = build_dataset(task_config, [
                                valid_df], args.prefix_mode, args.sample_rows, False, args.max_column_chars)
    test_dataset = build_dataset(task_config, [
                                 test_df], args.prefix_mode, args.sample_rows, False, args.max_column_chars)

    unit = "columns" if args.prefix_mode == "none" else "tables"
    if fold_idx is None:
        print(
            f"  Train: {len(train_dataset)} {unit}, Valid: {len(val_dataset)} {unit}, Test: {len(test_dataset)} {unit}")
    else:
        print(
            f"  Train: {len(train_dataset)} {unit}, Valid: {len(val_dataset)} {unit}")

    model, tokenizer, classifier, criterion, criterion_eval, con_criterion = create_training_state(
        args, train_df, task_config, num_classes)
    optimizer, scheduler = create_optimizer_scheduler(
        args, model, classifier, train_dataset)

    best_macro_f1 = -1.0
    best_micro_f1 = -1.0
    best_state_macro = None
    best_state_micro = None
    best_test_macro = None
    best_test_micro = None
    early_stopper = EarlyStopping(patience=args.patience)

    for epoch in range(args.num_epochs):
        avg_loss = train_one_epoch(
            model,
            tokenizer,
            classifier,
            optimizer,
            scheduler,
            criterion,
            con_criterion,
            args.contrastive_weight,
            train_dataset,
            args.num_epochs,
            epoch,
            args.grad_accum_steps,
            model.device,
            args.sample_rows,
            args.max_prefix_length,
            args.max_suffix_length,
            args.max_grad_norm,
            args.prefix_mode,
            repr_layers,
            args.repr_pool,
            args.repr_l2_norm,
            args.none_batch_size,
            args.max_train_steps_per_epoch,
            args.prefix_context_width,
        )

        val_metrics = evaluate(
            model,
            tokenizer,
            classifier,
            val_dataset,
            criterion,
            model.device,
            args.sample_rows,
            args.max_prefix_length,
            args.max_suffix_length,
            args.prefix_mode,
            repr_layers,
            args.repr_pool,
            args.repr_l2_norm,
            args.none_batch_size,
            args.prefix_context_width,
        )

        print(
            f"  Epoch [{epoch + 1}/{args.num_epochs}] "
            f"Train Loss: {avg_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Macro F1: {val_metrics['macro_f1']:.4f} | "
            f"Micro F1: {val_metrics['micro_f1']:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
        )

        state_dict = {
            "model": deepcopy(model.state_dict()),
            "classifier": deepcopy(classifier.state_dict()),
        }
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            best_state_macro = state_dict
            best_test_macro = evaluate_best_on_test(
                model,
                tokenizer,
                classifier,
                test_dataset,
                criterion_eval,
                args,
                repr_layers,
                "macro",
                best_macro_f1,
                fold_idx=fold_idx,
            )
            if args.save_model:
                filename = "best_macro.pt" if fold_idx is None else f"fold_{fold_idx}_best_macro.pt"
                torch.save(best_state_macro, f"{args.result_dir}/{filename}")
            print(f"  -> New best Macro F1: {best_macro_f1:.4f}")

        if val_metrics["micro_f1"] > best_micro_f1:
            best_micro_f1 = val_metrics["micro_f1"]
            best_state_micro = state_dict
            best_test_micro = evaluate_best_on_test(
                model,
                tokenizer,
                classifier,
                test_dataset,
                criterion_eval,
                args,
                repr_layers,
                "micro",
                best_micro_f1,
                fold_idx=fold_idx,
            )
            if args.save_model:
                filename = "best_micro.pt" if fold_idx is None else f"fold_{fold_idx}_best_micro.pt"
                torch.save(best_state_micro, f"{args.result_dir}/{filename}")
            print(f"  -> New best Micro F1: {best_micro_f1:.4f}")

        if early_stopper(val_metrics["macro_f1"]):
            print(f"  Early stopping at epoch {epoch + 1}")
            break

    result = {
        "best_val_macro_f1": best_macro_f1,
        "best_val_micro_f1": best_micro_f1,
        "test_at_best_macro": best_test_macro,
        "test_at_best_micro": best_test_micro,
    }
    return result, (model, classifier, optimizer, scheduler)
