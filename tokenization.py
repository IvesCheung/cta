import torch

from utils import serialize_columns_inline


SYSTEM_PROMPT = (
    "You are a semantic type annotator for tabular data. "
    "Analyze each column's semantic meaning, data type, value patterns, "
    "cross-column relationships, and domain-specific interpretation."
)

SUFFIX_TEMPLATE = (
    'Column "{col_name}" contains values: {col_values}. '
    "Based on the table above, the semantic type and meaning of this column is"
)

NO_PREFIX_TEMPLATE = (
    'Column "{col_name}" contains values: {col_values}. '
    "What is the semantic type of this column?"
)


def tokenize_messages_with_fallback(tokenizer, messages, max_length: int) -> torch.Tensor:
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
        except TypeError:
            input_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            if input_ids.shape[-1] > max_length:
                input_ids = input_ids[:, -max_length:]
            return input_ids
        except ValueError as exc:
            if "chat template" not in str(exc).lower():
                raise

    prompt_text = "\n".join(
        [
            f"System: {messages[0]['content']}",
            f"User: {messages[1]['content']}",
            "Assistant:",
        ]
    )
    return tokenizer(prompt_text, truncation=True, max_length=max_length, return_tensors="pt").input_ids


def tokenize_prefix(tokenizer, table_info, n_rows: int, max_length: int) -> torch.Tensor:
    del n_rows
    table_name = table_info["name"]
    columns_text = table_info["columns_text"]
    columns_block = serialize_columns_inline(columns_text)

    user_text = f"Table name: {table_name}\nColumns:\n{columns_block}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    input_ids = tokenize_messages_with_fallback(
        tokenizer, messages, max_length)
    if input_ids.shape[-1] > max_length:
        input_ids = input_ids[:, -max_length:]
    return input_ids


def tokenize_suffixes_batch(tokenizer, col_names, col_text_list, max_length: int):
    texts = []
    for name, col_text in zip(col_names, col_text_list):
        values_str = col_text if col_text else "(empty)"
        texts.append(SUFFIX_TEMPLATE.format(
            col_name=name, col_values=values_str))
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        add_special_tokens=False,
    )
    return encoded.input_ids, encoded.attention_mask


def tokenize_column_prompts_batch(tokenizer, col_names, col_text_list, max_length: int):
    encoded_samples = []
    for name, col_text in zip(col_names, col_text_list):
        values_str = col_text if col_text else "(empty)"
        user_text = NO_PREFIX_TEMPLATE.format(
            col_name=name, col_values=values_str)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]
        input_ids = tokenize_messages_with_fallback(
            tokenizer, messages, max_length)[0]
        if input_ids.shape[0] > max_length:
            input_ids = input_ids[-max_length:]
        encoded_samples.append({"input_ids": input_ids.tolist()})

    padded = tokenizer.pad(encoded_samples, padding=True, return_tensors="pt")
    return padded.input_ids, padded.attention_mask
