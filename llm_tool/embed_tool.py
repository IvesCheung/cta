import json
import os
from typing import List, Union

import numpy as np
import requests
from tenacity import retry, stop_after_attempt, wait_exponential


def _read_env_list(name: str):
    raw = os.getenv(name, "")
    return [item.strip() for item in raw.replace("\n", ",").split(",") if item.strip()]


USE_LOCAL_MODEL = os.getenv("SEMANTICCTA_EMBED_USE_LOCAL", "true").lower() in {
    "1", "true", "yes"}
LOCAL_MODEL_DEVICE = os.getenv("SEMANTICCTA_EMBED_DEVICE", "cuda:0")

modelname2path = {
    "qwen3-embedding-0.6b": os.getenv("SEMANTICCTA_EMBED_LOCAL_QWEN3_06B", "./model/qwen3-0.6B-embedding"),
    "qwen3-embedding-4b": os.getenv("SEMANTICCTA_EMBED_LOCAL_QWEN3_4B", ""),
}

_local_model_cache: dict = {}

MAX_PROCESS_NUM = int(os.getenv("SEMANTICCTA_EMBED_BATCH_SIZE", "256"))
TARGET_MODEL = os.getenv("SEMANTICCTA_EMBED_MODEL", "qwen3-embedding-4b")


def _get_embedding_config(model: str):
    config = {
        "qwen3-embedding-0.6b": {
            "url": os.getenv("SEMANTICCTA_EMBED_URL_QWEN3_06B", ""),
            "auth_tokens": _read_env_list("SEMANTICCTA_EMBED_AUTH_QWEN3_06B"),
        },
        "qwen3-embedding-4b": {
            "url": os.getenv("SEMANTICCTA_EMBED_URL_QWEN3_4B", ""),
            "auth_tokens": _read_env_list("SEMANTICCTA_EMBED_AUTH_QWEN3_4B"),
        },
    }
    if model not in config:
        raise EmbeddingAPIError(f"Unsupported embedding model: {model}")
    model_cfg = config[model]
    if not model_cfg["url"]:
        raise EmbeddingAPIError(
            f"Missing embedding endpoint for {model}. Set the corresponding SEMANTICCTA_EMBED_URL_* environment variable."
        )
    return {
        "url": model_cfg["url"],
        "model": model,
        "encoding_format": "float",
        "auth_tokens": model_cfg["auth_tokens"],
    }


class EmbeddingAPIError(Exception):
    pass


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1), reraise=True)
def _fetch_embedding_batch(model: str,
                           batch: List[str],
                           timeout: int = 10) -> List[List[float]]:
    config = _get_embedding_config(model)
    url = config['url']
    headers = {'Content-Type': 'application/json'}
    if config['auth_tokens']:
        headers['Authorization'] = config['auth_tokens'][np.random.randint(
            0, len(config['auth_tokens']))]

    payload = {
        "model": model,
        "input": batch,
        "encoding_format": "float"
    }

    try:
        resp = requests.post(
            url, json=payload, headers=headers, timeout=timeout)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise EmbeddingAPIError(f"HTTP请求失败: {e}") from e

    try:
        data = resp.json()
    except json.JSONDecodeError as e:
        raise EmbeddingAPIError(f"响应JSON解析失败: {e}") from e

    if 'data' not in data or not data['data']:
        raise EmbeddingAPIError(f"响应缺少 data 字段或为空: {data}")
    if 'embedding' not in data['data'][0]:
        raise EmbeddingAPIError(f"响应缺少 embedding 字段: {data}")

    embeddings = [item['embedding'] for item in data['data']]

    cleaned = []
    for emb in embeddings:
        if any(np.isnan(x) for x in emb):
            emb = [0.0 if (isinstance(x, float) and np.isnan(x))
                   else x for x in emb]
        cleaned.append(emb)
    return cleaned


def _get_or_load_local_model(model_name: str):
    if model_name in _local_model_cache:
        return _local_model_cache[model_name]

    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
    except ImportError as e:
        raise ImportError(
            "Local embedding inference requires torch and transformers") from e

    model_path = modelname2path[model_name]
    if not model_path:
        raise EmbeddingAPIError(
            f"No local model path configured for {model_name}")
    print(
        f"[embed_tool] loading local model {model_name} from {model_path} on {LOCAL_MODEL_DEVICE}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path, dtype=torch.float16)
    model.to(LOCAL_MODEL_DEVICE)
    model.eval()
    _local_model_cache[model_name] = (tokenizer, model)
    return tokenizer, model


def _last_token_pool(last_hidden_states, attention_mask):
    if attention_mask[:, -1].sum() == attention_mask.shape[0]:
        return last_hidden_states[:, -1]
    import torch
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths,
    ]


def _fetch_embedding_batch_local(model_name: str, batch: List[str]) -> List[List[float]]:
    import torch
    import torch.nn.functional as F

    tokenizer, model = _get_or_load_local_model(model_name)

    encoded = tokenizer(
        batch,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(LOCAL_MODEL_DEVICE)

    with torch.no_grad():
        outputs = model(**encoded)
        embeddings = _last_token_pool(
            outputs.last_hidden_state, encoded["attention_mask"]
        )
    embeddings = F.normalize(embeddings, p=2, dim=1)

    cleaned = []
    for emb in embeddings.cpu().float().numpy():
        emb_list = emb.tolist()
        if any(np.isnan(x) for x in emb_list):
            emb_list = [0.0 if np.isnan(x) else x for x in emb_list]
        cleaned.append(emb_list)
    return cleaned


def get_embeddings(texts: List[str], model: str = TARGET_MODEL, batch_size=MAX_PROCESS_NUM) -> Union[List[List[float]], str]:
    use_local = USE_LOCAL_MODEL and model in modelname2path and bool(
        modelname2path.get(model))

    all_embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    for i in range(0, len(texts), batch_size):
        batch = texts[i:min(i + batch_size, len(texts))]
        current_batch = i // batch_size + 1
        try:
            if use_local:
                embeddings = _fetch_embedding_batch_local(model, batch)
            else:
                embeddings = _fetch_embedding_batch(model=model, batch=batch)
            all_embeddings.extend(embeddings)
        except Exception as e:
            print(f"\n批次 {current_batch} 最终失败: {e}")
            raise

    print()
    return all_embeddings


def get_embeddings_by_model(texts, model, batch_size=64):
    all_embeddings = []
    model.eval()
    total_batches = (len(texts) + batch_size - 1) // batch_size
    print(f"总文本数量: {len(texts)}", f"批次大小: {batch_size}")
    for i in range(0, len(texts), batch_size):
        batch = texts[i:min(i + batch_size, len(texts))]
        current_batch = i // batch_size + 1
        progress = current_batch / total_batches
        bar_length = 20
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        print(
            f'\r[{bar}] {progress:.0%} 批次 {current_batch}/{total_batches}', end='')
        output = model(batch)
        output = output.tolist()
        all_embeddings.extend(output)
    return all_embeddings


def embedding_L2_normalization(embeddings: List[List[float]]):
    embeddings_array = np.array(embeddings)
    norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
    embeddings_array = embeddings_array / norms
    return embeddings_array


def get_cosine_similarity_matrix(embeddings: List[List[float]]):
    embeddings_array = np.array(embeddings)
    cosine_sim = np.dot(embeddings_array, embeddings_array.T)
    return cosine_sim


def cosine_similarity(a, b):
    dot_product = sum(a[i] * b[i] for i in range(len(a)))
    norm_a = sum(a[i] ** 2 for i in range(len(a))) ** 0.5
    norm_b = sum(b[i] ** 2 for i in range(len(b))) ** 0.5
    return dot_product / (norm_a * norm_b)


def euclidean_distance(a, b):
    return sum((a[i] - b[i]) ** 2 for i in range(len(a))) ** 0.5
