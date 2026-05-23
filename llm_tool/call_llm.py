import time
import torch
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import Dict

from llm_tool.client_pool import CLIENT_POOL
from openai import OpenAIError, RateLimitError
from transformers import pipeline, Pipeline
from llm_tool.util import _TIMEIT_DATA, mask_secret, print_timeit_summary, save_timeit, timeit


LLM_CACHE = None
APIModels = ['qwen2.5-72b-instruct', 'deepseek-v3.2', 'gpt-oss-20b']
TF_PIPELINES: Dict[str, Pipeline] = {}


class EmptyLLMResponseError(RuntimeError):
    pass


def _safe_extract_chat_content(resp) -> str:
    if resp is None:
        raise EmptyLLMResponseError("LLM response is None")

    choices = getattr(resp, "choices", None)
    if not choices:
        resp_id = getattr(resp, "id", None)
        model = getattr(resp, "model", None)
        try:
            dump = resp.model_dump()
        except Exception:
            dump = str(resp)
        raise EmptyLLMResponseError(
            f"LLM response missing choices (id={resp_id}, model={model}) | resp={dump}"
        )

    first = choices[0]
    message = getattr(first, "message", None)
    content = getattr(message, "content",
                      None) if message is not None else None
    if content is None:
        resp_id = getattr(resp, "id", None)
        model = getattr(resp, "model", None)
        finish_reason = getattr(first, "finish_reason", None)
        tool_calls = getattr(message, "tool_calls",
                             None) if message is not None else None
        raise EmptyLLMResponseError(
            f"LLM response has no content (id={resp_id}, model={model}, finish_reason={finish_reason}, tool_calls={tool_calls})"
        )

    return content.strip()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1))
def call_llm_api(model, msgs, temperature=0.1, max_tokens=2*1024, **args):
    resp = None
    if LLM_CACHE:
        is_hit, cache = LLM_CACHE.get_cache_by_params(
            prompt=msgs,
            temperature=temperature
        )
        if is_hit:
            return cache
    key, CLIENT = CLIENT_POOL.get_client()
    key_label = mask_secret(key)
    t0 = time.time()
    try:
        resp = CLIENT.chat.completions.create(
            model=model,
            messages=msgs,
            temperature=temperature,
            max_tokens=max_tokens
        )
        pred_str = _safe_extract_chat_content(resp)
        CLIENT_POOL.mark_success(key, latency=time.time() - t0)
        if LLM_CACHE:
            _ = LLM_CACHE.set_cache_by_params(
                prompt=msgs,
                temperature=temperature,
                cache=pred_str
            )
    except RateLimitError:
        CLIENT_POOL.cooldown(key, seconds=3, escalate=True)
        print(f"API key {key_label} rate limited; switching to another key")
        raise
    except OpenAIError as e:
        CLIENT_POOL.cooldown(key, seconds=3, escalate=True)
        status = getattr(getattr(e, "response", None), "status_code", None)
        body = None
        if getattr(e, "response", None) is not None:
            try:
                body = e.response.json()
            except Exception:
                try:
                    body = e.response.text
                except Exception:
                    body = None
        print(
            f"API key {key_label} failed; switching to another key | "
            f"type={e.__class__.__name__} status={status} message={e} body={body}"
        )
        import traceback
        traceback.print_exc()
        raise
    except Exception as e:
        CLIENT_POOL.cooldown(key, seconds=3, escalate=True)
        resp_id = getattr(resp, "id", None) if resp is not None else None
        resp_model = getattr(resp, "model", None) if resp is not None else None
        try:
            resp_dump = resp.model_dump() if resp is not None else None
        except Exception:
            resp_dump = str(resp)
        print(
            f"API key {key_label} returned an invalid response; switching to another key | "
            f"type={e.__class__.__name__} message={e} resp_id={resp_id} resp_model={resp_model} resp={resp_dump}"
        )
        import traceback
        traceback.print_exc()
        raise
    return pred_str


def _format_msgs_for_local_model(msgs, tokenizer=None):
    if isinstance(msgs, str):
        return msgs

    if tokenizer and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception as e:
            print(
                f"Warning: apply_chat_template failed: {e}; falling back to simple format")

    parts = []
    for msg in msgs:
        role = msg.get("role", "user").capitalize()
        content = msg.get("content", "")
        parts.append(f"{role}: {content}")
    parts.append("Assistant:")
    return "\n".join(parts)


@timeit(log=False)
def _get_text_generation_pipeline(model_name: str) -> Pipeline:
    if model_name not in TF_PIPELINES:
        torch_dtype = torch.float16 if torch.cuda.is_available() else None
        TF_PIPELINES[model_name] = pipeline(
            task="text-generation",
            model=model_name,
            dtype=torch_dtype,
            trust_remote_code=True,
            device_map="auto"
        )
    return TF_PIPELINES[model_name]


@timeit(log=False)
def call_llm_tf(model, msgs, temperature=0.1, max_tokens=2*1024, **generation_kwargs):
    if LLM_CACHE:
        is_hit, cache = LLM_CACHE.get_cache_by_params(
            prompt=msgs,
            temperature=temperature
        )
        if is_hit:
            return cache

    text_pipe = _get_text_generation_pipeline(model)
    tokenizer = getattr(text_pipe, "tokenizer", None)

    prompt = _format_msgs_for_local_model(msgs, tokenizer)
    gen_kwargs = {
        "max_new_tokens": max_tokens,
        "temperature": temperature,
        "do_sample": temperature > 0,
        "return_full_text": False,
        "repetition_penalty": 1.15,
    }
    if tokenizer:
        if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
            gen_kwargs.setdefault("eos_token_id", tokenizer.eos_token_id)
        if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None:
            gen_kwargs.setdefault("pad_token_id", tokenizer.pad_token_id)
        else:
            gen_kwargs.setdefault("pad_token_id", tokenizer.eos_token_id)
    gen_kwargs.update(generation_kwargs)

    outputs = text_pipe(prompt, **gen_kwargs)
    pred_str = outputs[0].get("generated_text", "").strip()

    if LLM_CACHE:
        _ = LLM_CACHE.set_cache_by_params(
            prompt=msgs,
            temperature=temperature,
            cache=pred_str
        )
    return pred_str


@timeit(log=False)
def call_llm(model, msgs, temperature=0.1, max_tokens=2*1024, **args):
    if model in APIModels:
        return call_llm_api(
            model,
            msgs,
            temperature=temperature,
            max_tokens=max_tokens,
            **args
        )
    return call_llm_tf(
        model,
        msgs,
        temperature=temperature,
        max_tokens=max_tokens,
        **args
    )
