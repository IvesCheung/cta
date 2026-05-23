import os
import random
import threading
import time
from collections import Counter

from openai import OpenAI


def _read_env_list(name: str):
    raw = os.getenv(name, "")
    return [item.strip() for item in raw.replace("\n", ",").split(",") if item.strip()]


API_KEYS = _read_env_list("SEMANTICCTA_LLM_API_KEYS")
BASE_URL = os.getenv("SEMANTICCTA_LLM_BASE_URL") or None
CLIENT_TIMEOUT = float(os.getenv("SEMANTICCTA_LLM_TIMEOUT", "40"))
DEFAULT_MODE = os.getenv("SEMANTICCTA_LLM_KEY_MODE", "round_robin")


class KeyRotator:
    def __init__(self, api_keys, mode="round_robin"):
        self.mode = mode
        self._lock = threading.Lock()
        self._weights = Counter(api_keys)
        unique_keys = list(self._weights.keys())
        self._meta = {
            k: {
                "cool_until": 0.0,
                "use_count": 0,
                "last_used": 0.0,
                "fail_streak": 0,
                "weight": self._weights[k],
                "avg_latency": None,
            } for k in unique_keys
        }
        self._last_key = None
        self._clients = {}
        for key in unique_keys:
            kwargs = {"api_key": key, "timeout": CLIENT_TIMEOUT}
            if BASE_URL:
                kwargs["base_url"] = BASE_URL
            self._clients[key] = OpenAI(**kwargs)

    def _eligible_keys(self, now):
        return [k for k, m in self._meta.items() if m["cool_until"] <= now]

    def _score(self, k):
        m = self._meta[k]
        latency = m["avg_latency"] if m["avg_latency"] is not None else 0
        return (m["use_count"] / m["weight"], latency, m["last_used"])

    def _pick_key(self):
        now = time.time()
        eligible = self._eligible_keys(now)
        if not eligible:
            k = min(self._meta.items(), key=lambda kv: kv[1]["cool_until"])[0]
            return k
        if self.mode == "random":
            weights = []
            for k in eligible:
                w = self._meta[k]["weight"]
                lat = self._meta[k]["avg_latency"]
                if lat and lat > 30:
                    w *= 0.5
                weights.append(w)
            total = sum(weights)
            r = random.uniform(0, total)
            acc = 0
            for k, w in zip(eligible, weights):
                acc += w
                if acc >= r:
                    return k
            return eligible[-1]
        eligible.sort(key=lambda k: self._score(k))
        return eligible[0]

    def get_client(self):
        with self._lock:
            if not self._clients:
                raise RuntimeError(
                    "No LLM API key configured. Set SEMANTICCTA_LLM_API_KEYS as a comma-separated list."
                )
            k = self._pick_key()
            self._meta[k]["use_count"] += 1
            self._meta[k]["last_used"] = time.time()
            self._last_key = k
            return k, self._clients[k]

    def cooldown(self, key, seconds=3, escalate=False):
        with self._lock:
            m = self._meta[key]
            if escalate:
                m["fail_streak"] += 1
            else:
                m["fail_streak"] = 0
            extra = (2 ** (m["fail_streak"] - 1)
                     ) if m["fail_streak"] > 0 else 0
            adj = seconds + extra
            adj /= m["weight"]
            m["cool_until"] = time.time() + adj

    def mark_success(self, key, latency=None):
        with self._lock:
            m = self._meta[key]
            m["fail_streak"] = 0
            if latency is not None:
                if m["avg_latency"] is None:
                    m["avg_latency"] = latency
                else:
                    m["avg_latency"] = m["avg_latency"] * 0.7 + latency * 0.3


CLIENT_POOL = KeyRotator(API_KEYS, mode=DEFAULT_MODE)
