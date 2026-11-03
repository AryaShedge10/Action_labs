"""Local LLM client (Ollama) with privacy guard and robust JSON parsing.

* Refuses non-local endpoints unless ALLOW_REMOTE_LLM=true (PHI must not leave the network).
* Never logs prompts or completions — they contain PHI.
* Disables Qwen3 "thinking" output and requests JSON mode, then still parses defensively.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import re
import time
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

_LOCAL_HOSTNAMES = {"localhost", "host.docker.internal", "ollama"}


class LLMError(RuntimeError):
    pass


class LLMUnavailableError(LLMError):
    pass


class PrivacyViolationError(LLMError):
    pass


def is_local_endpoint(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    if host in _LOCAL_HOSTNAMES or host.endswith(".local") or host.endswith(".internal"):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.is_loopback or ip.is_private


def parse_json_object(raw: str) -> dict:
    """Extract the first JSON object from an LLM completion."""
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise LLMError("LLM response did not contain a JSON object") from None
        try:
            data = json.loads(text[start:end + 1])
        except json.JSONDecodeError as exc:
            raise LLMError("LLM response contained malformed JSON") from exc
    if not isinstance(data, dict):
        raise LLMError("LLM response JSON is not an object")
    return data


class OllamaClient:
    def __init__(self, base_url: str, model: str, timeout_s: float = 180.0, allow_remote: bool = False,
                 session: requests.Session | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.session = session or requests.Session()
        self.is_local = is_local_endpoint(self.base_url)
        if not self.is_local and not allow_remote:
            raise PrivacyViolationError(
                f"Refusing to use non-local LLM endpoint {self.base_url!r}; set ALLOW_REMOTE_LLM=true to override")
        self._availability: tuple[float, bool] | None = None

    def is_available(self, cache_s: float = 30.0) -> bool:
        now = time.monotonic()
        if self._availability and now - self._availability[0] < cache_s:
            return self._availability[1]
        ok = False
        try:
            resp = self.session.get(f"{self.base_url}/api/tags", timeout=2)
            if resp.ok:
                names = {m.get("name") for m in resp.json().get("models", [])}
                ok = self.model in names or f"{self.model}:latest" in names
        except (requests.RequestException, ValueError):
            ok = False
        self._availability = (now, ok)
        return ok

    def generate(self, prompt: str, temperature: float = 0.0, json_mode: bool = True) -> str:
        body = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": {"temperature": temperature, "seed": 7, "num_ctx": 8192},
        }
        if json_mode:
            body["format"] = "json"
        try:
            resp = self.session.post(f"{self.base_url}/api/generate", json=body, timeout=self.timeout_s)
        except requests.RequestException as exc:
            self._availability = (time.monotonic(), False)
            raise LLMUnavailableError(f"LLM request failed: {type(exc).__name__}") from exc
        if resp.status_code >= 400:
            raise LLMError(f"LLM returned HTTP {resp.status_code}")
        return resp.json().get("response", "")

    def generate_json(self, prompt: str, temperature: float = 0.0) -> dict:
        return parse_json_object(self.generate(prompt, temperature=temperature, json_mode=True))
