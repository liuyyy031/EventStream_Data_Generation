"""
Generic OpenAI-compatible LLM client used by the data-generation pipeline.

Configuration is read from environment variables:

    LLM_API_KEY    (required)  API key (sent as `Authorization: Bearer ...`).
    LLM_BASE_URL   (optional)  Default: https://api.openai.com/v1
    LLM_MODEL      (optional)  Default: gpt-4o-mini

Any provider that exposes an OpenAI-style ``/chat/completions`` endpoint can
be plugged in by setting ``LLM_BASE_URL`` (e.g. OpenAI, OpenRouter, DeepSeek,
Together AI, vLLM, Ollama, etc.).
"""

from __future__ import annotations

import os
import time
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

MY_KEY = os.getenv("LLM_API_KEY")
# DEFAULT_BASE_URL = "https://api.openai.com/v1"
# DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_TIMEOUT = 120
DEFAULT_MAX_RETRIES = 5


class LLMClient:
    """Thin wrapper around OpenAI-compatible chat completions API."""

    def __init__(
        self,
        api_key: Optional[str] = MY_KEY,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        self.api_key = api_key or os.environ.get("LLM_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "LLM_API_KEY is not set. Export it before running, e.g.:\n"
                "  export LLM_API_KEY=<your_api_key>"
            )

        self.base_url = (base_url or os.environ.get("LLM_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.model = model or os.environ.get("LLM_MODEL") or DEFAULT_MODEL
        self.timeout = timeout
        self.max_retries = max_retries

    # ------------------------------------------------------------------ basics

    def complete(self, prompt: str, max_tokens: int = 4096, temperature: float = 1.0) -> str:
        """Single-turn text completion."""
        messages = [{"role": "user", "content": prompt}]
        return self._chat(messages, max_tokens=max_tokens, temperature=temperature)

    def complete_with_image(
        self,
        prompt: str,
        image_base64: str,
        image_media_type: str = "image/png",
        max_tokens: int = 4096,
        temperature: float = 1.0,
    ) -> str:
        """Single-turn multimodal completion (text + one image)."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{image_media_type};base64,{image_base64}"},
                    },
                ],
            }
        ]
        return self._chat(messages, max_tokens=max_tokens, temperature=temperature)

    # ----------------------------------------------------------------- batched

    def run_batch(
        self,
        requests_list: List[Dict[str, Any]],
        max_workers: int = 8,
    ) -> List[Dict[str, Any]]:
        """
        Run many requests concurrently and return results in the same shape
        previously produced by AWS Bedrock Batch API:

            {"recordId": <id>, "modelOutput": {"content": [{"text": <reply>}]}}

        On failure:

            {"recordId": <id>, "error": <error_message>}

        ``requests_list`` items must follow the schema:

            {
              "recordId": "<unique_id>",
              "modelInput": {
                  "max_tokens": <int>,
                  "messages": [
                      {"role": "user",
                       "content": [
                           {"type": "text", "text": "..."},
                           # optional image:
                           {"type": "image",
                            "source": {"type": "base64",
                                       "media_type": "image/png",
                                       "data": "<base64>"}}
                       ]}
                  ]
              }
            }

        This is the legacy Anthropic/Bedrock-style schema – we translate it to
        OpenAI chat-completions format internally.
        """
        if not requests_list:
            return []

        results: List[Optional[Dict[str, Any]]] = [None] * len(requests_list)

        def _worker(idx: int, req: Dict[str, Any]) -> None:
            record_id = req.get("recordId", f"req_{idx}")
            try:
                model_input = req["modelInput"]
                openai_messages = _to_openai_messages(model_input.get("messages", []))
                max_tokens = model_input.get("max_tokens", 4096)
                text = self._chat(openai_messages, max_tokens=max_tokens)
                results[idx] = {
                    "recordId": record_id,
                    "modelOutput": {"content": [{"text": text}]},
                }
            except Exception as exc:  # noqa: BLE001
                results[idx] = {"recordId": record_id, "error": str(exc)}

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                pool.submit(_worker, idx, req)
                for idx, req in enumerate(requests_list)
            ]
            for _ in as_completed(futures):
                pass

        return [r for r in results if r is not None]

    # -------------------------------------------------------------- internals

    def _chat(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        temperature: float = 1.0,
    ) -> str:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.model.startswith("deepseek-v4-"):
            payload["thinking"] = {
                "type": "disabled"
            }

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                '''if resp.status_code == 200:
                    data = resp.json()
                    return data["choices"][0]["message"]["content"]'''
                
                if resp.status_code == 200:
                    data = resp.json()

                    content = data["choices"][0]["message"].get("content")

                    if content and content.strip():
                        return content

                    last_err = RuntimeError(
                        f"HTTP 200 but model returned empty content: "
                        f"{json.dumps(data, ensure_ascii=False)[:1000]}"
                    )

                if resp.status_code in (408, 429) or 500 <= resp.status_code < 600:
                    last_err = RuntimeError(
                        f"HTTP {resp.status_code}: {resp.text[:300]}"
                    )
                else:
                    raise RuntimeError(
                        f"HTTP {resp.status_code}: {resp.text[:500]}"
                    )
            except (requests.RequestException, json.JSONDecodeError, KeyError) as exc:
                last_err = exc

            backoff = (2 ** attempt) + random.uniform(0, 1)
            time.sleep(backoff)

        raise RuntimeError(
            f"LLM request failed after {self.max_retries} retries: {last_err}"
        )


# --------------------------------------------------------------------- helpers


def _to_openai_messages(legacy_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Translate Anthropic/Bedrock-style messages to OpenAI chat format."""
    converted: List[Dict[str, Any]] = []
    for msg in legacy_messages:
        role = msg.get("role", "user")
        content = msg.get("content")

        if isinstance(content, str):
            converted.append({"role": role, "content": content})
            continue

        if not isinstance(content, list):
            converted.append({"role": role, "content": str(content)})
            continue

        new_parts: List[Dict[str, Any]] = []
        for part in content:
            ptype = part.get("type")
            if ptype == "text":
                new_parts.append({"type": "text", "text": part.get("text", "")})
            elif ptype == "image":
                source = part.get("source", {})
                if source.get("type") == "base64":
                    media_type = source.get("media_type", "image/png")
                    data = source.get("data", "")
                    new_parts.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{media_type};base64,{data}"},
                        }
                    )
            elif ptype == "image_url":
                new_parts.append(part)

        if len(new_parts) == 1 and new_parts[0]["type"] == "text":
            converted.append({"role": role, "content": new_parts[0]["text"]})
        else:
            converted.append({"role": role, "content": new_parts})

    return converted
