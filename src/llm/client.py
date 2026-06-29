from __future__ import annotations

import json
import logging
from typing import Any

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings
from src.llm.cost_tracker import CostTracker

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Thin wrapper around the OpenAI SDK that is compatible with LiteLLM proxies.
    Automatically records token usage in the shared CostTracker.
    """

    def __init__(self, tracker: CostTracker) -> None:
        self._client = OpenAI(
            api_key=settings.litellm_api_key,
            base_url=settings.litellm_base_url,
        )
        self.tracker = tracker

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        stage: str,
        source_name: str = "",
        entity_id: str = "",
        response_format: dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> str:
        """
        Call the LLM and return the response text.
        Token usage is automatically recorded in CostTracker.
        """
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format:
            kwargs["response_format"] = response_format

        response = self._client.chat.completions.create(**kwargs)
        usage = response.usage
        if usage:
            self.tracker.record(
                stage=stage,
                model=model,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                source_name=source_name,
                entity_id=entity_id,
            )
        content = response.choices[0].message.content or ""
        return content

    def chat_json(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        stage: str,
        source_name: str = "",
        entity_id: str = "",
        temperature: float = 0.0,
    ) -> Any:
        """Convenience wrapper that requests JSON output and parses the response."""
        raw = self.chat(
            model=model,
            messages=messages,
            stage=stage,
            source_name=source_name,
            entity_id=entity_id,
            response_format={"type": "json_object"},
            temperature=temperature,
        )
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("LLM returned non-JSON for stage=%s, returning raw string.", stage)
            return {"raw": raw}
