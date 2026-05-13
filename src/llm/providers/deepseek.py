import time
import logging

from openai import OpenAI

from src.config import settings
from src.llm.base import BaseLLMProvider

logger = logging.getLogger(__name__)


class DeepSeekProvider(BaseLLMProvider):
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        default_model: str | None = None,
    ):
        self.api_key = api_key or settings.deepseek_api_key
        self.base_url = base_url or settings.deepseek_base_url
        self.default_model = default_model or settings.deepseek_model
        self._client: OpenAI | None = None

    def _ensure_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def is_available(self) -> bool:
        return bool(self.api_key)

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        stream: bool = False,
    ) -> str:
        client = self._ensure_client()

        response = client.chat.completions.create(
            model=model or self.default_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
        )

        if stream:
            collected = []
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    collected.append(chunk.choices[0].delta.content)
            return "".join(collected)

        return response.choices[0].message.content or ""


class DeepSeekProviderV4(DeepSeekProvider):
    """DeepSeek V4 with reasoning_effort and thinking support."""

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        stream: bool = False,
        reasoning_effort: str = "high",
        thinking_enabled: bool = True,
    ) -> str:
        client = self._ensure_client()

        extra_body = {}
        if thinking_enabled:
            extra_body["thinking"] = {"type": "enabled"}

        response = client.chat.completions.create(
            model=model or self.default_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
            reasoning_effort=reasoning_effort,
            extra_body=extra_body,
        )

        return response.choices[0].message.content or ""
