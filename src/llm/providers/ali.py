import logging

from openai import OpenAI

from src.config import settings
from src.llm.base import BaseLLMProvider
from src.utils.ssl_utils import get_httpx_client

logger = logging.getLogger(__name__)


class AliProvider(BaseLLMProvider):
    """Alibaba Cloud DashScope LLM provider (OpenAI-compatible)."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        default_model: str | None = None,
    ):
        self.api_key = api_key or settings.ali_api_key
        self.base_url = base_url or settings.ali_base_url
        self.default_model = default_model or settings.ali_chat_model
        self._client: OpenAI | None = None

    def _ensure_client(self) -> OpenAI:
        if self._client is None:
            http_client = get_httpx_client()
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url, http_client=http_client)
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
        if stream:
            return "".join(self.chat_stream(
                messages, model=model, temperature=temperature, max_tokens=max_tokens,
            ))
        client = self._ensure_client()
        response = client.chat.completions.create(
            model=model or self.default_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )
        return response.choices[0].message.content or ""

    def chat_stream(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ):
        client = self._ensure_client()
        response = client.chat.completions.create(
            model=model or self.default_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
