import logging

from src.config import settings
from src.llm.base import BaseLLMProvider

logger = logging.getLogger(__name__)


class ZhipuProvider(BaseLLMProvider):
    """Zhipu (智谱 GLM) provider via the zai SDK."""

    def __init__(self, api_key: str | None = None, default_model: str | None = None):
        self.api_key = api_key or settings.zhipu_api_key
        self.default_model = default_model or settings.zhipu_model
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from zai import ZhipuAiClient
            self._client = ZhipuAiClient(api_key=self.api_key)
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
