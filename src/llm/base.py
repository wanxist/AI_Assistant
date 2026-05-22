"""(Sync) abstract base for LLM providers."""

from abc import ABC, abstractmethod
from collections.abc import Iterator


class BaseLLMProvider(ABC):
    """Every LLM backend implements this interface."""

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        stream: bool = False,
    ) -> str:
        """Send a chat completion request and return the text response."""
        ...

    def chat_stream(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> Iterator[str]:
        """Yield text chunks from a streaming chat completion.

        Default: call chat() with stream=True and yield the result as one chunk.
        Providers that support native streaming should override this.
        """
        text = self.chat(
            messages, model=model, temperature=temperature,
            max_tokens=max_tokens, stream=True,
        )
        if text:
            yield text

    @abstractmethod
    def is_available(self) -> bool:
        """Check whether the provider is configured and reachable."""
        ...
