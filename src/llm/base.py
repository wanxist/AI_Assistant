"""(Sync) abstract base for LLM providers."""

from abc import ABC, abstractmethod


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

    @abstractmethod
    def is_available(self) -> bool:
        """Check whether the provider is configured and reachable."""
        ...
