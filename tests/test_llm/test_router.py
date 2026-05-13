"""Tests for the LLM router with mock provider."""

import pytest

from src.llm.router import LLMRouter, MOCK
from src.llm.providers.mock import MockProvider


def test_mock_provider_returns_echo():
    provider = MockProvider()
    assert provider.is_available()

    result = provider.chat([{"role": "user", "content": "hello"}])
    assert "hello" in result
    assert result.startswith("[mock]")


def test_router_with_mock():
    router = LLMRouter()
    router.register(MOCK, MockProvider())

    result = router.chat(
        [{"role": "user", "content": "test question"}],
        provider=MOCK,
    )
    assert "test question" in result


def test_router_fallback_chain():
    """When primary is unavailable, should try fallback chain."""
    router = LLMRouter()
    # Only register mock — deepseek and openai are unavailable
    router.register(MOCK, MockProvider())

    # Request deepseek but it's not registered → falls back to mock
    result = router.chat(
        [{"role": "user", "content": "fallback test"}],
        provider=MOCK,
    )
    assert "fallback test" in result


def test_router_raises_when_all_unavailable():
    router = LLMRouter()
    # No providers registered at all
    with pytest.raises(RuntimeError):
        router.chat(
            [{"role": "user", "content": "should fail"}],
            provider="deepseek",
        )
