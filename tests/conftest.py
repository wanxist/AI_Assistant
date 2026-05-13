"""Shared fixtures for all tests."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.llm.providers.mock import MockProvider
from src.llm.router import get_llm, MOCK


@pytest.fixture
def mock_llm():
    """Ensure the mock LLM provider is registered."""
    llm = get_llm()
    llm.register(MOCK, MockProvider())
    return llm


@pytest.fixture
def client(mock_llm):
    """FastAPI TestClient with mock LLM."""
    return TestClient(app)
