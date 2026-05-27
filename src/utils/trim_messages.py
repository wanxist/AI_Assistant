"""Trim conversation history to fit within token budget and round limits."""

import logging

import tiktoken

logger = logging.getLogger(__name__)


def trim_messages(
    messages: list[dict],
    max_tokens: int = 8000,
    max_rounds: int = 30,
) -> list[dict]:
    """Keep the latest messages from the tail, bounded by token count and round count.

    System prompt (role="system" at index 0) is always preserved.
    One round = one user message and the assistant response(s) that follow it.
    """
    start = 1 if messages and messages[0].get("role") == "system" else 0

    enc = tiktoken.get_encoding("cl100k_base")
    result = list(messages[:start])
    tokens_used = sum(len(enc.encode(m.get("content", ""))) for m in result)

    kept = []
    rounds = 0
    for msg in reversed(messages[start:]):
        t = len(enc.encode(msg.get("content", "")))
        if tokens_used + t > max_tokens or rounds >= max_rounds:
            break
        kept.insert(0, msg)
        tokens_used += t
        if msg.get("role") == "user":
            rounds += 1

    trimmed = len(messages) - start - len(kept)
    if trimmed:
        logger.debug("Trimmed %d messages from context (%d tokens, %d rounds kept)", trimmed, tokens_used, rounds)

    return result + kept
