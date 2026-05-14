"""Load YAML prompt templates with {{ variable }} injection."""

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"


def load_prompt(template: str, **variables) -> str:
    """Load a YAML prompt template and replace {{ var }} placeholders.

    Args:
        template: path relative to prompts/ dir, e.g. "rag/query"
        **variables: key/value pairs to inject

    Returns:
        The rendered prompt string.
    """
    path = PROMPTS_DIR / f"{template}.yaml"
    if not path.exists():
        logger.warning("Prompt template not found: %s", path)
        return ""

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    # Template is the value of the first key
    text = list(raw.values())[0] if raw else ""

    for k, v in variables.items():
        text = text.replace(f"{{{{ {k} }}}}", str(v))
    return text


def load_system_prompt(name: str = "assistant") -> str:
    """Load a system-level prompt from prompts/{name}.yaml"""
    return load_prompt(name)
