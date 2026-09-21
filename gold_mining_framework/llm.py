"""Shared chat model for gold mining framework agents."""

from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from pydantic import ValidationError

# Load repo-root .env (XAI_API_KEY, TAVILY_API_KEY, etc.)
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# xAI Grok — current flagship used by this project's agents.
DEFAULT_MODEL = "xai:grok-4.7"


def get_chat_model(temperature: float = 0.0) -> BaseChatModel:
    """Return the xAI chat model used by pipeline agents.

    Args:
        temperature: Sampling temperature. Defaults to 0 for deterministic output.

    Returns:
        An initialized LangChain chat model.
    """
    try:
        return init_chat_model(model=DEFAULT_MODEL, temperature=temperature)
    except ValidationError as exc:
        raise RuntimeError(
            "xAI is not configured. Set XAI_API_KEY in the repo-root .env file."
        ) from exc
