"""Agent constants."""

from enum import Enum


class DeepAgentField(str, Enum):
    """Fields defined on :class:`~agents.types.DeepAgent` only."""

    SUBAGENTS = "subagents"
    MODEL_CONFIG = "model_config"
    SKILL_PATHS = "skill_paths"


DEEP_AGENT_ONLY_FIELDS = frozenset(field.value for field in DeepAgentField)
