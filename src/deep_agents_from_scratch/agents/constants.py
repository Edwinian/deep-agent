"""Agent constants."""

from enum import Enum


class DeepAgentField(str, Enum):
    """Fields defined on :class:`~deep_agents_from_scratch.agents.types.DeepAgent` only."""

    ID = "id"
    SUBAGENTS = "subagents"
    MODEL_CONFIG = "model_config"


DEEP_AGENT_ONLY_FIELDS = frozenset(field.value for field in DeepAgentField)
