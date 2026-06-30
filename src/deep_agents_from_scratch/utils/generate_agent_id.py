"""Generate integer IDs for agent specs."""

import random

from deep_agents_from_scratch.agents.agent_registry import AGENT_REGISTRY


def generate_agent_id() -> int:
    """Return a random agent ID between 1000 and 9999 not in AGENT_REGISTRY."""
    existing_ids = set(AGENT_REGISTRY.keys())
    new_id = random.randint(1000, 9999)

    while new_id in existing_ids:
        new_id = random.randint(1000, 9999)

    return new_id
