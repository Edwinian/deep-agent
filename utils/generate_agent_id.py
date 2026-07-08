"""Generate integer IDs for agent specs."""

import random

from db.agent_store import list_agent_ids


def generate_agent_id() -> int:
    """Return a random agent ID between 1000 and 9999 not in the agent database."""
    existing_ids = set(list_agent_ids())
    new_id = random.randint(1000, 9999)

    while new_id in existing_ids:
        new_id = random.randint(1000, 9999)

    return new_id
