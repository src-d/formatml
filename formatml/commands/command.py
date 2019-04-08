from typing import Any, Dict

from formatml.resources.resource import Context
from formatml.utils.from_params import from_params


@from_params
class Command:
    """Base class for all commands."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    def run(self, context: Context) -> None:
        raise NotImplementedError()
