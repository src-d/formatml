from pathlib import Path
from typing import Any, Dict

from formatml.resources.vocabulary import Vocabulary
from formatml.utils.helpers import date_template_to_path
from formatml.utils.registrable import register


class ResourceException(Exception):
    """Resource related exception."""

    pass


class Resource:
    pass


register(cls=Resource, name="vocabulary")(Vocabulary)
register(
    cls=Resource,
    name="date_template_path",
    factory=date_template_to_path,
    no_from_params=False,
)(Path)


class Context:
    def __init__(self) -> None:
        self._instance_registry: Dict[str, Any] = {}

    def register_resource(self, name: str, resource: Any) -> None:
        if name in self._instance_registry:
            raise ResourceException(f"Resource name {name} already in use.")
        self._instance_registry[name] = resource

    def get_resource(self, name: str) -> Any:
        """Get the resource registered under a given name."""
        if name not in self._instance_registry:
            raise ResourceException(f"Resource {name} not found.")
        return self._instance_registry[name]
