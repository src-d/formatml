from bz2 import open as bz2_open
from logging import getLogger
from pathlib import Path
from pickle import dump as pickle_dump, load as pickle_load
from typing import Any, Dict, Optional


class ResourceException(Exception):
    """Resource related exception."""

    pass


class Resource:
    pass


class Context:

    _logger = getLogger(__name__)

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        self._cache_dir = cache_dir
        self._instance_registry: Dict[str, Any] = {}
        self._name_registry: Dict[int, str] = {}

    def register_resource(self, name: str, resource: Any) -> None:
        if name in self._instance_registry:
            raise ResourceException(f"Resource name {name} already in use.")
        if self._cache_dir:
            cache_path = self._name_cache_path(name)
            if cache_path.is_file():
                self._logger.debug(f"Using the cached version of the {name} resource")
                with bz2_open(cache_path, "rb") as fh:
                    resource = pickle_load(fh)
        self._instance_registry[name] = resource
        self._name_registry[id(resource)] = name

    def get_resource(self, name: str) -> Any:
        """Get the resource registered under a given name."""
        if name not in self._instance_registry:
            raise ResourceException(f"Resource {name} not found.")
        if self._cache_dir:
            cache_path = self._name_cache_path(name)
            if cache_path.is_file():
                with bz2_open(cache_path, "rb") as fh:
                    return pickle_load(fh)
        return self._instance_registry[name]

    def save_resources(self) -> None:
        for _name, resource in self._instance_registry.items():
            self.save_resource(resource)

    def save_resource(self, resource: Any) -> None:
        if not self._cache_dir:
            raise ResourceException(
                f"Trying to save a resource but the cache dir was not set."
            )
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        with bz2_open(self._resource_cache_path(resource), "wb") as fh:
            pickle_dump(resource, fh)

    def _name_cache_path(self, name: str) -> Path:
        return self._cache_dir / f"{name}.pickle.bz2"

    def _resource_cache_path(self, resource: Any) -> Path:
        if id(resource) not in self._name_registry:
            raise ResourceException(f"Resource {resource} not found.")
        return self._name_cache_path(self._name_registry[id(resource)])
