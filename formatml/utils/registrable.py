from enum import Enum
from logging import getLogger
from typing import Any, Callable, Dict, Type, TypeVar

from formatml.utils.from_params import from_params


class RegistrationException(Exception):
    """Registration exception."""

    pass


_T = TypeVar("_T")
_U = TypeVar("_U")
_Enum = TypeVar("_Enum", bound=Enum)
_logger = getLogger(__name__)
_registry: Dict[Type[Any], Dict[str, Callable[..., Any]]] = {}


def register(
    cls: Type[_T],
    name: str,
    no_from_params: bool = False,
    factory: Callable[..., _T] = None,
) -> Callable[[Type[_U]], Type[_U]]:
    if cls not in _registry:
        _registry[cls] = {}
    registry = _registry[cls]

    def add_subclass_to_registry(registered_class: Type[_U]) -> Type[_U]:
        if name in registry:
            raise RegistrationException(
                f"Cannot register {name} as {cls.__name__}: "
                f"name already in use for {registry[name].__name__}"
            )
        registry[name] = (  # type: ignore
            registered_class if factory is None else factory
        )
        return registered_class if no_from_params else from_params(registered_class)

    return add_subclass_to_registry


def register_from_enum(enum: Type[_Enum]) -> Type[_Enum]:
    from_params(enum)
    for enum_member in enum:
        register(cls=enum, name=enum_member.value)
    return enum


def by_name(cls: Type[_T], name: str) -> Callable[..., _U]:
    if name not in _registry[cls]:
        raise RegistrationException(
            f"{name} is not a registered name for {cls.__name__}"
        )
    subclass = _registry[cls][name]
    _logger.debug(f"Using implementation {subclass.__name__} of {cls.__name__}")
    return subclass


def has_registrations(cls: Type[_T]) -> bool:
    return cls in _registry
