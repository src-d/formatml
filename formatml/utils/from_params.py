from inspect import (
    isclass,
    isfunction,
    ismethod,
    Parameter,
    Signature,
    signature as inspect_signature,
)
from logging import getLogger
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union


_logger = getLogger(__name__)  # pylint: disable=invalid-name


class ConfigurationException(Exception):
    """Configuration exception."""

    pass


_T = TypeVar("_T")
_NO_DEFAULT = Parameter.empty


def _compute_signature(obj: Any) -> Signature:
    if isclass(obj):
        return inspect_signature(obj.__init__)
    elif ismethod(obj) or isfunction(obj):
        return inspect_signature(obj)
    raise ConfigurationException(f"Object {obj} is not callable")


def _takes_arg(obj: Any, arg: str) -> bool:
    """
    Checks whether the provided obj takes a certain arg.
    If it's a class, we're really checking whether its constructor does.
    If it's a function or method, we're checking the object itself.
    Otherwise, we raise an error.
    """
    return arg in _compute_signature(obj).parameters


def _takes_kwargs(obj: Any) -> bool:
    """
    Checks whether a provided object takes in any positional arguments.
    Similar to takes_arg, we do this for both the __init__ function of
    the class or a function / method
    Otherwise, we raise an error
    """
    return bool(
        any(
            p.kind == Parameter.VAR_KEYWORD
            for p in _compute_signature(obj).parameters.values()
        )
    )


def _extract_optional(annotation: Type) -> Tuple[bool, Type]:
    """Extract the type argument to Optional when applicable."""
    origin = getattr(annotation, "__origin__", None)
    args = getattr(annotation, "__args__", ())
    if origin == Union and len(args) == 2 and issubclass(args[1], type(None)):
        return True, args[0]
    else:
        return False, annotation


def _create_kwargs(
    factory: Callable[..., _T],
    parameters: Dict[str, Any],
    context: Any,
    extras: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Create a dict of keyword args suitable for passing to the class's constructor.

    The function does this by finding the class's constructor, matching the constructor
    arguments to entries in the `params` object, and instantiating values for the
    parameters using the type annotation and possibly a from_params method.
    """
    # Get the signature of the constructor.
    signature = inspect_signature(
        factory.__init__ if isclass(factory) else factory  # type: ignore
    )
    kwargs: Dict[str, Any] = {}

    # Iterate over all the constructor parameters and their annotations.
    for param_name, param in signature.parameters.items():
        # Skip "self". You're not *required* to call the first parameter "self",
        # so in theory this logic is fragile, but if you don't call the self parameter
        # "self" you kind of deserve what happens.
        if param_name == "self":
            continue

        # If the annotation is a compound type like typing.Dict[str, int],
        # it will have an __origin__ field indicating `typing.Dict`
        # and an __args__ field indicating `(str, int)`. We capture both.
        kwargs[param_name] = _construct_arg(
            factory,
            param_name,
            param.annotation,
            param.default,
            parameters,
            context,
            extras,
        )

    return kwargs


def _construct_arg(
    factory: Callable[..., _T],
    param_name: str,
    annotation: Type,
    param_default: Any,
    parameters: Dict[str, Any],
    context: Any,
    extras: Optional[Dict[str, Any]],
) -> Any:
    """
    Construct an individual argument for :func:`create_kwargs`.

    Here we're in the inner loop of iterating over the parameters to a particular
    constructor, trying to construct just one of them.  The information we get for that
    parameter is its name, its type annotation, and its optionality.
    """
    optional, annotation = _extract_optional(annotation)

    if param_name not in parameters:
        if extras and param_name in extras:
            return extras[param_name]
        if param_default == _NO_DEFAULT:
            raise ConfigurationException(
                f"Parameter {param_name} found neither in config of {factory.__name__} "
                "nor in extras."
            )
        else:
            return param_default
    else:
        if extras and param_name in extras:
            raise ConfigurationException(
                f"Parameter {param_name} found in both extras and config."
            )
    return _construct_param_arg(
        factory, param_name, parameters[param_name], annotation, optional, context
    )


def _construct_param_arg(
    factory: Callable[..., _T],
    param_name: str,
    param_value: Any,
    annotation: Type,
    optional: bool,
    context: Any,
) -> Any:
    origin = getattr(annotation, "__origin__", None)
    args = getattr(annotation, "__args__", [])
    if hasattr(annotation, "from_params"):
        return annotation.from_params(parameters=param_value, context=context)

    # This is special logic for handling collections.
    elif origin in (dict, Dict):
        if len(args) != 2:
            raise ConfigurationException(
                f"Dict type arguments should be specified for {param_name} of "
                f"{factory.__name__}."
            )
        key_cls, value_cls = args
        if hasattr(value_cls, "from_params"):
            value_cls = annotation.__args__[-1]

            value_dict = {}
            for key, value_params in param_value.items():
                value_dict[key] = value_cls.from_params(
                    parameters=value_params, context=context
                )

            return value_dict
        else:
            return param_value

    elif origin in (list, List):
        if len(args) != 1:
            raise ConfigurationException(
                f"List type argument should be specified for {param_name} of "
                f"{factory.__name__}."
            )
        value_cls = args[0]
        if hasattr(value_cls, "from_params"):
            value_list = []

            for value_params in param_value:
                value_list.append(
                    value_cls.from_params(parameters=value_params, context=context)
                )

            return value_list
        else:
            return param_value

    elif origin in (tuple, Tuple):
        value_list = []

        for value_cls, value_params in zip(annotation.__args__, param_value):
            if hasattr(value_cls, "from_params"):
                value_list.append(
                    value_cls.from_params(parameters=value_params, context=context)
                )
            else:
                value_list.append(value_params)

        return tuple(value_list)

    elif origin in (set, Set):
        if len(args) != 1:
            raise ConfigurationException(
                f"Set type argument should be specified for {param_name} of "
                f"{factory.__name__}."
            )
        value_cls = args[0]
        if hasattr(value_cls, "from_params"):
            value_set = set()

            for value_params in param_value:
                value_set.add(value_cls.from_params(parameters=value_params))

            return value_set

    elif origin == Union:

        for arg in args:
            optional, arg = _extract_optional(arg)
            try:
                return _construct_param_arg(
                    factory, param_name, param_value, arg, optional, context
                )
            except (ValueError, TypeError, ConfigurationException, AttributeError):
                continue

        raise ConfigurationException(
            f"Failed to construct argument {param_name} with type {annotation}"
        )
    else:
        return param_value


def register_resources(context: Any, parameters: Dict[str, Any]) -> Any:
    """Register resources given their configuration."""
    from formatml.resources.resource import Resource
    from formatml.utils.registrable import by_name

    for name, subparameters in parameters.items():
        _logger.debug(f"Registering resource {name}.")
        subclass = by_name(Resource, subparameters["_type"])
        if hasattr(subclass, "from_params"):
            instance = subclass.from_params(subparameters["_config"], context)
        else:
            instance = subclass(**subparameters["_config"])
        context.register_resource(name, instance)

    return context


def from_params(clz: Type[_T]) -> Type[_T]:
    """Decorate a class with a from_params method."""

    def from_params(
        cls: Type[_T],
        parameters: Dict[str, Any],
        context: Any,
        extras: Optional[Dict[str, Any]] = None,
    ) -> _T:
        """
        Implement `from_params` automatically.

        If you need more complex logic in your from `from_params` method, you'll have to
        implement your own method that overrides this one.
        """
        # Import here to avoid circular imports.
        from formatml.utils.registrable import by_name, has_registrations

        if (
            "_resources_registration" in parameters
            and parameters["_resources_registration"]
        ):
            register_resources(context, parameters["_resources_registration"])

        if "_resource" in parameters:
            return context.get_resource(parameters["_resource"])

        if has_registrations(cls):
            subclass = by_name(cls, parameters["_type"])

            if hasattr(subclass, "from_params"):
                return subclass.from_params(
                    parameters=parameters["_config"], context=context, extras=extras
                )
            if extras:
                if _takes_kwargs(subclass):
                    return subclass(**parameters["_config"], **extras)
                return subclass(
                    **parameters["_config"],
                    **{
                        arg: value
                        for arg, value in extras.items()
                        if _takes_arg(subclass, arg)
                        and arg not in parameters["_config"]
                    },
                )
            return subclass(**parameters["_config"])
        else:
            # This is not a base class, so convert our params and extras into a dict of
            # kwargs.

            if cls.__init__ == object.__init__:
                # This class does not have an explicit constructor, so don't give it any
                # kwargs. Without this logic, create_kwargs will look at object.__init__
                # and see that it takes *args and **kwargs and look for those.
                kwargs: Dict[str, Any] = {}
            else:
                # This class has a constructor, so create kwargs for it.
                kwargs = _create_kwargs(cls, parameters, context, extras)
            return cls(**kwargs)  # type: ignore

    clz.from_params = classmethod(from_params)  # type: ignore
    return clz
