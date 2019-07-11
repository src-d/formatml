from argparse import ArgumentParser
from collections import OrderedDict
from typing import Any, Callable, Dict, NamedTuple


class StepDescription(NamedTuple):
    description: str
    parser_definer: Callable[[ArgumentParser], None]
    handler: Callable[..., Any]


pipelines: Dict[str, Dict[str, StepDescription]] = OrderedDict()


def register_step(
    pipeline_name: str, step_name: str, parser_definer: Callable[[ArgumentParser], None]
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def wrapper(handler: Callable[..., Any]) -> Callable[..., Any]:
        if pipeline_name not in pipelines:
            pipelines[pipeline_name] = OrderedDict()
        pipelines[pipeline_name][step_name] = StepDescription(
            description=handler.__doc__, parser_definer=parser_definer, handler=handler
        )
        return handler

    return wrapper
