from argparse import ArgumentParser
from collections import OrderedDict
from logging import _nameToLevel as logging_name_to_level
from typing import Any, Callable, Dict, List, NamedTuple, Tuple


class Step(NamedTuple):
    name: str
    description: str
    parser_definer: Callable[[ArgumentParser], None]
    handler: Callable[..., Any]
    graceful_keyboard_interruption: bool


pipelines: Dict[str, List[Step]] = OrderedDict()


def register_step(
    pipeline_name: str,
    parser_definer: Callable[[ArgumentParser], None],
    graceful_keyboard_interruption: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def wrapper(handler: Callable[..., Any]) -> Callable[..., Any]:
        if pipeline_name not in pipelines:
            pipelines[pipeline_name] = []
        pipelines[pipeline_name].append(
            Step(
                name=handler.__name__,
                description=handler.__doc__,
                parser_definer=parser_definer,
                handler=handler,
                graceful_keyboard_interruption=graceful_keyboard_interruption,
            )
        )
        return handler

    return wrapper


def _define_parser() -> ArgumentParser:
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(help="Pipelines")
    for pipeline_name, pipeline_steps in pipelines.items():
        pipeline_parser = subparsers.add_parser(pipeline_name)
        pipeline_subparsers = pipeline_parser.add_subparsers(help="Steps")
        for step in pipeline_steps:
            step_parser = pipeline_subparsers.add_parser(
                step.name, help=step.description
            )
            step_parser.set_defaults(handler=step.handler)
            step_parser.set_defaults(
                graceful_keyboard_interruption=step.graceful_keyboard_interruption
            )
            step.parser_definer(step_parser)
            step_parser.add_argument(
                "--log-level",
                default="DEBUG",
                choices=logging_name_to_level,
                help="Logging verbosity.",
            )
    return parser


def parse_args() -> Tuple[Callable[..., Any], Dict[str, Any], bool]:
    parser = _define_parser()
    args = parser.parse_args()
    args.log_level = logging_name_to_level[args.log_level]
    handler = args.handler
    graceful_keyboard_interruption = args.graceful_keyboard_interruption
    delattr(args, "handler")
    delattr(args, "graceful_keyboard_interruption")
    return handler, vars(args), graceful_keyboard_interruption
