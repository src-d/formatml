from argparse import ArgumentParser
from typing import Any, Callable


commands = []


def register_command(
    name: str, description: str, parser_definer: Callable[[ArgumentParser], None]
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def wrapper(handler: Callable[..., Any]) -> Callable[..., Any]:
        commands.append((name, description, parser_definer, handler))
        return handler

    return wrapper
