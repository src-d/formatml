from datetime import datetime
from itertools import chain
from logging import Filter, getLogger, Logger, LogRecord
from pathlib import Path
from sys import exit, stderr
from typing import Any, Iterable, List, Optional, Tuple, Type, TypeVar, Union

from coloredlogs import install as coloredlogs_install
from dulwich.errors import NotGitRepository
from dulwich.porcelain import status
from dulwich.repo import Repo


class ShortNameFilter(Filter):
    def filter(self, record: LogRecord) -> int:
        record.shortname = record.name.split(".")[-1]  # type: ignore
        return 1


def setup_logging(name: str, log_level: str) -> Logger:
    coloredlogs_install(
        level=log_level,
        fmt="%(asctime)s %(shortname)10s %(message)s",
        datefmt="%H:%M:%S",
    )
    getLogger().handlers[0].addFilter(ShortNameFilter())
    return getLogger(name)


def get_sha_and_dirtiness(prompt_on_dirty: bool = True) -> Optional[Tuple[str, bool]]:
    try:
        git_status = status()
    except NotGitRepository:
        return None
    dirty = False

    def to_str(string: Union[str, bytes]) -> str:
        if isinstance(string, str):
            return string
        return string.decode("utf8")

    def print_files(filenames: Iterable[Union[str, bytes]]) -> None:
        print("\n".join(f"  - {to_str(filename)}" for filename in filenames))

    if git_status.untracked:
        print("Those files are untracked:", file=stderr)
        print_files(git_status.untracked)
        dirty = True
    if git_status.unstaged:
        print("Those files are unstaged:", file=stderr)
        print_files(git_status.unstaged)
        dirty = True
    if any(git_status.staged.values()):
        print("Those files are uncommited:", file=stderr)
        print_files(chain(*git_status.staged.values()))
        dirty = True
    if dirty:
        print("Are you sure you want to continue [y/n]? ", end="")
        answer = input()
        if answer != "y":
            exit(1)

    repo = Repo(".")
    sha = to_str(repo.head())
    return sha, dirty


def date_template_to_path(date_template: str) -> Path:
    return Path(datetime.now().strftime(date_template))


def get_generic_arguments(cls: Type[Any], subclass: Type[Any]) -> Tuple[Any, ...]:
    if not issubclass(subclass, cls):
        raise RuntimeError(
            f"Cannot find the type arguments of {cls.__name__}: {subclass.__name__} is "
            "not a subtype."
        )
    current_subclass = subclass
    current_args: List[Type[Any]] = []
    while current_subclass != cls:
        if not hasattr(current_subclass, "__orig_bases__"):
            raise RuntimeError(
                f"Cannot find __orig_bases__ of {subclass.__name__}:"
                f"is {cls.__name__} Generic?"
            )
        bases = current_subclass.__orig_bases__  # type: ignore
        relevant_parent = None
        relevant_parent_arguments = None
        for base in bases:
            if not issubclass(base.__origin__, cls):
                for arg in base.__args__:
                    if istypevar(arg):
                        current_args.pop()
            else:
                if relevant_parent is not None:
                    raise RuntimeError(
                        "Cannot handle more than 1 parent subclassing the target type."
                    )
                relevant_parent = base.__origin__
                relevant_parent_arguments = [
                    current_args.pop() if istypevar(arg) else arg
                    for arg in base.__args__
                ]
        if relevant_parent is None:
            raise RuntimeError(
                "Could not trace back {current_subclass.__name__} to {cls.__name__}."
            )
        current_subclass = relevant_parent
        current_args = relevant_parent_arguments
    return tuple(current_args)


def istypevar(obj: Any) -> bool:
    return isinstance(obj, TypeVar)  # type: ignore
