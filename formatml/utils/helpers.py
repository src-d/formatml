from datetime import datetime
from importlib import import_module, invalidate_caches as importlib_invalidate_caches
from itertools import chain
from pathlib import Path
from pkgutil import walk_packages
from sys import exit, path as sys_path, stderr
from typing import Any, Iterable, List, Tuple, Type, TypeVar, Union

from dulwich.porcelain import status
from dulwich.repo import Repo


def get_sha_and_dirtiness(prompt_on_dirty: bool = True) -> Tuple[str, bool]:
    git_status = status()
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
        print_files(chain(git_status.staged))
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


def import_submodules(package_name: str, append_dot_path: bool = False) -> None:
    """
    Mostly from https://github.com/allenai/allennlp/blob/master/allennlp/common/util.py
    Import all submodules under the given package.
    Primarily useful so that people using AllenNLP as a library
    can specify their own custom packages and have their custom
    classes get loaded and registered.
    """
    importlib_invalidate_caches()

    if append_dot_path:
        sys_path.append(".")

    module = import_module(package_name)
    path = getattr(module, "__path__", [])
    path_string = "" if not path else path[0]

    for module_finder, name, _ in walk_packages(path):
        # Sometimes when you import third-party libraries that are on your path,
        # `pkgutil.walk_packages` returns those too, so we need to skip them.
        if path_string and module_finder.path != path_string:
            continue
        subpackage = f"{package_name}.{name}"
        import_submodules(subpackage)
