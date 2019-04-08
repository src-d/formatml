from datetime import datetime
from itertools import chain
from pathlib import Path
from sys import exit, stderr
from typing import Iterable, Tuple, Union

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
