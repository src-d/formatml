from argparse import ArgumentParser
from json import loads as json_loads
from pathlib import Path

from _jsonnet import evaluate_file as jsonnet_evaluate_file
from coloredlogs import install as coloredlogs_install

from formatml.commands.command import Command
from formatml.resources.resource import Context
from formatml.utils.helpers import import_submodules


def main() -> None:
    """
    CLI entry point of the module.

    Delegates to `work` after having set up logging and parsed CLI args.
    """
    parser = ArgumentParser()
    parser.add_argument(
        "config", help="Location of the configuration of the command to run."
    )
    parser.add_argument("--log-level", default="DEBUG", help="Logging verbosity.")
    parser.add_argument(
        "-i",
        "--import-package",
        help="Package to load recursively (to register classes).",
    )
    args = parser.parse_args()
    json_string = jsonnet_evaluate_file(args.config)
    config = json_loads(json_string, encoding="utf-8")
    coloredlogs_install(
        level=args.log_level, fmt="%(name)27s %(levelname)8s %(message)s"
    )
    if args.import_package:
        import_submodules(args.import_package)
    context = Context(Path(config.get("context_cache_dir")))
    Command.from_params(config["command"], context=context).run(  # type: ignore
        context=context
    )


if __name__ == "__main__":
    main()
