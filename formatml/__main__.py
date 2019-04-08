from argparse import ArgumentParser
from json import loads as json_loads

from _jsonnet import evaluate_file as jsonnet_evaluate_file
from coloredlogs import install as coloredlogs_install

from formatml.commands.command import Command
from formatml.resources.resource import Context


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
    args = parser.parse_args()
    json_string = jsonnet_evaluate_file(args.config)
    config = json_loads(json_string, encoding="utf-8")
    coloredlogs_install(
        level=args.log_level, fmt="%(name)27s %(levelname)8s %(message)s"
    )
    context = Context()
    Command.from_params(config, context=context).run(context=context)  # type: ignore


if __name__ == "__main__":
    main()
