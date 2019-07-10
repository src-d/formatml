from argparse import ArgumentParser

from formatml.commands.command import commands


def main() -> None:
    """CLI entry point of the formatml."""
    parser = ArgumentParser()
    subparsers = parser.add_subparsers()
    for name, description, parser_definer, handler in commands:
        subparser = subparsers.add_parser(name, help=description)
        parser_definer(subparser),
        subparser.set_defaults(handler=handler)
    args = parser.parse_args()
    handler = args.handler
    delattr(args, "handler")
    handler(**vars(args))


if __name__ == "__main__":
    main()
