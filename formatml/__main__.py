from argparse import ArgumentParser

from formatml.pipelines.pipeline import pipelines


def main() -> None:
    """CLI entry point of the formatml."""
    parser = ArgumentParser()
    subparsers = parser.add_subparsers()
    for pipeline_name, pipeline_steps in pipelines.items():
        pipeline_parser = subparsers.add_parser(pipeline_name)
        pipeline_subparsers = pipeline_parser.add_subparsers()
        for (
            step_name,
            (step_description, step_parser_definer, step_handler),
        ) in pipeline_steps.items():
            step_parser = pipeline_subparsers.add_parser(
                step_name, help=step_description
            )
            step_parser.set_defaults(handler=step_handler)
            step_parser_definer(step_parser)
    args = parser.parse_args()
    handler = args.handler
    delattr(args, "handler")
    handler(**vars(args))


if __name__ == "__main__":
    main()
