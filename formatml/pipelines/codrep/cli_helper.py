from argparse import ArgumentParser


class CLIHelper:
    def __init__(self, parser: ArgumentParser) -> None:
        self.parser = parser

    def add_configs_dir(self) -> None:
        self.parser.add_argument(
            "--configs-dir", required=True, help="Path to the configs."
        )

    def add_tensors_dir(self) -> None:
        self.parser.add_argument(
            "--tensors-dir", required=True, help="Path to the instance pickle."
        )

    def add_uasts_dir(self) -> None:
        self.parser.add_argument(
            "--uasts-dir", required=True, help="Path to the UASTs."
        )

    def add_instance_file(self) -> None:
        self.parser.add_argument(
            "--instance-file", required=True, help="Path to the pickled instance."
        )

    def add_log_level(self) -> None:
        self.parser.add_argument(
            "--log-level", default="DEBUG", help="Logging verbosity."
        )
