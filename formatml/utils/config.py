from copy import deepcopy
from json import dump as json_dump, load as json_load
from pathlib import Path
from typing import Any, Dict, Iterable


class Config:
    def __init__(self, *, options: Dict[str, Any], data: Dict[str, Any], config: str):
        self.options = options
        self.data = data
        self.config = config

    def save(self, path: Path) -> None:
        dict_config = {}
        dict_config["data"] = self.data
        with path.open("w", encoding="utf8") as fh:
            json_dump(vars(self), fh)

    @staticmethod
    def from_arguments(
        arguments: Dict[str, Any], data_arguments: Iterable[str], config_argument: str
    ) -> "Config":
        options = deepcopy(arguments)
        data = {}
        for key in data_arguments:
            data[key] = options.pop(key)
        config = options.pop(config_argument)
        return Config(options=options, data=data, config=config)

    @staticmethod
    def from_json(json_path: Path) -> "Config":
        with json_path.open(mode="r", encoding="utf8") as fh:
            dict_config = json_load(fh)
        return Config(**dict_config)
