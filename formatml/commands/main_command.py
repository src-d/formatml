from copy import deepcopy
from json import dump as json_dump
from logging import getLogger
from pathlib import Path
from typing import Any, Dict

from torch.optim.optimizer import Optimizer

from formatml.commands.command import Command
from formatml.datasets.dataset import Dataset
from formatml.models.model import Model
from formatml.modules.misc.scheduler import Scheduler
from formatml.resources.resource import Context
from formatml.trainer import Trainer
from formatml.utils.helpers import get_sha_and_dirtiness
from formatml.utils.registrable import register


@register(cls=Command, name="main")
class MainCommand(Command):
    """Dev command."""

    _logger = getLogger(__name__)

    def __init__(self, config: Dict[str, Any], run_dir: Path):
        super().__init__(config)
        self.run_dir = run_dir

    def run(self, context: Context) -> None:
        """Run the training."""
        git_info = get_sha_and_dirtiness()
        config_copy = deepcopy(self.config)
        if git_info is None:
            config_copy["git_info"] = None
        else:
            sha, dirty = git_info
            config_copy["git_info"] = {"sha": sha, "dirty": dirty}
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with (self.run_dir / "config.json").open(mode="w", encoding="utf8") as fh:
            json_dump(config_copy, fh, indent=2, sort_keys=True)

        dataset = Dataset.from_params(self.config["dataset"], context)  # type: ignore
        dataset.download()
        dataset.pre_tensorize()
        dataset.tensorize()
        context.save_resources()
        self._logger.info(f"Dataset of size {len(dataset)}")
        model = Model.from_params(self.config["model"], context)  # type: ignore
        model(dataset[0])
        self._logger.info(f"Configured model {model}")
        optimizer_config = deepcopy(self.config["optimizer"])
        optimizer_config["_config"]["params"] = model.parameters()
        optimizer = Optimizer.from_params(optimizer_config, context)  # type: ignore
        scheduler = Scheduler.from_params(  # type: ignore
            self.config["scheduler"], context, extras=dict(optimizer=optimizer)
        )
        trainer = Trainer.from_params(  # type: ignore
            self.config["trainer"],
            context,
            extras=dict(
                dataset=dataset, model=model, optimizer=optimizer, scheduler=scheduler
            ),
        )
        trainer.train()
