from bz2 import open as bz2_open
from collections import defaultdict
from enum import Enum, unique
from io import StringIO
from logging import DEBUG, getLogger, INFO
from pathlib import Path
from pickle import dump as pickle_dump
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from torch.nn.utils.rnn import PackedSequence
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import random_split
from torch.utils.tensorboard import SummaryWriter

from formatml.datasets.dataset import Dataset
from formatml.models.model import Model, ModelOutput
from formatml.modules.misc.scheduler import Scheduler
from formatml.utils.from_params import from_params
from formatml.utils.registrable import register_from_enum
from formatml.utils.torch_helpers import data_if_packed


@unique
@register_from_enum
class DataType(Enum):
    Train = "train"
    Eval = "eval"


_metrics = {}

Metric = Callable[[ModelOutput, PackedSequence], float]


def register_metric(name: str) -> Callable[[Metric], Metric]:
    def wrapper(metric: Metric) -> Metric:
        _metrics[name] = metric
        return metric

    return wrapper


@register_metric(name="cross_entropy")
def _cross_entropy(forward: ModelOutput, labels: PackedSequence) -> float:
    return data_if_packed(forward.loss).item()


@register_metric(name="perplexity")
def _perplexity(forward: ModelOutput, labels: PackedSequence) -> float:
    return 2 ** data_if_packed(forward.loss).item()


@register_metric(name="mrr")
def _accuracy(forward: ModelOutput, labels: PackedSequence) -> float:
    ground_truth = data_if_packed(labels).argmax(dim=0)
    predictions = data_if_packed(forward.output)[:, 1].argsort(descending=True)
    rank = (predictions == ground_truth).nonzero().item()
    return 1 / (rank + 1)


@register_metric(name="accuracy_max_decoding")
def _accuracy_max_decoding(forward: ModelOutput, labels: PackedSequence) -> float:
    return (
        data_if_packed(forward.output).argmax(dim=1) == data_if_packed(labels)
    ).sum().item() / data_if_packed(labels).nelement()


@from_params
class Trainer:

    _logger = getLogger(__name__)

    def __init__(
        self,
        dataset: Dataset,
        model: Model,
        optimizer: Optimizer,
        scheduler: Scheduler,
        epochs: int,
        batch_size: int,
        run_dir: Path,
        eval_every: int,
        train_eval_split: float,
        metric_names: List[str],
    ) -> None:
        self.dataset = dataset
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.batch_size = batch_size
        self.run_dir = run_dir
        self.eval_every = eval_every
        self.train_eval_split = train_eval_split
        self._checkpoints_dir = self.run_dir / "checkpoints"
        self._writers: Dict[DataType, SummaryWriter] = {}
        self._accumulated_metrics: Dict[
            DataType, Dict[Metric, List[float]]
        ] = defaultdict(lambda: defaultdict(lambda: []))
        self._metrics = [_metrics[metric_name] for metric_name in metric_names]
        self._metric_names = {
            metric: metric_name
            for metric, metric_name in zip(self._metrics, metric_names)
        }
        self._epochs_size = len(str(epochs))

    def train(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoints_dir.mkdir(parents=True, exist_ok=True)
        for data_type in DataType:
            self._writers[data_type] = SummaryWriter(
                str(self.run_dir / data_type.value)
            )
        self._global_step = 0
        self.epochs_size = len(str(self.epochs))
        self.iterations_size = len(str(1000))
        train_size = round(len(self.dataset) * self.train_eval_split)
        eval_size = len(self.dataset) - train_size
        train_dataset, eval_dataset = random_split(
            self.dataset, [train_size, eval_size]
        )
        self._dataloaders = {
            DataType.Train: DataLoader(
                train_dataset,
                shuffle=True,
                collate_fn=self.dataset.collate,
                batch_size=self.batch_size,
                num_workers=1,
            ),
            DataType.Eval: DataLoader(
                eval_dataset,
                shuffle=True,
                collate_fn=self.dataset.collate,
                batch_size=self.batch_size,
                num_workers=1,
            ),
        }
        self._iterations_size = len(
            str(max(len(dataloader) for dataloader in self._dataloaders.values()))
        )
        for epoch in range(1, self.epochs + 1):
            self._train_epoch(epoch)
            self.scheduler.step(epoch=None)

    def _train_epoch(self, epoch: int) -> None:
        self.model.train()
        for iteration, sample in enumerate(self._dataloaders[DataType.Train], start=1):
            forward = self.model.forward(sample)
            self._compute_metrics(
                forward=forward,
                labels=sample["label"].labels,
                data_type=DataType.Train,
                accumulate=False,
                send_event=True,
                epoch=epoch,
                iteration=iteration,
            )
            self.optimizer.zero_grad()
            forward.loss.backward()
            self.optimizer.step()
            if self.eval_every > 0 and (self._global_step + 1) % self.eval_every == 0:
                self._eval_epoch(epoch)
                self._save_checkpoint(epoch, iteration)
            self._global_step += 1

    def _eval_epoch(self, epoch: int) -> None:
        self.model.eval()
        for iteration, sample in enumerate(self._dataloaders[DataType.Eval], start=1):
            forward = self.model.forward(sample)
            self._compute_metrics(
                forward=forward,
                labels=sample["label"].labels,
                data_type=DataType.Eval,
                accumulate=True,
                send_event=False,
                epoch=epoch,
                iteration=iteration,
            )

        self._log_accumulated_metrics(
            data_type=DataType.Eval, send_event=True, epoch=epoch, iteration=iteration
        )

    def _save_checkpoint(self, epoch: int, iteration: Optional[int] = None) -> None:
        checkpoint_name = f"e{epoch}"
        if iteration is not None:
            checkpoint_name += f"-i{iteration}"
        checkpoint_name += ".pickle.bz2"
        with bz2_open(self._checkpoints_dir / checkpoint_name, "wb") as fh:
            pickle_dump(
                dict(
                    model_state_dict=self.model.state_dict,
                    optimizer_state_dict=self.optimizer.state_dict,
                    scheduler_state_dict=self.scheduler.state_dict,
                    epoch=epoch,
                    iteration=iteration,
                ),
                fh,
            )

    def _compute_metrics(
        self,
        forward: ModelOutput,
        labels: PackedSequence,
        data_type: DataType,
        accumulate: bool,
        send_event: bool,
        epoch: int,
        iteration: int,
    ) -> None:
        self._log_values(
            values=((metric, metric(forward, labels)) for metric in self._metrics),
            data_type=data_type,
            send_event=send_event,
            accumulate=accumulate,
            epoch=epoch,
            iteration=iteration,
            logging_level=DEBUG,
        )

    def _log_accumulated_metrics(
        self,
        data_type: DataType,
        send_event: bool,
        epoch: int,
        iteration: int,
        dont_reset_accumulated: bool = False,
    ) -> None:
        values = []
        for metric in self._metrics:
            average = self._average(self._accumulated_metrics[data_type][metric])
            if average is not None:
                values.append((metric, average))
        self._log_values(
            values=values,
            data_type=data_type,
            send_event=send_event,
            accumulate=False,
            epoch=epoch,
            iteration=iteration,
            logging_level=INFO,
        )
        self._reset_accumulated(data_type)

    def _log_values(
        self,
        *,
        values: Iterable[Tuple[Metric, float]],
        data_type: DataType,
        send_event: bool,
        accumulate: bool,
        epoch: int,
        iteration: int,
        logging_level: int,
    ) -> None:
        with StringIO() as buffer:
            buffer.write(
                f"{data_type.value} "
                f"{epoch:{self._epochs_size}d}/{self.epochs:{self._epochs_size}d} "
                f"{iteration:{self._iterations_size}d}"
                f"/{len(self._dataloaders[data_type]):{self._iterations_size}d}"
            )
            for metric, value in values:
                name = self._metric_names[metric]
                buffer.write(f" {name} {value:.4f}")
                if accumulate:
                    self._accumulated_metrics[data_type][metric].append(value)
                if send_event:
                    self._writers[data_type].add_scalar(name, value, self._global_step)
            self._logger.log(logging_level, buffer.getvalue())

    @staticmethod
    def _average(values: List[float]) -> Optional[float]:
        return (sum(values) / len(values)) if values else None

    def _reset_accumulated(self, data_type: DataType) -> None:
        del self._accumulated_metrics[data_type]

    def __del__(self) -> None:
        for writer in self._writers.values():
            if writer:
                writer.close()
