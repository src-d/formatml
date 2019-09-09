from bisect import bisect_right
from collections import defaultdict
from enum import Enum, unique
from heapq import heappush, heappushpop
from io import StringIO
from logging import DEBUG, getLogger, INFO
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from dgl import unbatch
from torch import device as torch_device, no_grad, save as torch_save
from torch.cuda import is_available as cuda_is_available
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import random_split
from torch.utils.tensorboard import SummaryWriter

from formatml.data.instance import Instance
from formatml.datasets.dataset import Dataset
from formatml.models.model import Model
from formatml.utils.torch_helpers import data_if_packed, log_grad_flow


@unique
class DataType(Enum):
    Train = "train"
    Eval = "eval"


_metrics = {}

Metric = Callable[[Dict[str, Any]], float]


def register_metric(name: str) -> Callable[[Metric], Metric]:
    def wrapper(metric: Metric) -> Metric:
        _metrics[name] = metric
        return metric

    return wrapper


@register_metric(name="cross_entropy")
def _cross_entropy(sample: Dict[str, Any]) -> float:
    return data_if_packed(sample["loss"]).item()


@register_metric(name="perplexity")
def _perplexity(sample: Dict[str, Any]) -> float:
    return 2 ** data_if_packed(sample["loss"]).item()


@register_metric(name="mrr")
def _mrr(sample: Dict[str, Any]) -> float:
    labels = sample["label"]
    ground_truth = data_if_packed(labels).argmax(dim=0)
    batched_graph = sample["typed_dgl_graph"].graph
    graphs = unbatch(batched_graph)
    start = 0
    total_number_of_nodes = 0
    bounds = []
    numpy_indexes = sample["indexes"].indexes.cpu().numpy()
    for graph in graphs:
        total_number_of_nodes += graph.number_of_nodes()
        end = bisect_right(numpy_indexes, total_number_of_nodes - 1)
        bounds.append((start, end))
        start = end
    ranks = []
    for start, end in bounds:
        predictions = data_if_packed(sample["forward"])[start:end, 1].argsort(
            descending=True
        )
        ground_truth = data_if_packed(labels)[start:end].argmax(dim=0)
        ranks.append((predictions == ground_truth).nonzero().item())
    return sum(1 / (rank + 1) for rank in ranks) / len(ranks)


@register_metric(name="accuracy_max_decoding")
def _accuracy_max_decoding(sample: Dict[str, Any]) -> float:
    label = sample["label"].labels
    return (
        data_if_packed(sample["forward"]).argmax(dim=1) == data_if_packed(label)
    ).sum().item() / data_if_packed(label).nelement()


class Trainer:

    _logger = getLogger(__name__)

    def __init__(
        self,
        dataset: Dataset,
        instance: Instance,
        model: Model,
        optimizer: Optimizer,
        scheduler: Any,
        epochs: int,
        batch_size: int,
        run_dir_path: Path,
        eval_every: int,
        limit_epochs_at: Optional[int],
        train_eval_split: float,
        metric_names: List[str],
        selection_metric: str,
        kept_checkpoints: int,
        cuda_device: Optional[int],
    ) -> None:
        self.dataset = dataset
        self.instance = instance
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.batch_size = batch_size
        self.run_dir_path = run_dir_path
        self.eval_every = eval_every
        self.train_eval_split = train_eval_split
        self.limit_epochs_at = 10e1000 if limit_epochs_at is None else limit_epochs_at
        self.selection_metric = selection_metric
        self.kept_checkpoints = kept_checkpoints
        self._checkpoints_stats: List[Tuple[float, str]] = []
        if cuda_device is not None:
            if not cuda_is_available():
                raise RuntimeError("CUDA is not available on this system.")
            self.use_cuda = True
            self.device = torch_device("cuda:%d" % cuda_device)
        else:
            self.use_cuda = False
            self.device = torch_device("cpu")
        self.model = model.to(self.device)
        self._checkpoints_dir = self.run_dir_path / "checkpoints"
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
        self.run_dir_path.mkdir(parents=True, exist_ok=True)
        self._checkpoints_dir.mkdir(parents=True, exist_ok=True)
        for data_type in DataType:
            self._writers[data_type] = SummaryWriter(
                str(self.run_dir_path / data_type.value)
            )
        self._global_step = 0
        train_size = round(len(self.dataset) * self.train_eval_split)
        eval_size = len(self.dataset) - train_size
        train_dataset, eval_dataset = random_split(
            self.dataset, [train_size, eval_size]
        )
        self._dataloaders = {
            DataType.Train: DataLoader(
                train_dataset,
                shuffle=True,
                collate_fn=self.instance.collate,
                batch_size=self.batch_size,
                pin_memory=self.use_cuda,
                num_workers=1,
            ),
            DataType.Eval: DataLoader(
                eval_dataset,
                shuffle=True,
                collate_fn=self.instance.collate,
                batch_size=self.batch_size,
                pin_memory=self.use_cuda,
                num_workers=1,
            ),
        }
        self._iterations_size = len(
            str(
                max(
                    min(self.limit_epochs_at, len(dataloader))
                    for dataloader in self._dataloaders.values()
                )
            )
        )
        for epoch in range(1, self.epochs + 1):
            self._train_epoch(epoch)
            self.scheduler.step(epoch=None)

    def _train_epoch(self, epoch: int) -> None:
        for iteration, sample in enumerate(self._dataloaders[DataType.Train], start=1):
            self.model.train()
            if self.limit_epochs_at is not None and iteration > self.limit_epochs_at:
                break
            sample = self.instance.to(sample, self.device)
            sample = self.model(sample)
            self._compute_metrics(
                sample=sample,
                data_type=DataType.Train,
                accumulate=False,
                send_event=True,
                epoch=epoch,
                iteration=iteration,
            )
            self.optimizer.zero_grad()
            sample["loss"].backward()
            log_grad_flow(
                self.model.named_parameters(),
                self._writers[DataType.Train],
                self._global_step,
            )
            self.optimizer.step()
            if self.eval_every > 0 and (self._global_step + 1) % self.eval_every == 0:
                score = self._eval_epoch(epoch)
                self._save_checkpoint(epoch, score, iteration)
            self._global_step += 1

    def _eval_epoch(self, epoch: int) -> float:
        self.model.eval()
        with no_grad():
            for iteration, sample in enumerate(
                self._dataloaders[DataType.Eval], start=1
            ):
                sample = self.instance.to(sample, self.device)
                sample = self.model.forward(sample)
                self._compute_metrics(
                    sample=sample,
                    data_type=DataType.Eval,
                    accumulate=True,
                    send_event=False,
                    epoch=epoch,
                    iteration=iteration,
                )

        results = self._log_accumulated_metrics(
            data_type=DataType.Eval, send_event=True, epoch=epoch, iteration=iteration
        )
        return results[self.selection_metric]

    def _save_checkpoint(
        self, epoch: int, score: float, iteration: Optional[int] = None
    ) -> None:
        checkpoint_name = "s%.4f-e%d" % (score, epoch)
        if iteration is not None:
            checkpoint_name += "-i%d" % iteration
        checkpoint_name += ".tar"
        checkpoint_path = str(self._checkpoints_dir / checkpoint_name)

        def save() -> None:
            torch_save(
                dict(
                    model_state_dict=self.model.state_dict(),
                    optimizer_state_dict=self.optimizer.state_dict(),
                    scheduler_state_dict=self.scheduler.state_dict(),
                    epoch=epoch,
                    iteration=iteration,
                ),
                checkpoint_path,
            )

        if len(self._checkpoints_stats) < self.kept_checkpoints:
            self._logger.info(
                "Saving checkpoint %s (performance %.4f).", checkpoint_path, score
            )
            heappush(self._checkpoints_stats, (score, checkpoint_path))
            save()
            return
        if score > self._checkpoints_stats[0][0]:
            to_remove = heappushpop(self._checkpoints_stats, (score, checkpoint_path))
            self._logger.info(
                "Saving checkpoint %s (performance %.4f).", checkpoint_path, score
            )
            save()
            self._logger.info(
                "Deleting checkpoint %s (performance %.4f).", to_remove[1], to_remove[0]
            )
            Path(to_remove[1]).unlink()

    def _compute_metrics(
        self,
        sample: Dict[str, Any],
        data_type: DataType,
        accumulate: bool,
        send_event: bool,
        epoch: int,
        iteration: int,
    ) -> None:
        self._log_values(
            values=((metric, metric(sample)) for metric in self._metrics),
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
    ) -> Dict[str, float]:
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
        return {self._metric_names[m]: v for m, v in values}

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
                "%s %*d/%*d %*d/%*d"
                % (
                    data_type.value,
                    self._epochs_size,
                    epoch,
                    self._epochs_size,
                    self.epochs,
                    self._iterations_size,
                    iteration,
                    self._iterations_size,
                    min(self.limit_epochs_at, len(self._dataloaders[data_type])),
                )
            )
            for metric, value in values:
                name = self._metric_names[metric]
                buffer.write(" %s %.4f" % (name, value))
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
        if hasattr(self, "_writers"):
            for writer in self._writers.values():
                if writer:
                    writer.close()
