from argparse import ArgumentParser
from bz2 import open as bz2_open
from enum import Enum
from pathlib import Path
from pickle import load as pickle_load
from typing import List, Optional

from torch.nn import Linear, LSTM, Module, Sequential
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR
from torch.optim.optimizer import Optimizer as TorchOptimizer

from formatml.data.instance import Instance
from formatml.datasets.codrep_dataset import CodRepDataset
from formatml.models.codrep_model import CodRepModel
from formatml.modules.graph_encoders.ggnn import GGNN
from formatml.modules.misc.graph_embedding import GraphEmbedding
from formatml.modules.misc.item_getter import ItemGetter
from formatml.modules.misc.squeezer import Squeezer
from formatml.modules.misc.unsqueezer import Unsqueezer
from formatml.pipelines.codrep.cli_builder import CLIBuilder
from formatml.pipelines.pipeline import register_step
from formatml.training.trainer import Trainer
from formatml.utils.config import Config
from formatml.utils.helpers import setup_logging


class DecoderType(Enum):
    FF = "ff"
    RNN = "rnn"


class Optimizer(Enum):
    Adam = "adam"
    SGD = "sgd"


def add_arguments_to_parser(parser: ArgumentParser) -> None:
    cli_builder = CLIBuilder(parser)
    cli_builder.add_instance_file()
    cli_builder.add_tensors_dir()
    parser.add_argument(
        "--train-dir",
        required=True,
        help="Directory where the run artifacts will be output.",
    )
    cli_builder.add_configs_dir()
    parser.add_argument(
        "--model-encoder-iterations",
        help="Number of message passing iterations to apply (defaults to %(default)s).",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--model-encoder-output-dim",
        help="Dimensionality of the encoder output (defaults to %(default)s).",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--model-encoder-message-dim",
        help="Dimensionality of the encoder messages (defaults to %(default)s).",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--model-decoder-type",
        help="Type of decoder to use (defaults to %(default)s).",
        choices=[t.value for t in DecoderType],
        default=DecoderType.FF.value,
    )
    parser.add_argument(
        "--optimizer-type",
        help="Optimizer to use (defaults to %(default)s).",
        default="adam",
        choices=[o.value for o in Optimizer],
    )
    parser.add_argument(
        "--optimizer-learning-rate",
        help="Learning rate of the optimizer (defaults to %(default)s).",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--scheduler-step-size",
        help="Number of epochs before the scheduler reduces the learning rate "
        "(defaults to %(default)s).",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--scheduler-gamma",
        help="Factor of the learning rate reduction (defaults to %(default)s).",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--trainer-epochs",
        help="Number of epochs to train for (defaults to %(default)s).",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--trainer-batch-size",
        help="Size of the training batches (defaults to %(default)s).",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--trainer-eval-every",
        help="Number of iterations before an evaluation epoch (defaults to "
        "%(default)s).",
        type=int,
        default=360,
    )
    parser.add_argument(
        "--trainer-limit-epochs-at",
        help="Number of batches to limit an epoch to.",
        type=int,
    )
    parser.add_argument(
        "--trainer-train-eval-split",
        help="Proportion kept for training (defaults to %(default)s). "
        "Rest goes to evaluation.",
        type=float,
        default=0.90,
    )
    parser.add_argument(
        "--trainer-metric-names",
        help="Names of the metrics to use (defaults to %(default)s).",
        nargs="+",
        default=["mrr", "cross_entropy"],
    )
    parser.add_argument(
        "--trainer-selection-metric",
        help="Name of the metric to use for checkpoint selection "
        "(defaults to %(default)s).",
        default="mrr",
    )
    parser.add_argument(
        "--trainer-kept-checkpoints",
        help="Number of best checkpoints to keep (defaults to %(default)s).",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--trainer-cuda", help="CUDA index of the device to use for training.", type=int
    )


@register_step(
    pipeline_name="codrep",
    parser_definer=add_arguments_to_parser,
    graceful_keyboard_interruption=True,
)
def train(
    *,
    instance_file: str,
    tensors_dir: str,
    train_dir: str,
    configs_dir: str,
    model_encoder_iterations: int,
    model_encoder_output_dim: int,
    model_encoder_message_dim: int,
    model_decoder_type: str,
    optimizer_type: str,
    optimizer_learning_rate: float,
    scheduler_step_size: int,
    scheduler_gamma: float,
    trainer_epochs: int,
    trainer_batch_size: int,
    trainer_eval_every: int,
    trainer_limit_epochs_at: Optional[int],
    trainer_train_eval_split: float,
    trainer_metric_names: List[str],
    trainer_selection_metric: str,
    trainer_kept_checkpoints: int,
    trainer_cuda: Optional[int],
    log_level: str,
) -> None:
    """Run the training."""
    Config.from_arguments(
        locals(), ["instance_file", "tensors_dir", "train_dir"], "configs_dir"
    ).save(Path(configs_dir) / "train.json")
    logger = setup_logging(__name__, log_level)

    tensors_dir_path = Path(tensors_dir).expanduser().resolve()
    train_dir_path = Path(train_dir).expanduser().resolve()
    train_dir_path.mkdir(parents=True, exist_ok=True)

    with bz2_open(instance_file, "rb") as fh:
        instance = pickle_load(fh)

    dataset = CodRepDataset(input_dir=tensors_dir_path)
    logger.info("Dataset of size %d", len(dataset))

    model = build_model(
        instance=instance,
        model_encoder_iterations=model_encoder_iterations,
        model_encoder_output_dim=model_encoder_output_dim,
        model_encoder_message_dim=model_encoder_message_dim,
        model_decoder_type=model_decoder_type,
    )
    # The model needs a forward to be completely initialized.
    model(instance.collate([dataset[0]]))
    logger.info("Configured model %s", model)

    if Optimizer(optimizer_type) is Optimizer.Adam:
        optimizer: TorchOptimizer = Adam(
            params=model.parameters(), lr=optimizer_learning_rate
        )
    else:
        optimizer = SGD(params=model.parameters(), lr=optimizer_learning_rate)
    scheduler = StepLR(
        optimizer=optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma
    )
    trainer = Trainer(
        instance=instance,
        epochs=trainer_epochs,
        batch_size=trainer_batch_size,
        eval_every=trainer_eval_every,
        train_eval_split=trainer_train_eval_split,
        limit_epochs_at=trainer_limit_epochs_at,
        metric_names=trainer_metric_names,
        selection_metric=trainer_selection_metric,
        kept_checkpoints=trainer_kept_checkpoints,
        run_dir_path=train_dir_path,
        dataset=dataset,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        cuda_device=trainer_cuda,
    )
    trainer.train()


def build_model(
    instance: Instance,
    model_encoder_iterations: int,
    model_encoder_output_dim: int,
    model_encoder_message_dim: int,
    model_decoder_type: str,
) -> CodRepModel:
    graph_field = instance.get_field_by_type("graph")
    label_field = instance.get_field_by_type("label")
    indexes_field = instance.get_field_by_type("indexes")
    graph_input_fields = instance.get_fields_by_type("input")
    graph_input_dimensions = [48, 48, 32]
    feature_names = [field.name for field in graph_input_fields]
    if DecoderType(model_decoder_type) is DecoderType.FF:
        class_projection: Module = Linear(
            in_features=model_encoder_output_dim, out_features=2
        )
    else:
        class_projection = Sequential(
            Unsqueezer(0),
            LSTM(
                input_size=model_encoder_output_dim,
                hidden_size=model_encoder_output_dim // 2,
                batch_first=True,
                bidirectional=True,
            ),
            ItemGetter(0),
            Squeezer(),
            Linear(in_features=model_encoder_output_dim, out_features=2),
        )
    return CodRepModel(
        graph_embedder=GraphEmbedding(
            graph_input_dimensions,
            [field.vocabulary for field in graph_input_fields],  # type: ignore
        ),
        graph_encoder=GGNN(
            in_feats=sum(graph_input_dimensions),
            out_feats=model_encoder_output_dim,
            n_steps=model_encoder_iterations,
            n_etypes=len(graph_field.vocabulary),  # type: ignore
        ),
        class_projection=class_projection,
        graph_field_name=graph_field.name,
        feature_field_names=feature_names,
        indexes_field_name=indexes_field.name,
        label_field_name=label_field.name,
    )
