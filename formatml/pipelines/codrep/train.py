from argparse import ArgumentParser
from bz2 import open as bz2_open
from logging import getLogger
from pathlib import Path
from pickle import load as pickle_load
from typing import List, Optional

from torch.nn import Linear
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from formatml.datasets.codrep_dataset import CodRepDataset
from formatml.models.gnn_ff import GNNFFModel
from formatml.modules.graph_encoders.ggnn import GGNN
from formatml.modules.misc.graph_embedding import GraphEmbedding
from formatml.pipelines.pipeline import register_step
from formatml.training.trainer import Trainer
from formatml.utils.helpers import setup_logging


def add_arguments_to_parser(parser: ArgumentParser) -> None:
    parser.add_argument(
        "instance_file", metavar="instance-file", help="Path to the pickled instance."
    )
    parser.add_argument(
        "tensors_dir", metavar="tensors-dir", help="Path the pickled tensors."
    )
    parser.add_argument(
        "output_dir",
        metavar="output-dir",
        help="Directory where the run artifacts will be output.",
    )
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
        default=3,
    )
    parser.add_argument(
        "--trainer-batch-size",
        help="Size of the training batches (defaults to %(default)s).",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--trainer-eval-every",
        help="Number of iterations before an evaluation epoch (defaults to "
        "%(default)s).",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--trainer-train-eval-split",
        help="Proportion kept for training (defaults to %(default)s). "
        "Rest goes to evaluation.",
        type=float,
        default=0.80,
    )
    parser.add_argument(
        "--trainer-metric-names",
        help="Names of the metrics to use (defaults to %(default)s).",
        nargs="+",
        default=["mrr", "cross_entropy"],
    )
    parser.add_argument(
        "--trainer-cuda", help="CUDA index of the device to use for training.", type=int
    )
    parser.add_argument("--log-level", default="DEBUG", help="Logging verbosity.")


@register_step(
    pipeline_name="codrep", step_name="train", parser_definer=add_arguments_to_parser
)
def train(
    *,
    instance_file: str,
    tensors_dir: str,
    output_dir: str,
    model_encoder_iterations: int,
    model_encoder_output_dim: int,
    model_encoder_message_dim: int,
    optimizer_learning_rate: float,
    scheduler_step_size: int,
    scheduler_gamma: float,
    trainer_epochs: int,
    trainer_batch_size: int,
    trainer_eval_every: int,
    trainer_train_eval_split: float,
    trainer_metric_names: List[str],
    trainer_cuda: Optional[int],
    log_level: str,
) -> None:
    """Run the training."""
    setup_logging(log_level)
    logger = getLogger(__name__)

    tensors_dir_path = Path(tensors_dir).expanduser().resolve()
    output_dir_path = Path(output_dir).expanduser().resolve()

    with bz2_open(instance_file, "rb") as fh:
        instance = pickle_load(fh)

    graph_field = instance.fields[0]
    label_field = instance.fields[1]
    graph_input_fields = instance.fields[2:]
    graph_input_dimensions = [48, 48, 32]

    dataset = CodRepDataset(input_dir=tensors_dir_path)
    logger.info(f"Dataset of size {len(dataset)}")

    feature_names = [name for name, _ in graph_input_fields]
    model = GNNFFModel(
        graph_embedder=GraphEmbedding(
            graph_input_dimensions,
            [field.vocabulary for _, field in graph_input_fields],  # type: ignore
        ),
        graph_encoder=GGNN(
            iterations=model_encoder_iterations,
            n_types=len(graph_field[1].vocabulary),
            x_dim=sum(graph_input_dimensions),
            h_dim=model_encoder_output_dim,
            m_dim=model_encoder_message_dim,
        ),
        class_projection=Linear(in_features=model_encoder_output_dim, out_features=2),
        graph_field_name=graph_field[0],
        feature_field_names=feature_names,
        label_field_name=label_field[0],
    )
    # The model needs a forward to be completely initialized.
    model(dataset[0])
    logger.info(f"Configured model {model}")

    optimizer = Adam(params=model.parameters(), lr=optimizer_learning_rate)
    scheduler = StepLR(
        optimizer=optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma
    )
    trainer = Trainer(
        instance=instance,
        epochs=trainer_epochs,
        batch_size=trainer_batch_size,
        eval_every=trainer_eval_every,
        train_eval_split=trainer_train_eval_split,
        metric_names=trainer_metric_names,
        run_dir_path=output_dir_path,
        dataset=dataset,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        cuda_device=trainer_cuda,
    )
    trainer.train()
