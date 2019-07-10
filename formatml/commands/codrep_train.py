from argparse import ArgumentParser
from copy import deepcopy
from json import dump as json_dump
from logging import getLogger
from pathlib import Path
from time import strftime
from typing import Iterator, List, NamedTuple, Tuple

from coloredlogs import install as coloredlogs_install
from torch.nn import Linear
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from formatml.commands.command import register_command
from formatml.data.fields.binary_label_field import BinaryLabelsField
from formatml.data.fields.field import Field
from formatml.data.fields.graph_fields.internal_type_field import InternalTypeField
from formatml.data.fields.graph_fields.length_field import LengthField
from formatml.data.fields.graph_fields.roles_field import RolesField
from formatml.data.fields.graph_fields.typed_dgl_graph_field import TypedDGLGraphField
from formatml.data.instance import Instance
from formatml.datasets.codrep_dataset import CodRepDataset
from formatml.models.gnn_ff import GNNFFModel
from formatml.modules.graph_encoders.ggnn import GGNN
from formatml.modules.misc.graph_embedding import GraphEmbedding
from formatml.parsing.java_parser import JavaParser
from formatml.parsing.parser import Nodes
from formatml.training.trainer import Trainer
from formatml.utils.helpers import get_sha_and_dirtiness


def add_arguments_to_parser(parser: ArgumentParser) -> None:
    parser.add_argument(
        "dataset_input_dir",
        metavar="dataset-input-dir",
        help="Path to the CodRep 2019 dataset.",
    )
    parser.add_argument(
        "--run-dir",
        help="Directory where the run artifacts will be output "
        "(defaults to %(default)s).",
        default=strftime("runs/codrep/%m-%d-%H:%M:%S%z"),
    )
    parser.add_argument(
        "--dataset-cache-dir",
        help="Where to cache dataset preprocessing (defaults to %(default)s).",
        default="cache-codrep",
    )
    parser.add_argument(
        "--encoder-edge-types",
        help="Edge types to use in the graph encoder (defaults to %(default)s).",
        nargs="+",
        default=["child", "parent", "previous_token", "next_token"],
    )
    parser.add_argument(
        "--max-length",
        help="Maximum token length to consider before clipping "
        "(defaults to %(default)s).",
        type=int,
        default=128,
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
        default=10,
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
        default=2000,
    )
    parser.add_argument(
        "--trainer-train-eval-split",
        help="Proportion kept for training (defaults to %(default)s). "
        "Rest goes to evaluation.",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--trainer-metric-names",
        help="Names of the metrics to use (defaults to %(default)s).",
        nargs="+",
        default=["mrr", "cross_entropy"],
    )
    parser.add_argument("--log-level", default="DEBUG", help="Logging verbosity.")


class FieldsDeclaration(NamedTuple):
    graph: Tuple[str, Field]
    label: Tuple[str, Field]
    graph_inputs: List[Tuple[str, Field, int]]

    def items(self) -> Iterator[Tuple[str, Field]]:
        yield self.graph
        yield self.label
        yield from (field_info[0:2] for field_info in self.graph_inputs)


@register_command(
    name="codrep-train",
    description="Train a model for the CodRep 2019 competition.",
    parser_definer=add_arguments_to_parser,
)
def codrep_train(
    *,
    dataset_input_dir: str,
    run_dir: str,
    dataset_cache_dir: str,
    encoder_edge_types: List[str],
    max_length: int,
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
    log_level: str,
) -> None:
    """Run the training."""
    args = deepcopy(locals())
    coloredlogs_install(level=log_level, fmt="%(name)27s %(levelname)8s %(message)s")
    logger = getLogger(__name__)
    git_info = get_sha_and_dirtiness()
    if git_info is None:
        args["git_info"] = None
    else:
        sha, dirty = git_info
        args["git_info"] = {"sha": sha, "dirty": dirty}
    run_dir_path = Path(run_dir).expanduser().resolve()
    run_dir_path.mkdir(parents=True, exist_ok=True)
    with (run_dir_path / "config.json").open(mode="w", encoding="utf8") as fh:
        json_dump(args, fh, indent=2, sort_keys=True)

    dataset_cache_dir_path = Path(dataset_cache_dir).expanduser().resolve()
    dataset_input_dir_path = Path(dataset_input_dir).expanduser().resolve()

    instance: Instance[Nodes] = Instance.load_or_create(
        fields=[
            ("typed_dgl_graph", TypedDGLGraphField(edge_types=encoder_edge_types)),
            ("label", BinaryLabelsField()),
            ("internal_type", InternalTypeField()),
            ("roles", RolesField()),
            ("length", LengthField(max_length=max_length)),
        ],
        cache_dir=dataset_cache_dir_path,
    )
    graph_field = instance.fields[0]
    label_field = instance.fields[1]
    graph_input_fields = instance.fields[2:]
    graph_input_dimensions = [48, 48, 32]

    dataset = CodRepDataset(
        parse_dir=dataset_cache_dir_path / "parse",
        tensor_dir=dataset_cache_dir_path / "tensor",
        input_dir=dataset_input_dir_path,
        instance=instance,
        parser=JavaParser(split_formatting=True),
    )
    dataset.download()
    dataset.pre_tensorize()
    dataset.tensorize()
    instance.save()
    logger.info(f"Dataset of size {len(dataset)}")

    feature_names = [name for name, _ in graph_input_fields]
    model = GNNFFModel(
        graph_embedder=GraphEmbedding(
            graph_input_dimensions,
            [field.vocabulary for _, field in graph_input_fields],  # type: ignore
        ),
        graph_encoder=GGNN(
            iterations=model_encoder_iterations,
            n_types=len(encoder_edge_types),
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
        epochs=trainer_epochs,
        batch_size=trainer_batch_size,
        eval_every=trainer_eval_every,
        train_eval_split=trainer_train_eval_split,
        metric_names=trainer_metric_names,
        run_dir_path=run_dir_path,
        dataset=dataset,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    trainer.train()
