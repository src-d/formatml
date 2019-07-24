from argparse import ArgumentParser
from bz2 import open as bz2_open
from json import load as json_load
from pathlib import Path
from pickle import load as pickle_load

from torch import load as torch_load, no_grad
from torch.utils.data import DataLoader

from formatml.datasets.codrep_dataset import CodRepDataset
from formatml.pipelines.codrep.cli_helper import CLIHelper
from formatml.pipelines.codrep.parse import parse
from formatml.pipelines.codrep.tensorize import tensorize
from formatml.pipelines.codrep.train import build_model
from formatml.pipelines.pipeline import register_step
from formatml.utils.config import Config
from formatml.utils.helpers import setup_logging


def add_arguments_to_parser(parser: ArgumentParser) -> None:
    cli_helper = CLIHelper(parser)
    cli_helper.add_raw_dir()
    cli_helper.add_uasts_dir()
    cli_helper.add_instance_file()
    cli_helper.add_tensors_dir()
    parser.add_argument(
        "--checkpoint_file", required=True, help="Path to the model checkpoint."
    )
    cli_helper.add_configs_dir()
    parser.add_argument(
        "--training-configs-dir",
        required=True,
        help="Path to the configs used for training.",
    )
    cli_helper.add_log_level()


@register_step(pipeline_name="codrep", parser_definer=add_arguments_to_parser)
def run(
    *,
    raw_dir: str,
    uasts_dir: str,
    instance_file: str,
    tensors_dir: str,
    checkpoint_file: str,
    configs_dir: str,
    training_configs_dir: str,
    log_level: str,
) -> None:
    """Run the model and output CodRep predictions."""
    arguments = locals()
    configs_dir_path = Path(configs_dir).expanduser().resolve()
    configs_dir_path.mkdir(parents=True, exist_ok=True)
    training_configs_dir_path = Path(training_configs_dir).expanduser().resolve()
    tensors_dir_path = Path(tensors_dir).expanduser().resolve()
    Config.from_arguments(
        arguments, ["instance_file", "checkpoint_file"], "configs_dir"
    ).save(configs_dir_path / "train.json")
    logger = setup_logging(__name__, log_level)

    training_configs = {}
    for step in ["parse", "tensorize", "train"]:
        with (training_configs_dir_path / step).with_suffix(".json").open(
            "r", encoding="utf8"
        ) as fh:
            training_configs[step] = json_load(fh)

    parse(
        raw_dir=raw_dir,
        uasts_dir=uasts_dir,
        configs_dir=configs_dir,
        log_level=log_level,
    )

    tensorize(
        uasts_dir=uasts_dir,
        instance_file=instance_file,
        tensors_dir=tensors_dir,
        configs_dir=configs_dir,
        n_workers=training_configs["tensorize"]["options"]["n_workers"],
        pickle_protocol=training_configs["tensorize"]["options"]["pickle_protocol"],
        log_level=log_level,
    )

    dataset = CodRepDataset(input_dir=tensors_dir_path)
    logger.info(f"Dataset of size {len(dataset)}")

    with bz2_open(instance_file, "rb") as fh:
        instance = pickle_load(fh)

    model = build_model(
        instance=instance,
        model_encoder_iterations=training_configs["train"]["options"][
            "model_encoder_iterations"
        ],
        model_encoder_output_dim=training_configs["train"]["options"][
            "model_encoder_output_dim"
        ],
        model_encoder_message_dim=training_configs["train"]["options"][
            "model_encoder_message_dim"
        ],
    )
    # The model needs a forward to be completely initialized.
    model(dataset[0])
    logger.info(f"Configured model {model}")

    model.load_state_dict(torch_load(checkpoint_file)["model_state_dict"])
    model.eval()
    logger.info(f"Loaded model parameters from %s", checkpoint_file)

    dataloader = DataLoader(
        dataset,
        shuffle=False,
        collate_fn=instance.collate,
        batch_size=10,
        num_workers=1,
    )

    with no_grad():
        for sample in dataloader:
            sample = model(sample)
            model.decode(sample)
