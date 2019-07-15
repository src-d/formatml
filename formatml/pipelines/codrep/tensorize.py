from argparse import ArgumentParser
from bz2 import open as bz2_open
from functools import partial
from logging import getLogger, Logger
from multiprocessing import cpu_count
from multiprocessing.pool import Pool
from pathlib import Path
from pickle import dump as pickle_dump, load as pickle_load

from asdf import open as asdf_open

from formatml.data.instance import Instance
from formatml.data.types.codrep_label import CodRepLabel
from formatml.parsing.parser import Nodes
from formatml.pipelines.pipeline import register_step
from formatml.utils.helpers import setup_logging


def add_arguments_to_parser(parser: ArgumentParser) -> None:
    parser.add_argument("uasts_dir", metavar="uasts-dir", help="Path to the UASTs.")
    parser.add_argument(
        "instance_file", metavar="instance-file", help="Path to the pickled instance."
    )
    parser.add_argument(
        "output_dir", metavar="output-dir", help="Where to output the pickled tensors."
    )
    parser.add_argument(
        "--n-workers",
        help="Number of workers to use to tensorize the UASTs "
        "(defaults to %(default)s).",
        default=cpu_count(),
    )
    parser.add_argument(
        "--pickle-protocol",
        help="Pickle protocol to use (defaults to %(default)s).",
        default=4,
    )
    parser.add_argument("--log-level", default="DEBUG", help="Logging verbosity.")


@register_step(
    pipeline_name="codrep",
    step_name="tensorize",
    parser_definer=add_arguments_to_parser,
)
def tensorize(
    *,
    uasts_dir: str,
    instance_file: str,
    output_dir: str,
    n_workers: int,
    pickle_protocol: int,
    log_level: str,
) -> None:
    """Tensorize the UASTs."""
    setup_logging(log_level)
    logger = getLogger(__name__)

    uasts_dir_path = Path(uasts_dir).expanduser().resolve()
    output_dir_path = Path(output_dir).expanduser().resolve()

    with bz2_open(instance_file, "rb") as fh:
        instance = pickle_load(fh)

    worker = partial(
        _tensorize_worker,
        instance=instance,
        logger=logger,
        uasts_dir_path=uasts_dir_path,
        output_dir_path=output_dir_path,
        pickle_protocol=pickle_protocol,
    )

    logger.info(f"Tensorizing %s", uasts_dir_path)
    with Pool(n_workers) as pool:
        pool.map(
            worker,
            (p.relative_to(uasts_dir_path) for p in uasts_dir_path.rglob("*.asdf")),
        )
    logger.info(f"Tensorized  %s", uasts_dir_path)


def _tensorize_worker(
    file_path: Path,
    instance: Instance,
    logger: Logger,
    uasts_dir_path: Path,
    output_dir_path: Path,
    pickle_protocol: int,
) -> None:
    logger.debug(f"Tensorizing {file_path}")
    with asdf_open(str(uasts_dir_path / file_path)) as af:
        nodes_instance = Nodes.from_tree(af.tree["nodes"])
        codrep_label = CodRepLabel.from_tree(af.tree["codrep_label"])
    tensors = instance.tensorize({Nodes: nodes_instance, CodRepLabel: codrep_label})
    output_dir = (output_dir_path / file_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    with bz2_open((output_dir / file_path.name).with_suffix(".pickle.bz2"), "wb") as fh:
        pickle_dump(tensors, fh, protocol=pickle_protocol)
    logger.debug(f"Tensorized  {file_path}")
