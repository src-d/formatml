from argparse import ArgumentParser
from logging import getLogger
from pathlib import Path
from typing import List

from asdf import open as asdf_open

from formatml.data.fields.binary_label_field import BinaryLabelsField
from formatml.data.fields.graph_fields.internal_type_field import InternalTypeField
from formatml.data.fields.graph_fields.length_field import LengthField
from formatml.data.fields.graph_fields.roles_field import RolesField
from formatml.data.fields.graph_fields.typed_dgl_graph_field import TypedDGLGraphField
from formatml.data.instance import Instance
from formatml.data.types.codrep_label import CodRepLabel
from formatml.parsing.parser import Nodes
from formatml.pipelines.pipeline import register_step
from formatml.utils.helpers import setup_logging


def add_arguments_to_parser(parser: ArgumentParser) -> None:
    parser.add_argument("input_dir", metavar="input-dir", help="Path to the UASTs.")
    parser.add_argument(
        "output_file",
        metavar="output-file",
        help="Where to output the pickled indexes.",
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
    parser.add_argument("--log-level", default="DEBUG", help="Logging verbosity.")


@register_step(
    pipeline_name="codrep", step_name="index", parser_definer=add_arguments_to_parser
)
def index(
    *,
    input_dir: str,
    output_file: str,
    encoder_edge_types: List[str],
    max_length: int,
    log_level: str,
) -> None:
    """Index UASTs with respect to some fields."""
    setup_logging(log_level)
    logger = getLogger(__name__)

    input_dir_path = Path(input_dir).expanduser().resolve()
    output_file_path = Path(output_file).expanduser().resolve()

    instance = Instance(
        fields=[
            ("typed_dgl_graph", TypedDGLGraphField(edge_types=encoder_edge_types)),
            ("label", BinaryLabelsField()),
            ("internal_type", InternalTypeField()),
            ("roles", RolesField()),
            ("length", LengthField(max_length=max_length)),
        ]
    )

    logger.info(f"Indexing %s", input_dir_path)
    for file_path in input_dir_path.rglob("*.asdf"):
        with asdf_open(str(file_path)) as af:
            nodes_instance = Nodes.from_tree(af.tree["nodes"])
            codrep_label = af.tree["codrep_label"]
            instance.index({Nodes: nodes_instance, CodRepLabel: codrep_label})
    instance.save(output_file_path)
    logger.info(f"Indexed  %s", input_dir_path)
