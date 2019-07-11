from argparse import ArgumentParser
from logging import getLogger
from pathlib import Path
from time import time

from asdf import AsdfFile
from coloredlogs import install as coloredlogs_install

from formatml.data.types.codrep_label import CodRepLabel
from formatml.parsing.java_parser import JavaParser
from formatml.parsing.parser import ParsingException
from formatml.pipelines.pipeline import register_step


def add_arguments_to_parser(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--input-dir",
        help="Path to the CodRep 2019 formatted dataset (defaults to %(default)s).",
        default="codrep-data/raw",
    )
    parser.add_argument(
        "--output-dir",
        help="Where to output the UASTs (defaults to %(default)s).",
        default="codrep-data/uasts",
    )
    parser.add_argument("--log-level", default="DEBUG", help="Logging verbosity.")


@register_step(
    pipeline_name="codrep", step_name="parse", parser_definer=add_arguments_to_parser
)
def codrep_train(*, input_dir: str, output_dir: str, log_level: str) -> None:
    """Parse a CodRep 2019 dataset into UASTs."""
    coloredlogs_install(level=log_level, fmt="%(name)27s %(levelname)8s %(message)s")
    logger = getLogger(__name__)
    input_dir_path = Path(input_dir).expanduser().resolve()
    output_dir_path = Path(output_dir).expanduser().resolve()
    output_dir_path.mkdir(parents=True, exist_ok=True)

    parser = JavaParser(split_formatting=True)
    logger.info("Parsing %s", input_dir_path)
    error_offsets = {}
    for i, line in enumerate((input_dir_path / "out.txt").open("r", encoding="utf8")):
        error_offsets["%d.txt" % i] = int(line) - 1
    for file_path in input_dir_path.rglob("*.txt"):
        if file_path.name == "out.txt":
            continue
        file_path_relative = file_path.relative_to(input_dir_path)
        try:
            start = time()
            logger.debug("Parsing %s", file_path_relative)
            nodes = parser.parse(input_dir_path, file_path_relative)
            logger.debug(
                "Parsed  %s into %d nodes in %.2fms",
                file_path_relative,
                len(nodes.nodes),
                (time() - start) * 1000,
            )
            token_indexes = set(nodes.token_indexes)
            error_offset = error_offsets[file_path.name]
            error_node = None
            for i, node in enumerate(nodes.nodes):
                if i not in token_indexes:
                    continue
                if node.start == error_offset:
                    assert node.internal_type == parser.formatting_internal_type
                    error_node = node
                    break
            else:
                for i, node in list(enumerate(nodes.nodes)):
                    if i not in token_indexes:
                        continue
                    if node.start <= error_offset < node.end:
                        logger.warning(
                            "Could not retrieve a formatting node for the error at "
                            "offset %d of file %s. Retrieved %s instead.",
                            error_offset,
                            file_path.with_suffix("").name,
                            node,
                        )
                        break
            formatting_indexes = []
            j = 0
            for i, node in enumerate(nodes.nodes):
                if node.internal_type == parser.formatting_internal_type:
                    if node is error_node:
                        error_node_index = j
                    formatting_indexes.append(i)
                    j += 1
            codrep_label = CodRepLabel(
                formatting_indexes=formatting_indexes,
                error_index=error_node_index,
                n_nodes=len(nodes.nodes),
            )
        except ParsingException:
            continue
        output_subdirectory = output_dir_path / file_path_relative.parent
        output_subdirectory.mkdir(parents=True, exist_ok=True)
        with (output_subdirectory / file_path.with_suffix(".asdf").name).open(
            "wb"
        ) as fh:
            af = AsdfFile(
                dict(
                    nodes=nodes.to_tree(file_path.read_text(encoding="utf-8")),
                    codrep_label=codrep_label.to_tree(),
                )
            )
            af.write_to(fh, all_array_compression="bzp2")
