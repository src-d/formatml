from bz2 import open as bz2_open
from logging import getLogger
from multiprocessing import cpu_count
from multiprocessing.pool import Pool
from pathlib import Path
from pickle import dump, load
from time import time
from typing import Any, Dict, List, NamedTuple, Optional

from asdf import AsdfFile, open as asdf_open
from numpy import array, uint32

from formatml.data.instance import Instance
from formatml.datasets.dataset import Dataset
from formatml.parsing.parser import Nodes, Parser, ParsingException


class CodRepLabel(NamedTuple):
    formatting_indexes: List[int]
    error_index: Optional[int]
    n_nodes: int

    def to_tree(self) -> Dict[str, Any]:
        return dict(
            formatting_indexes=array(self.formatting_indexes, dtype=uint32),
            error_index=self.error_index,
            n_nodes=self.n_nodes,
        )

    @staticmethod
    def from_tree(tree: Dict[str, Any]) -> "CodRepLabel":
        return CodRepLabel(
            formatting_indexes=tree["formatting_indexes"].tolist(),
            error_index=tree["error_index"],
            n_nodes=tree["n_nodes"],
        )


class CodRepDataset(Dataset):

    _logger = getLogger(__name__)

    def __init__(
        self,
        *,
        input_dir: Path,
        parse_dir: Path,
        tensor_dir: Path,
        instance: Instance,
        parser: Parser,
        n_workers: int = cpu_count(),
        pickle_protocol: int = 4,
        pre_tensorize: bool = False,
        tensorize: bool = False,
    ) -> None:
        # Output directories.
        self.input_dir = input_dir
        self.parse_dir = parse_dir
        self.tensor_dir = tensor_dir
        # Tensorizing parameters.
        self.instance = instance
        self.parser = parser
        self.n_workers = n_workers
        self.pickle_protocol = pickle_protocol
        if pre_tensorize:
            self.pre_tensorize()
            if tensorize:
                self.tensorize()

    def download(self) -> None:
        pass

    def pre_tensorize(self) -> None:
        self._logger.info(f"Parsing {self.input_dir.name}")
        self.parse_dir.mkdir(parents=True, exist_ok=True)
        if (self.parse_dir / "finished").is_file():
            return
        error_offsets = {}
        for i, line in enumerate(
            (self.input_dir / "out.txt").open("r", encoding="utf8")
        ):
            error_offsets[f"{i}.txt"] = int(line) - 1
        for file_path in self.input_dir.rglob("*.txt"):
            if file_path.name == "out.txt":
                continue
            file_path_relative = file_path.relative_to(self.input_dir)
            try:
                start = time()
                self._logger.debug(f"Parsing {file_path_relative}")
                nodes = self.parser.parse(self.input_dir, file_path_relative)
                self._logger.debug(
                    f"Parsed  {file_path_relative} "
                    f"into {len(nodes.nodes)} nodes "
                    f"in {(time() - start) * 1000:.2f}ms"
                )
                token_indexes = set(nodes.token_indexes)
                error_offset = error_offsets[file_path.name]
                error_node = None
                for i, node in enumerate(nodes.nodes):
                    if i not in token_indexes:
                        continue
                    if node.start == error_offset:
                        assert (
                            node.internal_type == self.parser.formatting_internal_type
                        )
                        error_node = node
                        break
                else:
                    for i, node in list(enumerate(nodes.nodes)):
                        if i not in token_indexes:
                            continue
                        if node.start <= error_offset < node.end:
                            self._logger.warning(
                                "Could not retrieve a formatting node for the "
                                f"error at offset {error_offset} of file "
                                f"{file_path.with_suffix('').name}. Retrieved {node} "
                                "instead."
                            )
                            break
                formatting_indexes = []
                j = 0
                for i, node in enumerate(nodes.nodes):
                    if node.internal_type == self.parser.formatting_internal_type:
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
            output_subdirectory = self.parse_dir / "asdf" / file_path_relative.parent
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
        (self.parse_dir / "finished").touch()

    def tensorize(self) -> None:
        try:
            # Cannot use multiprocessing if parser is an attribute of self.
            # Hence this hack with try, finally and a backup/restore of self.parser.
            parser = self.parser
            del self.parser
            finished_marker = self.tensor_dir / "finished"
            if not finished_marker.is_file():
                self._logger.info(f"Pre-tensorizing {self.input_dir}")
                for file_path in self.parse_dir.rglob("*.asdf"):
                    with asdf_open(str(file_path)) as af:
                        nodes_instance = Nodes.from_tree(af.tree["nodes"])
                        codrep_label = af.tree["codrep_label"]
                        self.instance.pre_tensorize(
                            {Nodes: nodes_instance, CodRepLabel: codrep_label}
                        )
                self._logger.info(f"Pre-tensorized  {self.input_dir}")
                self._logger.info(f"Tensorizing {self.input_dir}")
                with Pool(self.n_workers) as pool:
                    pool.map(
                        self._tensorize_worker,
                        (
                            p.relative_to(self.parse_dir)
                            for p in self.parse_dir.rglob("*.asdf")
                        ),
                    )
                self._logger.info(f"Tensorized  {self.input_dir}")
            self.pickles = list(
                sorted((self.tensor_dir / "pickle").rglob("*.pickle.bz2"), key=str)
            )
            finished_marker.touch()
        finally:
            self.parser = parser

    def collate(self, tensors: List[Dict[str, Any]]) -> Dict[str, Any]:
        return self.instance.collate(tensors)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        with bz2_open(self.pickles[index], "rb") as fh:
            return load(fh)

    def __len__(self) -> int:
        return len(self.pickles)

    def _tensorize_worker(self, file_path: Path) -> None:
        self._logger.debug(f"Tensorizing {file_path}")
        with asdf_open(str(self.parse_dir / file_path)) as af:
            nodes_instance = Nodes.from_tree(af.tree["nodes"])
            codrep_label = CodRepLabel.from_tree(af.tree["codrep_label"])
        tensors = self.instance.tensorize(
            {Nodes: nodes_instance, CodRepLabel: codrep_label}
        )
        output_dir = (self.tensor_dir / "pickle" / file_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        with bz2_open(
            (output_dir / file_path.name).with_suffix(".pickle.bz2"), "wb"
        ) as fh:
            dump(tensors, fh, protocol=self.pickle_protocol)
        self._logger.debug(f"Tensorized  {file_path}")
