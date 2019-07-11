from bz2 import open as bz2_open
from logging import getLogger
from multiprocessing import cpu_count
from multiprocessing.pool import Pool
from pathlib import Path
from pickle import dump, load
from tarfile import open as tarfile_open
from tempfile import TemporaryDirectory
from time import time
from typing import Any, Dict, List

from asdf import AsdfFile, open as asdf_open
from requests import get as requests_get

from formatml.data.instance import Instance
from formatml.datasets.dataset import Dataset
from formatml.parsing.parser import Nodes, Parser, ParsingException


class RepositoryDataset(Dataset):

    _logger = getLogger(__name__)

    def __init__(
        self,
        *,
        root_download_dir: str,
        root_parse_dir: str,
        root_tensor_dir: str,
        user_name: str,
        repo_name: str,
        version: str,
        instance: Instance,
        parser: Parser,
        bblfsh_endpoint: str = "0.0.0.0:9999",
        formatting_internal_type: str = "Formatting",
        n_workers: int = cpu_count(),
        pickle_protocol: int = 4,
    ) -> None:
        # Output directories.
        self.root_download_dir = Path(root_download_dir).expanduser().resolve()
        self.root_parse_dir = Path(root_parse_dir).expanduser().resolve()
        self.root_tensor_dir = Path(root_tensor_dir).expanduser().resolve()
        # Downloading parameters.
        self.user_name = user_name
        self.repo_name = repo_name
        self.version = version
        # Parsing parameters.
        self.bblfsh_endpoint = bblfsh_endpoint
        self.formatting_internal_type = formatting_internal_type
        # Tensorizing parameters.
        self.instance = instance
        self.parser = parser
        self.n_workers = n_workers
        self.pickle_protocol = pickle_protocol
        self.canonical_name = f"{self.user_name}-{self.repo_name}-{self.version}"
        self.download_path = self.root_download_dir / f"{self.canonical_name}.tar.gz"
        self.parse_dir = self.root_parse_dir / self.canonical_name
        self.tensor_dir = self.root_tensor_dir / self.canonical_name

    def download(self) -> None:
        self.root_download_dir.mkdir(parents=True, exist_ok=True)
        if self.download_path.is_file():
            return
        url = (
            f"https://github.com/{self.user_name}/{self.repo_name}/tarball/"
            f"{self.version}"
        )
        try:
            with requests_get(url, stream=True) as response:
                response.raise_for_status()
                with self.download_path.open("wb") as fh:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            fh.write(chunk)
                response.raise_for_status()
        except Exception as e:
            self.download_path.unlink()
            raise e

    def pre_tensorize(self) -> None:
        self._logger.info(f"Parsing {self.download_path.name}")
        self.parse_dir.mkdir(parents=True, exist_ok=True)
        if (self.parse_dir / "finished").is_file():
            return
        with TemporaryDirectory(prefix="formatml-") as temporary_directory_name:
            with tarfile_open(self.download_path) as tar:
                tar.extractall(path=temporary_directory_name)
            children_path = list(Path(temporary_directory_name).iterdir())
            assert len(children_path) == 1, "Expected a root directory only in the tar."
            repository_root_path = children_path[0].resolve(strict=True)
            for file_path in repository_root_path.rglob("*.js"):
                file_path_relative = file_path.relative_to(repository_root_path)
                try:
                    start = time()
                    self._logger.debug(f"Parsing {file_path_relative}")
                    nodes = self.parser.parse(repository_root_path, file_path_relative)
                    self._logger.debug(
                        f"Parsed  {file_path_relative} "
                        f"into {len(nodes.nodes)} nodes "
                        f"in {(time() - start) * 1000:.2f}ms"
                    )
                except ParsingException:
                    continue
                output_subdirectory = (
                    self.parse_dir / "asdf" / file_path_relative.parent
                )
                output_subdirectory.mkdir(parents=True, exist_ok=True)
                with (output_subdirectory / file_path.with_suffix(".asdf").name).open(
                    "wb"
                ) as fh:
                    af = AsdfFile(
                        dict(nodes=nodes.to_tree(file_path.read_text(encoding="utf-8")))
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
                self._logger.info(f"Pre-tensorizing {self.canonical_name}")
                for file_path in self.parse_dir.rglob("*.asdf"):
                    with asdf_open(str(file_path)) as af:
                        nodes_instance = Nodes.from_tree(af.tree["nodes"])
                        self.instance.index(nodes_instance)
                self._logger.info(f"Pre-tensorized  {self.canonical_name}")
                self._logger.info(f"Tensorizing {self.canonical_name}")
                with Pool(self.n_workers) as pool:
                    pool.map(
                        self._tensorize_worker,
                        [
                            p.relative_to(self.parse_dir)
                            for p in self.parse_dir.rglob("*.asdf")
                        ],
                    )
                self._logger.info(f"Tensorized  {self.canonical_name}")
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
        tensors = self.instance.tensorize(nodes_instance)
        output_dir = (self.tensor_dir / "pickle" / file_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        with bz2_open(
            (output_dir / file_path.name).with_suffix(".pickle.bz2"), "wb"
        ) as fh:
            dump(tensors, fh, protocol=self.pickle_protocol)
        self._logger.debug(f"Tensorized  {file_path}")
