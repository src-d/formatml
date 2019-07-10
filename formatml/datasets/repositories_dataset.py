from bisect import bisect_right
from logging import getLogger
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from typing import Any, Dict, Iterable, List, Sized, Tuple


from formatml.data.instance import Instance
from formatml.datasets.dataset import Dataset
from formatml.datasets.repository_dataset import RepositoryDataset
from formatml.parsing.parser import Parser


class RepositoriesDataset(Dataset):

    _logger = getLogger(__name__)

    def __init__(
        self,
        download_dir: str,
        parse_dir: str,
        tensor_dir: str,
        repositories: List[Tuple[str, str, str]],
        instance: Instance,
        parser: Parser,
        parallel_downloads: int = 10,
        bblfsh_endpoint: str = "0.0.0.0:9999",
        formatting_internal_type: str = "Formatting",
        n_workers: int = cpu_count(),
        pickle_protocol: int = 4,
    ) -> None:
        # Downloading parameters.
        self.parallel_downloads = parallel_downloads
        # Tensorizing parameters.
        self.instance = instance
        self.n_workers = n_workers
        self._repositories = [
            RepositoryDataset(
                root_download_dir=download_dir,
                root_parse_dir=parse_dir,
                root_tensor_dir=tensor_dir,
                user_name=user_name,
                repo_name=repo_name,
                version=version,
                instance=instance,
                parser=parser,
                bblfsh_endpoint=bblfsh_endpoint,
                formatting_internal_type=formatting_internal_type,
                n_workers=n_workers,
                pickle_protocol=pickle_protocol,
            )
            for user_name, repo_name, version in repositories
        ]

    def download(self) -> None:
        with ThreadPool(self.parallel_downloads) as pool:
            pool.map(self._download_repository, self._repositories)

    def pre_tensorize(self) -> None:
        for repository in self._repositories:
            repository.pre_tensorize()

    def tensorize(self) -> None:
        for repository in self._repositories:
            repository.tensorize()
        self._cumulative_lengths = self._compute_cumulative_lengths(self._repositories)

    def collate(self, tensors: List[Dict[str, Any]]) -> Dict[str, Any]:
        return self.instance.collate(tensors)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx
        dataset_idx = bisect_right(self._cumulative_lengths, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self._cumulative_lengths[dataset_idx - 1]
        return self._repositories[dataset_idx][sample_idx]

    def __len__(self) -> int:
        if not hasattr(self, "_cumulative_lengths"):
            raise ValueError("The dataset is not tensorized, its length is unknown")
        return self._cumulative_lengths[-1]

    @staticmethod
    def _compute_cumulative_lengths(sequences: Iterable[Sized]) -> List[int]:
        cumulative_lengths, current = [], 0
        for sequence in sequences:
            current += len(sequence)
            cumulative_lengths.append(current)
        return cumulative_lengths

    @staticmethod
    def _download_repository(repository: RepositoryDataset) -> None:
        repository.download()
