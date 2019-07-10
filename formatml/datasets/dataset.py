from typing import Any, Dict, List

from torch.utils.data import Dataset as TorchDataset


class Dataset(TorchDataset):
    def download(self) -> None:
        raise NotImplementedError()

    def pre_tensorize(self) -> None:
        raise NotImplementedError()

    def tensorize(self) -> None:
        raise NotImplementedError()

    def collate(self, tensors: List[Dict[str, Any]]) -> Dict[str, Any]:
        raise NotImplementedError()
