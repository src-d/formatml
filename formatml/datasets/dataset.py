from typing import Any, Dict, List

from torch.utils.data import ConcatDataset, Dataset as TorchDataset, Subset

from formatml.utils.from_params import from_params
from formatml.utils.registrable import register


@from_params
class Dataset(TorchDataset):
    def download(self) -> None:
        raise NotImplementedError()

    def pre_tensorize(self) -> None:
        raise NotImplementedError()

    def tensorize(self) -> None:
        raise NotImplementedError()

    def collate(self, tensors: List[Dict[str, Any]]) -> Dict[str, Any]:
        raise NotImplementedError()


register(cls=Dataset, name="concat")(ConcatDataset)
register(cls=Dataset, name="subset")(Subset)
