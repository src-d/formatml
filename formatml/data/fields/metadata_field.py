from typing import Iterable, List, Tuple

from torch import device as torch_device

from formatml.data.fields.field import Field
from formatml.parsing.parser import Nodes


class MetadataField(Field[Tuple[str, Nodes], List[str]]):
    def __init__(self, *, name: str, type: str) -> None:
        super().__init__(name, type)

    def tensorize(self, sample: Tuple[str, Nodes]) -> List[str]:
        filepath, nodes = sample
        return [filepath]

    def collate(self, tensors: Iterable[List[str]]) -> List[str]:
        return [tensor[0] for tensor in tensors]

    def to(self, tensor: List[str], device: torch_device) -> List[str]:
        return tensor
