from typing import Iterable

from torch import (
    cat as torch_cat,
    device as torch_device,
    long as torch_long,
    Tensor,
    tensor,
)

from formatml.data.fields.field import Field
from formatml.data.vocabulary import Vocabulary
from formatml.parsing.parser import Nodes


class InternalTypeField(Field[Nodes, Tensor]):
    def __init__(self, name: str, type: str) -> None:
        super().__init__(name, type)
        self.vocabulary = Vocabulary(unknown="<UNK>")

    def index(self, sample: Nodes) -> None:
        for node in sample.nodes:
            self.vocabulary.add_item(node.internal_type)

    def tensorize(self, sample: Nodes) -> Tensor:
        return tensor(
            self.vocabulary.get_indexes(node.internal_type for node in sample.nodes),
            dtype=torch_long,
        )

    def collate(self, tensors: Iterable[Tensor]) -> Tensor:
        return torch_cat(tensors=list(tensors), dim=0)

    def to(self, tensor: Tensor, device: torch_device) -> Tensor:
        return tensor.to(device)
