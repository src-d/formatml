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
from formatml.parsing.parser import FORMATTING_INTERNAL_TYPE, Node, Nodes


class LengthField(Field[Nodes, Tensor]):
    def __init__(self, *, name: str, type: str, max_length: int) -> None:
        super().__init__(name, type)
        self.max_length = max_length
        self.vocabulary: Vocabulary[int] = Vocabulary()
        self.vocabulary.add_items(range(self.max_length + 2))

    def tensorize(self, sample: Nodes) -> Tensor:
        return tensor([self._length(node) for node in sample.nodes], dtype=torch_long)

    def collate(self, tensors: Iterable[Tensor]) -> Tensor:
        return torch_cat(tensors=list(tensors), dim=0)

    def to(self, tensor: Tensor, device: torch_device) -> Tensor:
        return tensor.to(device)

    def _length(self, node: Node) -> int:
        if node.internal_type == FORMATTING_INTERNAL_TYPE:
            return self.max_length + 1
        return min(self.max_length, node.end - node.start)
