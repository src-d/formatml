from typing import Iterable

from torch import (
    cat as torch_cat,
    device as torch_device,
    long as torch_long,
    Tensor,
    tensor,
)

from formatml.data.fields.graph_fields.graph_field import GraphField
from formatml.data.vocabulary import Vocabulary
from formatml.parsing.parser import Node, Nodes


class LengthField(GraphField[Tensor]):
    def __init__(
        self, *, max_length: int, formatting_internal_type: str = "Formatting"
    ) -> None:
        self.max_length = max_length
        self.formatting_internal_type = formatting_internal_type
        self.vocabulary: Vocabulary[int] = Vocabulary()
        self.vocabulary.add_items(range(self.max_length + 2))

    def tensorize(self, sample: Nodes) -> Tensor:
        return tensor([self._length(node) for node in sample.nodes], dtype=torch_long)

    def collate(self, tensors: Iterable[Tensor]) -> Tensor:
        return torch_cat(tensors=list(tensors), dim=0)

    def to(self, tensor: Tensor, device: torch_device) -> Tensor:
        return tensor.to(device)

    def _length(self, node: Node) -> int:
        if node.internal_type == self.formatting_internal_type:
            return self.max_length + 1
        return min(self.max_length, node.end - node.start)
