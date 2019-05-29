from typing import Iterable

from torch import cat as torch_cat, long as torch_long, Tensor, tensor

from formatml.data.fields.field import Field
from formatml.data.fields.graph_fields.graph_field import GraphField
from formatml.parsing.parser import Node, Nodes
from formatml.resources.vocabulary import Vocabulary
from formatml.utils.registrable import register


@register(cls=Field, name="length")
class LengthField(GraphField[Tensor]):
    def __init__(
        self,
        *,
        max_length: int,
        formatting_internal_type: str = "Formatting",
        vocabulary: Vocabulary
    ) -> None:
        self.max_length = max_length
        self.formatting_internal_type = formatting_internal_type
        self.vocabulary = vocabulary
        vocabulary.add_items(range(max_length + 2))

    def pre_tensorize(self, sample: Nodes) -> None:
        for node in sample.nodes:
            self.vocabulary.add_item(node.internal_type)

    def tensorize(self, sample: Nodes) -> Tensor:
        return tensor([self._length(node) for node in sample.nodes], dtype=torch_long)

    def collate(self, tensors: Iterable[Tensor]) -> Tensor:
        return torch_cat(tensors=list(tensors), dim=0)

    def _length(self, node: Node) -> int:
        if node.internal_type == self.formatting_internal_type:
            return self.max_length + 1
        return min(self.max_length, node.end - node.start)
