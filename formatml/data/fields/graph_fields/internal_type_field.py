from typing import Iterable

from torch import cat as torch_cat, long as torch_long, Tensor, tensor

from formatml.data.fields.field import Field
from formatml.data.fields.graph_fields.graph_field import GraphField
from formatml.parsing.parser import Nodes
from formatml.resources.vocabulary import Vocabulary
from formatml.utils.registrable import register


@register(cls=Field, name="internal_type")
class InternalTypeField(GraphField[Tensor]):
    def __init__(self, vocabulary: Vocabulary) -> None:
        self.vocabulary = vocabulary

    def pre_tensorize(self, sample: Nodes) -> None:
        for node in sample.nodes:
            self.vocabulary.add_item(node.internal_type)

    def tensorize(self, sample: Nodes) -> Tensor:
        return tensor(
            self.vocabulary.get_indexes(node.internal_type for node in sample.nodes),
            dtype=torch_long,
        )

    def collate(self, tensors: Iterable[Tensor]) -> Tensor:
        return torch_cat(tensors=list(tensors), dim=0)
