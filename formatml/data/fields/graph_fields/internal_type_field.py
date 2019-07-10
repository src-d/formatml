from typing import Iterable

from torch import cat as torch_cat, long as torch_long, Tensor, tensor

from formatml.data.fields.graph_fields.graph_field import GraphField
from formatml.data.vocabulary import Vocabulary
from formatml.parsing.parser import Nodes


class InternalTypeField(GraphField[Tensor]):
    def __init__(self) -> None:
        self.vocabulary = Vocabulary(unknown="<UNK>")

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
