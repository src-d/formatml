from typing import Iterable, List, NamedTuple

from torch import cat, long as torch_long, Tensor, tensor

from formatml.data.fields.field import Field
from formatml.data.fields.graph_fields.graph_field import GraphField
from formatml.parsing.parser import Nodes
from formatml.resources.vocabulary import Vocabulary
from formatml.utils.registrable import register


class RolesFieldOutput(NamedTuple):
    """Output of the roles field."""

    input: Tensor
    offsets: Tensor


@register(cls=Field, name="roles")
class RolesField(GraphField[RolesFieldOutput]):
    def __init__(self, vocabulary: Vocabulary) -> None:
        self.vocabulary = vocabulary

    def pre_tensorize(self, sample: Nodes) -> None:
        for node in sample.nodes:
            self.vocabulary.add_items(node.roles)

    def tensorize(self, sample: Nodes) -> RolesFieldOutput:
        roles_offsets = []
        roles: List[int] = []
        for node in sample.nodes:
            roles_offsets.append(len(roles))
            roles.extend(self.vocabulary.get_indexes(node.roles))
        return RolesFieldOutput(
            input=tensor(roles, dtype=torch_long),
            offsets=tensor(roles_offsets, dtype=torch_long),
        )

    def collate(self, tensors: Iterable[RolesFieldOutput]) -> RolesFieldOutput:
        tensors = list(tensors)
        offset = 0
        shifted_offsets = []
        for t in tensors:
            shifted_offsets.append(t.offsets + offset)
            offset += t.input.shape[0]
        return RolesFieldOutput(
            input=cat([t.input for t in tensors], dim=0),
            offsets=cat(shifted_offsets, dim=0),
        )
