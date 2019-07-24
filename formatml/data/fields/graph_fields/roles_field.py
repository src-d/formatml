from typing import Iterable, List, NamedTuple

from torch import cat, device as torch_device, long as torch_long, Tensor, tensor

from formatml.data.fields.graph_fields.graph_field import GraphField
from formatml.data.vocabulary import Vocabulary
from formatml.parsing.parser import Nodes


class RolesFieldOutput(NamedTuple):
    """Output of the roles field."""

    input: Tensor
    offsets: Tensor


class RolesField(GraphField[RolesFieldOutput]):
    def __init__(self, name: str, type: str) -> None:
        super().__init__(name, type)
        self.vocabulary = Vocabulary(unknown="<UNK>")

    def index(self, sample: Nodes) -> None:
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

    def to(self, tensor: RolesFieldOutput, device: torch_device) -> RolesFieldOutput:
        return RolesFieldOutput(
            input=tensor.input.to(device), offsets=tensor.offsets.to(device)
        )
