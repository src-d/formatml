from typing import Iterable, NamedTuple

from torch import cat, device as torch_device, long as torch_long, Tensor, tensor

from formatml.data.fields.field import Field
from formatml.parsing.parser import Nodes


class IndexesFieldOutput(NamedTuple):
    indexes: Tensor
    offsets: Tensor
    n_nodes: int


class IndexesField(Field[Nodes, IndexesFieldOutput]):
    def tensorize(self, inputs: Nodes) -> IndexesFieldOutput:
        return IndexesFieldOutput(
            indexes=tensor(inputs.formatting_indexes, dtype=torch_long),
            offsets=tensor(
                [inputs.nodes[i].start for i in inputs.formatting_indexes],
                dtype=torch_long,
            ),
            n_nodes=len(inputs.nodes),
        )

    def collate(self, tensors: Iterable[IndexesFieldOutput]) -> IndexesFieldOutput:
        tensors_list = list(tensors)
        offset_indexes = []
        offset = 0
        for t in tensors_list:
            offset_indexes.append(t.indexes + offset)
            offset += t.n_nodes
        return IndexesFieldOutput(
            indexes=cat(offset_indexes),
            offsets=cat([t.offsets for t in tensors_list]),
            n_nodes=sum(t.n_nodes for t in tensors_list),
        )

    def to(
        self, tensor: IndexesFieldOutput, device: torch_device
    ) -> IndexesFieldOutput:
        return IndexesFieldOutput(
            indexes=tensor.indexes.to(device),
            offsets=tensor.offsets.to(device),
            n_nodes=tensor.n_nodes,
        )
