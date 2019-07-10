from typing import Iterable, NamedTuple

from torch import cat, long as torch_long, Tensor, tensor, zeros

from formatml.data.fields.field import Field
from formatml.datasets.codrep_dataset import CodRepLabel


class BinaryLabelsFieldOutput(NamedTuple):
    """Output of the binary labels field."""

    indexes: Tensor
    labels: Tensor
    n_nodes: int


class BinaryLabelsField(Field[CodRepLabel, BinaryLabelsFieldOutput]):
    def tensorize(self, inputs: CodRepLabel) -> BinaryLabelsFieldOutput:
        indexes = tensor(inputs.formatting_indexes, dtype=torch_long)
        labels = zeros(indexes.size(), dtype=torch_long)
        labels[inputs.error_index] = 1
        return BinaryLabelsFieldOutput(
            indexes=indexes, labels=labels, n_nodes=inputs.n_nodes
        )

    def collate(
        self, tensors: Iterable[BinaryLabelsFieldOutput]
    ) -> BinaryLabelsFieldOutput:
        tensors_list = list(tensors)
        offset_indexes = []
        offset = 0
        for binary_labels_output in tensors_list:
            offset_indexes.append(binary_labels_output.indexes + offset)
            offset += binary_labels_output.n_nodes
        return BinaryLabelsFieldOutput(
            indexes=cat(offset_indexes),
            labels=cat([t.labels for t in tensors_list]),
            n_nodes=sum(t.n_nodes for t in tensors_list),
        )
