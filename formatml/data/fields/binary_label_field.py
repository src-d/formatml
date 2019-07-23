from typing import Iterable

from torch import cat, device as torch_device, long as torch_long, Tensor, zeros

from formatml.data.fields.field import Field
from formatml.data.types.codrep_label import CodRepLabel


class BinaryLabelsField(Field[CodRepLabel, Tensor]):
    def tensorize(self, inputs: CodRepLabel) -> Tensor:
        labels = zeros(inputs.n_formatting_nodes, dtype=torch_long)
        labels[inputs.error_index] = 1
        return labels

    def collate(self, tensors: Iterable[Tensor]) -> Tensor:
        return cat(list(tensors))

    def to(self, tensor: Tensor, device: torch_device) -> Tensor:
        return tensor.to(device)
