from typing import Any, List, Union

from torch import Tensor, tensor
from torch.nn.utils.rnn import PackedSequence


def unpack_packed_sequence(packed_sequence: PackedSequence) -> List[Tensor]:
    result: List[List[Any]] = [[] for _ in packed_sequence.batch_sizes]
    batch_sizes = packed_sequence.batch_sizes.clone()
    current = 0
    while batch_sizes[0] > 0:
        i = 0
        while i < len(batch_sizes) and batch_sizes[i] > 0:
            result[i].append(packed_sequence.data[current])
            current += 1
            batch_sizes[i] -= 1
            i += 1
    return [tensor(l, dtype=packed_sequence.data.dtype) for l in result]


def data_if_packed(input: Union[PackedSequence, Tensor]) -> Tensor:
    if isinstance(input, PackedSequence):
        return input.data
    return input
