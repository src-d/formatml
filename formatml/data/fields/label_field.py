from typing import Iterable, List, NamedTuple, Tuple

from torch import device as torch_device, long as torch_long, Tensor, tensor
from torch.nn.utils.rnn import pack_sequence, PackedSequence

from formatml.data.fields.field import Field
from formatml.data.vocabulary import Vocabulary
from formatml.parsing.parser import FORMATTING_INTERNAL_TYPE, Nodes
from formatml.utils.torch_helpers import unpack_packed_sequence


class LabelFieldOutput(NamedTuple):
    """Output of the label field."""

    indexes: Tensor
    decoder_inputs: PackedSequence
    labels: PackedSequence
    n_nodes: int


class LabelField(Field[Nodes, LabelFieldOutput]):
    def __init__(self, name: str, type: str) -> None:
        super().__init__(name, type)
        self.vocabulary = Vocabulary(unknown="<UNK>")
        self.vocabulary.add_item("<PAD>")
        self.vocabulary.add_item("<GO>")
        self.vocabulary.add_item("<STOP>")

    def index(self, sample: Nodes) -> None:
        for node in sample.nodes:
            if node.internal_type == FORMATTING_INTERNAL_TYPE:
                self.vocabulary.add_items(
                    list(node.token if node.token is not None else "")
                )

    def tensorize(self, sample: Nodes) -> LabelFieldOutput:
        node_sequences = []
        for i, node in enumerate(sample.nodes):
            if node.internal_type == FORMATTING_INTERNAL_TYPE:
                mapped = self.vocabulary.get_indexes(
                    list(node.token if node.token else "")
                )
                labels = tensor(
                    mapped + [self.vocabulary.get_index("<STOP>")], dtype=torch_long
                )
                decoder_inputs = tensor(
                    [self.vocabulary.get_index("<GO>")] + mapped, dtype=torch_long
                )
                node_sequences.append((i, decoder_inputs, labels))
        node_sequences.sort(reverse=True, key=lambda s: s[1].shape[0])
        indexes, decoder_inputs_tensor, labels_tensor = map(list, zip(*node_sequences))
        assert len(indexes) == len(decoder_inputs_tensor) and len(indexes) == len(
            labels_tensor
        )
        return LabelFieldOutput(
            indexes=tensor(indexes, dtype=torch_long),
            decoder_inputs=pack_sequence(decoder_inputs_tensor),
            labels=pack_sequence(labels_tensor),
            n_nodes=len(sample.nodes),
        )

    def collate(self, tensors: Iterable[LabelFieldOutput]) -> LabelFieldOutput:
        inputs_list: List[Tuple[int, Tensor, Tensor]] = []
        offset = 0
        for t in tensors:
            for indexes, decoder_inputs, labels in zip(
                (t.indexes + offset).tolist(),
                unpack_packed_sequence(t.decoder_inputs),
                unpack_packed_sequence(t.labels),
            ):
                inputs_list.append((indexes, decoder_inputs, labels))
            offset += t.n_nodes
        inputs_list.sort(reverse=True, key=lambda t: t[1].shape[0])
        indexes, decoder_inputs_tensor, labels_tensor = map(list, zip(*inputs_list))
        return LabelFieldOutput(
            indexes=tensor(indexes, dtype=torch_long),
            decoder_inputs=pack_sequence(decoder_inputs_tensor),
            labels=pack_sequence(labels_tensor),
            n_nodes=offset,
        )

    def to(self, tensor: LabelFieldOutput, device: torch_device) -> LabelFieldOutput:
        return LabelFieldOutput(
            indexes=tensor.indexes.to(device),
            decoder_inputs=tensor.decoder_inputs.to(device),
            labels=tensor.labels.to(device),
            n_nodes=tensor.n_nodes,
        )
